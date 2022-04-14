import functools
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import blocks as B
import cfg


class BaseModel(nn.Module):
    """Initiate a model class with generic methods"""

    def __init__(self, session):
        super().__init__()

        self.session = session
        self.exp = session.experiment
        self.model = ArbRDNPlus_net  ## for method linting, this is overwritten

        self.device = cfg.device
        self.use_amp = cfg.use_amp and "cuda" in cfg.device.type
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        self.num_epochs = cfg.num_epochs
        self.start_epoch = 0
        self.curr_epoch = -1  # iterate at start of loop, start 0
        self.curr_iteration = -1

        self.model_out_path = session.session_dir / "model"
        self.model_out_path.mkdir(exist_ok=True, parents=True)

        self.loss_dict = {}
        self.metric_dict = {}

    def save_model(self, name=None, for_inference_only: bool = True):
        """Save model state for inference / continuation

        Args:
             save_full_model: Include optimisers in order to continue
             training from checkpoint.
        """

        if for_inference_only:
            filename = name or f"inference_model_{self.curr_iteration}.pth"
            torch.save(
                self.model.state_dict(),
                self.model_out_path / filename,
            )
            # logging.info(f"Saved model for inference to {self.model_out_path}")
        else:
            filename = name or f"full_model_epoch_{self.curr_epoch:05}.tar"
            torch.save(
                {
                    # "comet_experiment": self.experiment.get_key(),
                    "iteration": self.curr_iteration,
                    "epoch": self.curr_epoch,
                    "net_g": self.model.state_dict(),
                    # "net_d": self.discriminator.state_dict(),
                    "optimiser_g": self.optimiser_g.state_dict(),
                    # "optimiser_d": self.optimiser_d.state_dict(),
                    "scheduler_g": self.scheduler_g.state_dict(),  # TODO
                    # "scheduler_d": self.scheduler_d.state_dict(), TODO these may be matched
                },
                self.model_out_path / filename,
            )
            # logging.info(f"Saved model and training state to {self.model_out_path}")

    def load_pretrained_model(self, pretrained_experiment, from_epoch=-1):
        """Load a model checkpoint for continuation or inference

        model_pth: Experiment ID of previous experiment, to search in local
        experiments directory.
        """
        try:
            saved_model_dir = next(Path().glob(f"**/*{pretrained_experiment}*"))
            saved_model_path = sorted(list(saved_model_dir.glob("**/*.tar")))[
                from_epoch
            ]
            logging.info(
                f"Loading experiment {pretrained_experiment}, epoch: {from_epoch}"
            )
            # from_epoch = saved_model_path.as_posix()[-9:-4]
        except:
            raise FileNotFoundError(
                f"Files for experiment {pretrained_experiment} were not found on this device"
            )

        checkpoint = torch.load(saved_model_path, map_location=self.device)
        self.curr_iteration = checkpoint["iteration"]
        self.start_epoch = checkpoint["epoch"]
        self.num_epochs += cfg.num_epochs  # Add more epochs
        self.optimiser_g.load_state_dict(checkpoint["optimiser_g"])
        self.scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser_g,
            max_lr=cfg.max_lr,
            total_steps=cfg.iters_per_epoch * cfg.num_epochs,
        )  # Make new OCLR scheduler - this is like a warm restart? Maybe?
        # .load_state_dict(checkpoint["scheduler_g"]) # TODO add more iters to scheduler?
        self.model.load_state_dict(checkpoint["net_g"])
        self.model.to(self.device)  # TODO ? self.model = self.model.to....

    def feed_data(self, batch):
        """Load data"""
        self.batch = batch

    def train(self):
        """Train model on a single batch, per iteration"""
        self.model.train()
        self._forward()

        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimiser_g)
        self.scaler.update()
        self.optimiser_g.zero_grad(set_to_none=True)
        self.scheduler_g.step()
        self.loss_dict["Loss_Train_L1"] = self.loss_L1.item()
        self.loss_dict["Train_PSNR"] = self._calc_psnr()

    def train_discriminator(self):
        """Base method for training a discriminator for GAN networks"""
        raise NotImplementedError

    def validate(self):
        """Per iteration, calculate validation losses for model"""
        self.model.eval()
        with torch.no_grad():
            self._forward()

        self.loss_dict["Loss_Val_L1"] = self.loss_L1.item()
        self.loss_dict["Val_PSNR"] = self._calc_psnr()

    def _forward(self):
        """Apply model on data and calculate loss. Used in both train and val."""
        inp = self.batch["lr"].to(self.device)
        tgt = self.batch["hr"].to(self.device)

        with torch.autocast(self.device.type, enabled=self.use_amp):
            pred = self.model(inp)
            self.loss_L1 = self.cri_L1(pred, tgt) * self.cri_L1_w
            self.loss = self.loss_L1

            self.metric_mse = self.mse(pred, tgt)

    def _calc_psnr(self, value_range_squared=1.0):
        if self.metric_mse == 0:
            return float("inf")
        return 10 * np.log10(value_range_squared / self.metric_mse.item())

    def log_metrics(self, log_to_disk: bool = True, log_to_comet: bool = False):
        """Save metrics to Comet.ml, and/or log locally"""
        if log_to_comet:
            self.exp.set_step(self.curr_iteration)
            self.exp.log_metrics(self.loss_dict)
            self.exp.log_metric("Current LR", self.scheduler_g.get_last_lr())
            self.exp.log_metric(
                "Seconds per step",
                (time.perf_counter() - cfg.t0) / (self.curr_iteration + 2),
            )
        if log_to_disk:
            [
                logging.getLogger("train").info(
                    f"Iter: {self.curr_iteration:4d} {k}: {v:3f}"
                )
                for k, v in self.loss_dict.items()
            ]

        self.loss_dict.clear()  # Remove old keys

    def save_previews(self, log_to_disk: bool = True, log_to_comet: bool = False):
        """Convert current batch to images and log to comet.ml"""

        self.model.eval()  # ensure no training occurs
        with torch.no_grad():
            sr = self.model(self.batch["lr"].to(self.device, non_blocking=True))
            data = [["SR", sr.detach().cpu().numpy()]]
            if (
                self.curr_epoch == self.start_epoch
            ):  # Log LR and HR input and target once
                data.append(["LR", self.batch["lr"].detach().cpu().numpy()])
                data.append(["HR", self.batch["hr"].detach().cpu().numpy()])

        for i, (name, batch) in enumerate(data):  # For each resolution data
            v = 0  # Reset tile index for each Resolution batch
            for j, d in enumerate(batch):  # For each tensor data in the batch
                # if log_to_comet:
                self.exp.log_image(
                    (255 * (d - d.min()) / (d.max() - d.min())).astype(np.uint8),
                    name=f"Tile_{cfg.preview_indices[v]}_{name}",
                    image_scale=1,
                    step=self.curr_iteration,
                )
                v += 1  # Track tile within batch

    def forward(self, x):  # pass forward calls to model forward call
        return self.model(x)


class ArbRDNPlus(BaseModel):
    """Arbitray scale factor RDN-like model, as implemented in ESGRAN+

    Uses PyTorch CUDA amp.
    https://github.com/LongguangWang/ArbSR
    https://github.com/ncarraz/ESRGANplus
    """

    def __init__(self, session):
        super().__init__(session)
        self.scale = None
        self.model = ArbRDNPlus_net()
        self.model.to(self.device)

        self.optimiser_g = torch.optim.Adam(self.model.parameters(), lr=cfg.max_lr)
        self.scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser_g,
            max_lr=cfg.max_lr,
            total_steps=cfg.iters_per_epoch * cfg.num_epochs,
            pct_start=0.3,
        )

        self.mse = nn.MSELoss().to(self.device)  # Used for PSNR, not in .backward()
        self.cri_L1 = nn.L1Loss().to(self.device)
        self.cri_L1_w = 1.0

    def set_scale(self, scale: tuple):
        """Set scale factor for model"""
        # is_distributed = False #Not implemented
        # if is_distributed:
        #     return self.generator
        # else:
        #     return self.generator.module

        if scale[0] != scale[1]:
            raise NotImplementedError
        self.scale_h = self.scale_w = scale[0]
        self.model.scale_h = self.model.scale_w = scale[0]  # f"{scale[0]:.1f}"


class ArbRDNPlus_net(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nf=64,
        nb=23,
        gc=32,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        k=2,  # ArbSR Frequency of ArbSR upsampling
    ):
        super().__init__()

        self.k = k
        # https://github.com/LongguangWang/ArbSR/blob/master/model/arbrcan.py
        # We (LSgeo) use ArbitrarySR  to implement scale-arbitrary super-
        # resolution in RDN like network, i.e. allow choosing scale factor
        # at inference time.

        self.fea_conv = B.conv_block(
            in_nc, nf, kernel_size=3, norm_type=None, act_type=None
        )

        rrdb = B.RRDB(
            nf,
            kernel_size=3,
            gc=gc,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm_type,
            act_type=act_type,
            mode="CNA",
        )
        # We need to generate n sequences of k rrdb blocks...
        self.trunks = nn.ModuleList(
            [nn.Sequential(*[rrdb for _ in range(self.k)]) for _ in range(nb // self.k)]
        )
        # ...then ensure any remaining basic blocks are included...
        self.trunks.append(nn.Sequential(*[rrdb for _ in range(nb % self.k)]))
        # ...followed by an sa_adapt layer for each trunk block.
        self.sa_adapts = nn.ModuleList(
            [B.SA_adapt(64) for _ in range(len(self.trunks))]
        )
        self.LR_conv = B.conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )
        self.sa_upsample = B.SA_upsample(64)  # scale-aware final upsampling layer
        self.HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        self.HR_conv1 = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

    def forward(self, x):
        x0 = self.fea_conv(x)
        x_trunk = x0
        for i in range(len(self.trunks)):
            x_trunk = self.trunks[i](x_trunk)
            x_trunk = self.sa_adapts[i](x_trunk, self.scale_h, self.scale_w)
        x_trunk += x0  # ESRGANplus "shortcut block"
        x1 = self.sa_upsample(x_trunk, self.scale_h, self.scale_w)
        x2 = self.HR_conv0(x1)
        x3 = self.HR_conv1(x2)

        return x3


# def _weights_init_kaiming(m, scale=1):
#     """from https://github.com/ncarraz/ESRGANplus/blob/master/codes/models/networks.py#L30"""
#     classname = m.__class__.__name__
#     if (classname.find("Conv") != -1) or (classname.find("Linear") != -1):
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#         m.weight.data *= scale
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif classname.find("BatchNorm2d") != -1:
#         nn.init.constant_(m.weight.data, 1.0)
#         nn.init.constant_(m.bias.data, 0.0)


class RDNPlus(BaseModel):
    def __init__(self, session, total_steps):
        super().__init__(session)
        self.model = RDNPlus_net()
        self.model.to(self.device)

        self.optimiser_g = torch.optim.Adam(self.model.parameters(), lr=cfg.max_lr)
        self.scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser_g,
            max_lr=cfg.max_lr,
            total_steps=total_steps,
            # pct_start=0.3,
        )

        self.cri_L1 = nn.L1Loss().to(self.device)
        self.cri_L1_w = 1  # TODO confirm if its better to wrap as tensor / device
        self.mse = nn.MSELoss().to(self.device)  # Used for PSNR, not in .backward()

    def set_scale(self, *args, **kwargs):
        """Scale is fixed at 4"""
        self.scale_h = self.scale_w = self.model.scale_h = self.model.scale_w = 4


class RDNPlus_net(nn.Module):
    """RDN like model using RRDB blocks, from ESRGAN+"""

    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nf=64,
        nb=23,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
    ):
        super().__init__()

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        rb_blocks = [
            B.RRDB(
                nf,
                kernel_size=3,
                gc=gc,
                stride=1,
                bias=True,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
            )
            for _ in range(nb)
        ]

        LR_conv = B.conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = B.upconv_block
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode}] is not found")
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]

        HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        HR_conv1 = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        self.model = B.sequential(
            fea_conv,
            B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
        )

    def forward(self, x):
        x = self.model(x)
        return x
