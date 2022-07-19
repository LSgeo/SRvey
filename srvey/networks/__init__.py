""" srvey.networks

Here I collect network agnostic (but still network) code. 
"""


import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR

import srvey
import srvey.cfg as cfg
import mlnoddy.datasets


class BaseModel(nn.Module):
    def __init__(self, session: srvey.Session):
        super().__init__()
        self.session = session
        self.d = self.session.device
        self.exp = self.session.experiment
        self.num_epochs = cfg.num_epochs
        self.start_epoch = 1
        self.curr_epoch = 0
        self.curr_iteration = 0
        self.use_amp = cfg.use_amp and "cuda" in self.d.type
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        self.norm = mlnoddy.datasets.Norm(clip=5000).min_max_clip

        # self.cri_mse = nn.MSELoss().to(self.d, non_blocking=True)
        self.cri_L1 = nn.L1Loss().to(self.d, non_blocking=True)
        self.psnr = None  # TODO psnr_func -
        # ? https://torchmetrics.readthedocs.io/en/stable/image/peak_signal_noise_ratio.html

        self.loss_dict = {}
        # self.metric_dict = {}
        self.train_batches = 0
        self.val_batches = 0

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.max_lr)

    def _init_scheduler(self):
        if "msrl" in cfg.scheduler_spec["name"]:
            self.scheduler = MultiStepLR(
                self.optimizer,
                cfg.scheduler_spec["milestones"],
            )
        elif "oclr" in cfg.scheduler_spec["name"]:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=cfg.max_lr,
                total_steps=self.session.total_steps,
            )
        else:
            raise ValueError("Expected a valid scheduler spec in config")

    def load_pretrained_model(self, from_epoch=-1):
        saved_model_dir = next(Path().glob(f"**/*{cfg.pretrained_model_id}*"))
        saved_model_path = sorted(list(saved_model_dir.glob("**/*.tar")))
        if not saved_model_path:
            raise FileNotFoundError(
                f"No .tar saved experiments found in {saved_model_dir.absolute()}"
            )
        # logging.info(f"Loading experiment {cfg.pretrained_model_id}, epoch: {from_epoch}")
        checkpoint = torch.load(saved_model_path, map_location=self.d)
        self.curr_iteration = checkpoint["iteration"]
        self.start_epoch = checkpoint["epoch"]
        self.num_epochs += cfg.num_epochs  # Add more epochs
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )  # TODO add more iters to scheduler?
        self.load_state_dict(checkpoint["net_g"])
        self.to(self.d, non_blocking=True)

    def feed_data(self, batch):
        self.lr = batch["lr_grid"].to(self.d, non_blocking=True)
        self.hr = batch["hr_vals"].to(self.d, non_blocking=True)
        self.coord = batch["hr_coord"].to(self.d, non_blocking=True)
        self.cell = batch["hr_cell"].to(self.d, non_blocking=True)
        self.data_time = (
            batch["Sample processing time"] / self.lr.shape[0]
        )  # Stays on cpu.
        self.data_time = [self.data_time.mean(), self.data_time.std()]  # Stays on cpu.

    def train_on_batch(self):
        """Train model using Automatic mixed precision on fed batch"""
        self.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.d.type, enabled=self.use_amp):
            self.sr = self(self.lr, self.coord, self.cell)  # self *is* "the model"
            loss_L1 = self.cri_L1(self.sr, self.hr)
            # metric_mse = self.cri_mse(self.sr, self.hr)
            # metric_psnr =

        self.scaler.scale(loss_L1).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if "oclr" in cfg.scheduler_spec["name"]:
            self.scheduler.step()

        self.loss_dict["Train_L1"] = loss_L1.detach().cpu().numpy()
        self.train_batches += 1

    def validate_on_batch(self):
        if "Val_L1" not in self.loss_dict.keys():
            self.loss_dict["Val_L1"] = 0
            # self.loss_dict["Val_MSE"] = 0

        self.eval()
        with torch.no_grad():
            self.sr = self(self.lr, self.coord, self.cell)
            self.loss_dict["Val_L1"] += (
                self.cri_L1(self.sr, self.hr).detach().cpu().numpy()
            )
            # self.loss_dict["Val_MSE"] += self.cri_mse(self.sr, self.hr).item()
        self.val_batches += 1

    def save_previews(
        self, batch, log_to_disk: bool = True, log_to_comet: bool = False, num_ch=1
    ):
        """Convert current batch to images and log to comet.ml
        This is a similar process to validation, and could be included in that
        method. However, it is simple to do it here with a specifically selected
        dataset.

        """

        self.eval()
        with torch.no_grad():
            sr = (
                self(
                    batch["lr_grid"].to(self.d, non_blocking=True),
                    batch["hr_coord"].to(self.d, non_blocking=True),
                    batch["hr_cell"].to(self.d, non_blocking=True),
                )
                .detach()
                .cpu()
            )

        sr = reshape_coordinate_array(batch["lr_grid"], batch["hr_grid"], sr)
        data = {"SR": self.norm(sr, inverse=True).numpy()}

        if self.curr_epoch == self.start_epoch:  # Log LR and HR only once
            data["LR"] = batch["lr_grid"].cpu().numpy()
            data["HR"] = batch["hr_grid"].cpu().numpy()

            # For each resolution batch
            for name, res_batch in data.items():
                # For each tensor in the batch
                for i, grid in enumerate(res_batch):
                    if log_to_comet:
                        self.exp.log_image(
                            (
                                255 * (grid - grid.min()) / (grid.max() - grid.min())
                            ).astype(np.uint8),
                            name=f"Grid_{cfg.preview_indices[i]}_{name}",
                            image_scale=1,
                            step=self.curr_iteration,
                        )

                    if log_to_disk:
                        import matplotlib.pyplot as plt

                        plt.imshow(data["SR"][i, :, :, :])
                        plt.savefig(f"Preview_{i}.png")
                        plt.close()

    def log_metrics(self, log_to_disk: bool = True, log_to_comet: bool = True):
        """Save metrics to Comet.ml, and/or log locally"""

        if "Train_L1" in self.loss_dict.keys():
            value = self.loss_dict["Train_L1"] / self.train_batches  # Average
            self.loss_dict["Train_L1"] = value
        if "Val_L1" in self.loss_dict.keys():
            value = self.loss_dict["Val_L1"] / self.val_batches
            self.loss_dict["Val_L1"] = value
            # self.loss_dict["Val_MSE"] /= self.val_batches

        self.metric_dict = {
            "Current LR": self.scheduler.get_last_lr()[0],
            "Sample time Mean": self.data_time[0],
            "Sample time Std": self.data_time[1],
            "Samples per second": (
                (self.curr_iteration * cfg.trn_batch_size)
                / (time.perf_counter() - self.session.t1)
            ),
        }

        if log_to_comet:
            self.exp.set_step(self.curr_iteration)
            self.exp.log_metrics({**self.loss_dict, **self.metric_dict})
        if log_to_disk:
            logging.getLogger("Train").info(
                f"| Iter: {self.curr_iteration:5d} "
                f"| {' | '.join([f'{k:>10}: {v:0.2f}' for k, v in {**self.loss_dict, **self.metric_dict}.items()])} |"
            )  # Merge metrics and losses and print fixed width strings using | separator

        self.train_batches = 0  # reset average counter
        self.val_batches = 0
        self.loss_dict.clear()  # Remove old keys
        self.metric_dict.clear()

    def save_model(self, name: str = None, for_inference_only: bool = True):
        """Save model state for inference / continuation
        Args:
             save_full_model: Include optimisers in order to continue
             training from checkpoint.
        """

        if for_inference_only:
            filename = name or f"inference_model_{self.curr_iteration}.pth"
            torch.save(
                self.state_dict(),
                self.session.model_out_path / filename,
            )
            # logging.info(f"Saved model for inference to {self.model_out_path}")
        else:
            filename = name or f"full_model_epoch_{self.curr_epoch:05}.tar"
            torch.save(
                {
                    # "comet_experiment": self.experiment.get_key(),
                    "iteration": self.curr_iteration,
                    "epoch": self.curr_epoch,
                    "net_g": self.state_dict(),
                    # "net_d": self.discriminator.state_dict(),
                    "optimiser_g": self.optimizer.state_dict(),
                    # "optimiser_d": self.optimizer_d.state_dict(),
                    "scheduler_g": self.scheduler.state_dict(),  # TODO
                    # "scheduler_d": self.scheduler_d.state_dict(), TODO these may be matched
                },
                self.session.model_out_path / filename,
            )
            # logging.info(f"Saved model and training state to {self.model_out_path}")


def reshape_coordinate_array(lr_grid, hr_grid, sr_pred):
    """Reshape a coordinate array to match spatial dimensions

    Args:
        lr_grid: Batched, n channel, HxW array of LR grid
        hr_grid: Batched, n channel, HxW array of groundtruth grid (used for shape)
        hr_coord: Batched, n channel, 2xn_points of SR coordinates

    Remember, Pytorch is [B,C,H,W], while Numpy is [H,W,C].

    See https://github.com/jaewon-lee-b/lte/blob/main/test.py#L98
    """

    # if inp == "hr_vals":
    #     # gt reshape
    #     b, c, lrh, lrw = lr_grid.shape
    #     s = np.sqrt(c / (lrh * lrw))  # Assumes square, I guess
    #     shape = [b, round(lrh * s), round(lrw * s), c]
    #     hr_grid = hr_grid.view(*shape).permute(0, 3, 1, 2).contiguous()

    # prediction reshape
    # ih += h_pad
    # iw += w_pad
    b, c, hrh, hrw = hr_grid.shape
    # s = np.sqrt(c / (hrh * hrw))  # Assumes square, I guess
    shape = (b, hrh, hrw, c)
    sr_pred = sr_pred.reshape(shape)  # .permute(0, 3, 1, 2).contiguous()
    # sr_pred = sr_pred[..., :hrw, :hrh]

    return sr_pred
