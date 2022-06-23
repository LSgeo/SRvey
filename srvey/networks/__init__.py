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


class BaseModel(nn.Module):
    def __init__(self, session: srvey.Session):
        super().__init__()
        self.session = session
        self.d = self.session.device
        self.exp = self.session.experiment
        self.num_epochs = cfg.num_epochs
        self.start_epoch = 0
        self.curr_epoch = 0
        self.curr_iteration = 0
        self.use_amp = cfg.use_amp and "cuda" in self.d.type
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        self.cri_mse = nn.MSELoss().to(self.d, non_blocking=True)
        self.cri_L1 = nn.L1Loss().to(self.d, non_blocking=True)
        self.psnr = None  # TODO psnr_func -
        # ? https://torchmetrics.readthedocs.io/en/stable/image/peak_signal_noise_ratio.html

        self.loss_dict = {}
        self.metric_dict = {}

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

    def train_on_batch(self):
        """Train model using CUDA AMP on fed batch"""
        self.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.d.type, enabled=self.use_amp):
            self.sr = self(self.lr, self.coord, self.cell)  # self *is* the model

            loss_L1 = self.cri_L1(self.sr, self.hr)
            # loss_mse = self.cri_mse(self.sr, self.hr)
            # self.metric_dict["PSNR"] = self.psnr(self.sr, self.hr)

            self.loss_dict["Train_L1"] = loss_L1.item()
            self.loss_dict["Train_MSE"] = self.cri_mse(self.sr, self.hr).item()

            self.scaler.scale(loss_L1).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if "oclr" in cfg.scheduler_spec["name"]:
                self.scheduler.step()

    def validate_on_batch(self):
        self.eval()
        with torch.no_grad():
            self.sr = self(self.lr, self.coord, self.cell)
            loss_L1 = self.cri_L1(self.sr, self.hr)
            self.loss_dict["Val_L1"] = loss_L1.item()
            self.loss_dict["Val_MSE"] = self.cri_mse(self.sr, self.hr).item()


    def save_previews(self, log_to_disk: bool = True, log_to_comet: bool = False):
        """Convert current batch to images and log to comet.ml"""
        self.validate_on_batch()
        data = [["SR", self.sr.detach().cpu().numpy()]]

        if (self.curr_epoch == self.start_epoch):  # Log LR and HR only once
            data.append(["LR", self.lr.detach().cpu().numpy()])
            data.append(["HR", self.hr.detach().cpu().numpy()])

        for name, batch in data:  # For each resolution data
            v = 0  # Reset tile index for each Resolution batch
            for grid in batch:  # For each tensor data in the batch
                # if log_to_comet:
                self.session.experiment.log_image(
                    (255 * (grid - grid.min()) / (grid.max() - grid.min())).astype(np.uint8),
                    name=f"Grid_{cfg.preview_indices[v]}_{name}",
                    image_scale=1,
                    step=self.curr_iteration,
                )
                v += 1  # Track tile within batch

    def log_metrics(self, log_to_disk: bool = True, log_to_comet: bool = True):
        """Save metrics to Comet.ml, and/or log locally"""
        if log_to_comet:
            self.exp.set_step(self.curr_iteration)
            self.exp.log_metrics(
                {**self.metric_dict, **self.loss_dict}
            )  # TODO Ensure unscaled
            self.exp.log_metric("Current LR", self.scheduler.get_last_lr())
            self.exp.log_metric(
                "Samples per second",
                (self.curr_iteration * cfg.trn_batch_size)
                / (time.perf_counter() - self.session.t0),
            )
        if log_to_disk:
            logging.getLogger("train").info(
                f"Iter {self.curr_iteration:4d}: "
                f"| {' | '.join([f'{k} {v:3f}' for k, v in self.loss_dict.items()])} |"
            )

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
