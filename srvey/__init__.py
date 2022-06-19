""" SRvey 

Here I collect functions such as the Session handler, and set "run-time" 
options like seeds, deterministic mode, etc.
"""
print(f"Loading {__name__}")
from pathlib import Path
from datetime import datetime
import logging

from comet_ml import Experiment
import numpy as np
import torch
import time

import srvey.cfg as cfg

# Config


class Session:
    """Create folder structure, create comet experiment"""

    def __init__(self, debug=False):
        self.t0 = time.perf_counter()
        np.seterr(all="raise")
        torch.manual_seed(cfg.manual_seed)
        torch.backends.cudnn.benchmark = not cfg.reproducibile_mode
        torch.use_deterministic_algorithms(cfg.reproducibile_mode)
        self.device = torch.device("cuda")

        self.experiment = Experiment(disabled=debug)

        root = Path()
        self.session_id = (
            "debug" if debug else "Session_" + datetime.now().strftime("%y%m%d-%H%M")
        )
        self.session_dir = (
            root
            / "experiments"
            / (
                "debug"
                if debug
                else f"{cfg.pretrained_model or self.experiment.get_key()}"
            )
        )
        if cfg.pretrained_model:
            self.session_dir = (
                self.session_dir / f"continued_in_{self.experiment.get_key()}"
            )

        self.session_dir.mkdir(exist_ok=True, parents=True)

        self._init_comet()
        self._init_log()

    def _init_comet(self):
        self.experiment.add_tags(cfg.tags)
        self.experiment.log_parameters(
            {
                "HR tile size": cfg.hr_size,
                "Preview indices": cfg.preview_indices,
                "Seed": cfg.manual_seed,
                "AMP enabled": cfg.use_amp,
                "Benchmark mode": not cfg.reproducibile_mode,
                "Deterministic mode": cfg.reproducibile_mode,
                "Max LR": cfg.max_lr,
                "Load discriminator weights": cfg.load_d_weights,
                "Number of epochs": cfg.num_epochs,
                "Batch size Train": cfg.trn_batch_size,
                "Batch size Validation": cfg.val_batch_size,
                # "Iterations per epoch": iters_per_epoch,
                "Validation frequency": cfg.val_freq,
                "Preview frequency": cfg.preview_freq,
                "Checkpoint frequency": cfg.checkpoint_freq,
                "Number of DataLoader workers": cfg.num_workers,
                "LR scheduler": cfg.scheduler_type,
            }
        )

    def _init_log(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filename=f"{self.session_dir / 'session.log'}",
            filemode="w",
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-6s: %(levelname)-8s %(message)s"
        )
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

        self.train_log = logging.getLogger("train")
        self.train_log.info("Initialised logging")

    def begin_epoch(self, epoch):
        """Hook for beginning an epoch"""
        self.train_log.info(f"Beginning epoch {epoch}")
        self.experiment.set_epoch(epoch)
        self.epoch = epoch

    def end(self):
        self.experiment.end()
        self.train_log.info(f"Finished in {time.perf_counter() - self.t0} seconds.")
