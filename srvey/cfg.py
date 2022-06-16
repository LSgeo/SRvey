from pathlib import Path

########################## User Input Config ##########################
pretrained_model = None
train_from_epoch = -1  # specify epoch to train from, -1 for final

root_tiles_path = Path("noddyverse_data")
train_tiles_path = root_tiles_path / "train"
val_tiles_path = root_tiles_path / "val"

dataset_kwargs = {
    "load_magnetics": True,
    "load_gravity": False,
    "load_geology": False,
    "augment": False,
    "scale": 2,
    "line_spacing": None,
    "sample_spacing": None,
    "heading": None,
}

hr_size = 200
preview_indices = range(4)  # [6, 18, 20, 23, 24, 30, 38, 44, 46, 48]

## Torch config
manual_seed = 21
use_amp = True
reproducibile_mode = True

## Cometl.ml setup
# api_key and config recorded in .comet.config
tags = [root_tiles_path.stem]

## Parameters
max_lr = 4e-4
load_d_weights = False
num_epochs = 10
shuffle = False  # Need to use a sampler for this number of samples.

trn_batch_size = 2
val_batch_size = len(preview_indices)

metric_freq = 10  # iterations
val_freq = 50  # epochs
preview_freq = 200  # epochs
checkpoint_freq = 2000  # epochs
# preview_indices = preview_indices[:val_batch_size]  # Ensure previews included in Val

num_workers = 0
scheduler_type = "mslr"  # "oclr" or "mslr" or None

########################### Computed Config ###########################
from datetime import datetime
import logging

from comet_ml import Experiment
import numpy as np
import torch
import time

# Config
np.seterr(all="raise")
device = torch.device("cuda")
torch.manual_seed(manual_seed)
torch.backends.cudnn.benchmark = not reproducibile_mode
torch.use_deterministic_algorithms(reproducibile_mode)

root = Path()
t0 = time.perf_counter()


class Session:
    """Create folder structure, create comet experiment"""

    def __init__(self, debug=False):
        self.experiment = Experiment()
        self.session_id = (
            "debug" if debug else "Session_" + datetime.now().strftime("%y%m%d-%H%M")
        )
        self.session_dir = (
            root
            / "experiments"
            / ("debug" if debug else f"{pretrained_model or self.experiment.get_key()}")
        )
        if pretrained_model:
            self.session_dir = (
                self.session_dir / f"continued_in_{self.experiment.get_key()}"
            )

        self.session_dir.mkdir(exist_ok=True, parents=True)

        self._init_comet()
        self._init_log()

    def _init_comet(self):
        self.experiment.add_tags(tags)
        self.experiment.log_parameters(
            {
                "HR tile size": hr_size,
                "Preview indices": preview_indices,
                "Seed": manual_seed,
                "AMP enabled": use_amp,
                "Benchmark mode": not reproducibile_mode,
                "Deterministic mode": reproducibile_mode,
                "Max LR": max_lr,
                "Load discriminator weights": load_d_weights,
                "Number of epochs": num_epochs,
                "Batch size Train": trn_batch_size,
                "Batch size Validation": val_batch_size,
                # "Iterations per epoch": iters_per_epoch,
                "Validation frequency": val_freq,
                "Preview frequency": preview_freq,
                "Checkpoint frequency": checkpoint_freq,
                "Number of DataLoader workers": num_workers,
                "LR scheduler": scheduler_type,
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

    def end(self):
        self.experiment.end()
        self.train_log.info(f"Finished in {time.perf_counter() - t0} seconds.")
