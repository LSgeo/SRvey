########################## User Input Config ##########################
from pathlib import Path

debug = False
colab = False
use_comet = True and not debug

pretrained_model = None
train_from_epoch = -1  # specify epoch to train from, -1 for final
# root_tiles_path = Path("/home/luke/srgeo/data_train/HR_20_LR_80_edge_padded")
root_tiles_path = Path("D:/luke/SRvey/data_train/P738")
train_tiles_path = root_tiles_path / "train"
val_tiles_path = root_tiles_path / "val"
hr_size = 256
# preview_indices = [44, 38, 24, 399, 457, 200, 158, 156]
preview_indices = range(6)  # [6, 18, 20, 23, 24, 30, 38, 44, 46, 48]

## Torch config
manual_seed = 21
use_amp = True
cudnn_benchmark = False  # False for reproducibility
cudnn_deterministic = True  # True for reproducibility

## Cometl.ml setup
# api_key and config recorded in .comet.config
tags = [root_tiles_path.stem]

## Parameters
max_lr = 2e-4
load_d_weights = False
num_epochs = 40000
trn_batch_size = 16
val_batch_size = len(preview_indices)
iters_per_epoch = 64
val_freq = 50  # epochs
preview_freq = 2000
checkpoint_freq = 4000
# preview_indices = preview_indices[:val_batch_size]  # Ensure previews included in Val

num_workers = 0  # 0 on windows platforms, until bugfixed! Maybe?
scheduler_type = "oclr"  # "oclr" or "mslr" or None

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
torch.backends.cudnn.benchmark = cudnn_benchmark
torch.backends.cudnn.deterministic = cudnn_deterministic

root = Path()
t0 = time.perf_counter()


class Session:
    """Create folder structure, create comet experiment"""

    def __init__(self):
        self.experiment = Experiment()
        self.session_id = (
            "Session_" + datetime.now().strftime("%y%m%d-%H%M")
            if not debug
            else "debug"
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
                "Preview tiles": preview_indices,
                "Seed": manual_seed,
                "AMP enabled": use_amp,
                "Benchmark mode": cudnn_benchmark,
                "Deterministic mode": cudnn_deterministic,
                "Max LR": max_lr,
                "Load discriminator weights": load_d_weights,
                "Number of epochs": num_epochs,
                "Batch size Train": trn_batch_size,
                "Batch size Validation": val_batch_size,
                "Iterations per epoch": iters_per_epoch,
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

    def begin_epoch(self, epoch):
        """Hook for beginning an epoch"""
        self.train_log.info(f"Beginning epoch {epoch}")
        self.experiment.set_epoch(epoch)

    def end(self):
        self.experiment.end()
        self.train_log.info("Finished.")
