########################## User Input Config ##########################
from pathlib import Path

debug = False
colab = False
use_comet = True and not debug

pretrained_model = None  # "93cc62f4e1dd42b0a29e86b6277a3c8f"
train_from_epoch = -1  # specify epoch to train from, -1 for final
# root_tiles_path = Path("/home/luke/srgeo/data_train/HR_20_LR_80_edge_padded")
root_tiles_path = Path("/home/luke/srgeo/data_train/HR_20m_LR_80m_train_val_test/srvey")
train_tiles_path = root_tiles_path / "train"
val_tiles_path = root_tiles_path / "val"
hr_size = 128
preview_indices = [0, 1, 2, 3]

## Torch Reproducibility
manual_seed = 21
use_amp = True  # TODO
cudnn_benchmark = True  # False for final
cudnn_deterministic = False  # True for final

## Cometl.ml setup
# api_key and config recorded in .comet.config
tags = []

## Parameters
max_lr = 3e-4
load_d_weights = False
iters_per_epoch = 5000
num_epochs = 5
trn_batch_size = 4
val_batch_size = 4
val_freq = 500
preview_freq = 1000
# preview_indices = preview_indices[:val_batch_size]  # Ensure previews included in Val

num_workers = 0  # 0 on windows platforms, until bugfixed!
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
        self.experiment.get_key()
        self.session_id = (
            "Session_" + datetime.now().strftime("%y%m%d")  # -%H%M
            if not debug
            else "debug"
        )
        self.session_dir = (
            root / "experiments" / f"{self.session_id}_{self.experiment.get_key()}"
        )
        self.session_dir.mkdir(exist_ok=True, parents=True)

        self._init_log()

    def _init_log(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filename=f"{self.session_dir / 'session.log'}",
            filemode="w",
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)-6s: %(levelname)-8s %(message)s")
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
