########################## User Input Config ##########################
from pathlib import Path

debug = False
colab = False
use_comet = True and not debug

pretrained_model = None  # "93cc62f4e1dd42b0a29e86b6277a3c8f"
train_from_epoch = -1  # specify epoch to train from, -1 for final
# root_tiles_path = Path("/home/luke/srgeo/data_train/HR_20_LR_80_edge_padded")
root_tiles_path = Path("C:/Luke/data/Train_val_test_WA/HR_20m_LR_80m_train_val_test")
train_tiles_path = root_tiles_path / "train"
val_tiles_path = root_tiles_path / "val"
hr_size = 128
preview_indices = [44, 38, 24, 399, 457, 200, 158, 156]

## Torch Reproducibility
manual_seed = 21
use_amp = False  # TODO
cudnn_benchmark = True  # False for final
cudnn_deterministic = False  # True for final

## Cometl.ml setup
api_key = "s369wWTzEgIV5VwVg9vVtzeq7"
project_name = "srvey"
workspace = "lukesmithgeo"
tags = []

## Parameters
max_lr = 3e-4
load_d_weights = False
iters_per_epoch = 1000
num_epochs = 5
trn_batch_size = 2
val_batch_size = 2
val_freq = 100
# preview_indices = preview_indices[:val_batch_size]  # Ensure previews included in Val

num_workers = 0  # 0 on windows platforms, until bugfixed!
scheduler_type = "oclr"  # "oclr" or "mslr" or None
lr_scheduler_opts = {
    "max_lr": max_lr,
    "total_steps": iters_per_epoch * num_epochs,
    "pct_start": 0.3,
}


########################### Computed Config ###########################
from datetime import datetime

import numpy as np
import torch

# Config
np.seterr(all="raise")
device = torch.device("cuda")
torch.manual_seed(manual_seed)
torch.backends.cudnn.benchmark = cudnn_benchmark
torch.backends.cudnn.deterministic = cudnn_deterministic

root = Path()


class Session:
    """Create folder structure, create comet experiment"""

    def __init__(self):
        self.session_id = (
            "Session_" + datetime.now().strftime("%y%m%d-%H%M")
            if not debug
            else "debug"
        )
        self.session_dir = root / "experiments" / self.session_id
        self.session_dir.mkdir(exist_ok=True, parents=True)

    def end(self):
        print("Finished.")
