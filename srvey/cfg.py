print(f"Loading {__name__}")
from pathlib import Path

########################## User Input Config ##########################
root_tiles_path = Path("noddyverse_data")
train_tiles_path = root_tiles_path / "train"
val_tiles_path = root_tiles_path / "val"
preview_indices = range(4)  # [6, 18, 20, 23, 24, 30, 38, 44, 46, 48]

pretrained_model = None

dataset_config = {
    "load_magnetics": True,
    "load_gravity": False,
    "load_geology": False,
    "augment": False,
    "scale": 2,
    "line_spacing": None,
    "sample_spacing": None,
    "heading": None,
}

# LTE options
encoder_spec = {
    # "name": "swinir",
    "img_size": 48,
    "in_chans": 1,
    "upscale": 2,
    "no_upsampling": True,
}
imnet_spec = {
    # "name": "mlp",
    "in_dim": 256,
    "out_dim": 3,
    "hidden_list": [256, 256, 256],
    "hidden_dim": 256,
}

## Cometl.ml setup
# api_key and config recorded in .comet.config
tags = [root_tiles_path.stem]

## Torch config
manual_seed = 21
use_amp = True
reproducibile_mode = True

## Parameters
train_from_epoch = -1  # specify epoch to train from, -1 for final
hr_size = 200

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
