print(f"Loading {__name__}")
from pathlib import Path

########################## User Input Config ##########################
root_tiles_path = Path(
    "C:\Luke\PhD\paper2\MLnoddy\DYKE_DYKE_DYKE\models_by_code\models\DYKE_DYKE_DYKE"
)
train_tiles_path = root_tiles_path  # / "train"
val_tiles_path = root_tiles_path / "val"
preview_indices = range(4)  # [6, 18, 20, 23, 24, 30, 38, 44, 46, 48]

pretrained_model_id: str = None  # will search for model file
train_from_epoch = -1  # specify epoch to train from, -1 for final

dataset_config = {
    "load_magnetics": True,
    "load_gravity": False,
    "load_geology": False,
    "augment": False,
    "scale": 2,
    "line_spacing": 5 * 20,
    "sample_spacing": 20,
    "heading": "NS",
}

# LTE options
encoder_spec = {
    # "name": "swinir",
    "img_size": 48,
    "in_chans": 1, # also sets out_chans
    "upscale": 2,
    "no_upsampling": True,
}
lte_spec = {
    "hidden_dim": 256,
}
imnet_spec = {
    # "name": "mlp",
    "in_dim": 256,
    "out_dim": 3,
    "hidden_list": [256, 256, 256],
}
scheduler_spec = {
    "name": "msrl",
    "milestones": [500, 800, 900, 950],
    "gamma": 0.5,
}

## Cometl.ml setup
# api_key and config recorded in .comet.config
tags = [root_tiles_path.stem]

## Torch config
manual_seed = 21
reproducibile_mode = True
use_amp = False

## Parameters
hr_size = 200

max_lr = 4e-4
num_epochs = 10
shuffle = False  # Need to use a sampler for this number of samples.

num_workers = 0
pin_memory = False
trn_batch_size = 1
val_batch_size = 1  # len(preview_indices)

metric_freq = 100  # iterations
val_freq = 1  # epochs
preview_freq = 1  # epochs
checkpoint_freq = 5  # epochs
# preview_indices = preview_indices[:val_batch_size]  # Ensure previews included in Val
