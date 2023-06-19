from pathlib import Path

########################## User Input Config ##########################
root_path = Path("D:/luke/Noddy_data")
train_path = root_path / "noddyverse_train_data"
val_path = root_path / "noddyverse_val_data"
preview_indices = [6, 13, 11, 12]  # , 24, 30, 38, 44, 46, 48]

pretrained_model_id: str = None  # will search for model file
train_from_epoch = -1  # specify epoch to train from, -1 for final

## Cometl.ml setup
# api_key and config recorded in .comet.config
tags = [root_path.stem]

## Torch config
manual_seed = 21
reproducibile_mode = True
use_amp = True

## Parameters
max_lr = 7e-5
num_epochs = 100
len_epochs = 200  # We have 1M samples, so we need to have an artificial number
shuffle = True  # Need to use a sampler for this number of samples.

num_workers = 6
pin_memory = True
trn_batch_size = 4
val_batch_size = 4  # len(preview_indices)

metric_freq = 100  # iterations
val_freq = 1  # epochs
preview_freq = 1  # epochs
checkpoint_freq = 100000  # iters # epochs
# preview_indices = preview_indices[:val_batch_size]  # Ensure previews included in Val

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
scheduler_spec = {
    "name": "msrl",
    "milestones": [500, 800, 900, 950],
    "gamma": 0.5,
}
# Network specifications
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
    "out_dim": encoder_spec["in_chans"],
    "hidden_list": [256, 256, 256],
}
lte_spec = {
    "hidden_dim": 256,
}
