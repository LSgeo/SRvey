import logging
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import cfg


class BaseDataset(Dataset):
    def __init__(self, tile_path, augment: bool = False):
        super().__init__()

        self.data = {}
        self.augment = augment

        self.tile_path = Path(tile_path)
        self.hr_size = cfg.hr_size

        self._get_npy_data()
        self._len = len(self.data[self.hr_key])

    def _get_npy_data(self, search="**/*.npy"):
        for f in Path(self.tile_path).glob(search):
            self.data[f"{f.stem}"] = np.load(f, mmap_mode="c").astype(np.float32)
            if "hr" in f.stem.lower():
                self.hr_key = f"{f.stem}"
            if "lr" in f.stem.lower():
                self.lr_key = f"{f.stem}"
        logging.getLogger("data").info(f"Found {self.data.keys()} in {self.tile_path}")

    def min_max_norm(self, tile):
        return (tile - self.min) / (self.max - self.min)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        tile_hr = self.data[self.hr_key][index].astype(np.float32)
        tile_lr = self.data[self.lr_key][index].astype(np.float32)

        tile_hr = self.min_max_norm(tile_hr)
        tile_lr = self.min_max_norm(tile_lr)

        tile_hr = torch.as_tensor(tile_hr, dtype=torch.float32)
        tile_lr = torch.as_tensor(tile_lr, dtype=torch.float32)

        if self.augment:
            if torch.rand(1) < 0.5:
                tile_hr = tile_hr.flip(-1)  # hflip
                tile_lr = tile_lr.flip(-1)
            if torch.rand(1) < 0.5:
                tile_hr = tile_hr.flip(-2)  # vflip
                tile_lr = tile_lr.flip(-2)
            if torch.rand(1) < 0.5:
                tile_hr = tile_hr.rot90(1, [1, 2])  # rotate CHW Tensor CCW
                tile_lr = tile_lr.rot90(1, [1, 2])  # Tensor is not batched yet

        return {"hr": tile_hr, "lr": tile_lr}


class ArbsrDset(BaseDataset):
    """Read multiple LR and HR data with different geophysical scales.

    These data are float32 single channel, and their resolution are fixed
    at n times different to "hr". 1 / [2, 2.5,  3,  4,  5]
    For tiles, this means HR 240, LR [120, 96, 80, 60, 48] pixels in each dim.
    Train, Val, and Test data are organised into folders appended by "xxx",
    indicating their resolution.

    We include all available file paths for each optioned scale in a dictionary,
    load the memmapped numpy arrays into another dictionary, and access them
    during training by specifying the desired scale in a method. It is important
    to ensure the values of each dict are sorted so matching indices refers to
    data from the same extent.

    """

    def __init__(self, tile_path, augment: bool = False):
        self.hr_key = "1.0"
        self.scales = []
        self.set_scale((4, 4))

        super().__init__(tile_path, augment)

        self.max = 14051.333
        self.min = -4403.574
        self.mean = -26.282
        self.std = 224.668

    def set_scale(self, scale: tuple):
        """Set scale factor to load next iteration"""
        if scale[0] != scale[1]:
            raise NotImplementedError
        self.curr_scale = scale[0]
        self.lr_key = f"{self.curr_scale:.1f}"

    def _get_npy_data(self, search="**/*.npy"):
        for f in Path(self.tile_path).glob(search):
            scale = f"{cfg.hr_size / int(f.stem.split('_')[-1]):.1f}"
            self.data[scale] = np.load(f, mmap_mode="c").astype(np.float32)
            self.scales.append(scale)

        logging.getLogger("data").info(f"Found {self.data.keys()} in {self.tile_path}")
        if not self.data["1.0"].any():
            raise FileNotFoundError(f"Could not load HR data from {self.tile_path}")


def build_dataloaders():
    """Returns dataloaders for Training, Validation, and image previews"""

    train_dataset = ArbsrDset(Path(cfg.train_tiles_path), augment=True)
    val_dataset = ArbsrDset(Path(cfg.val_tiles_path))
    preview_dataset = Subset(val_dataset, cfg.preview_indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.trn_batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    preview_dataloader = DataLoader(
        preview_dataset,
        batch_size=min(cfg.val_batch_size, len(preview_dataset)),
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader, preview_dataloader

