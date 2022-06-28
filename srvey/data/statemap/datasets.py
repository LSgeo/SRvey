import logging
from pathlib import Path


import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import cfg
from mlnoddy.datasets import HRLRNoddyDataset


class BaseDataset(Dataset):
    def __init__(self, tile_path, augment: bool = False):
        super().__init__()

        self.data = {}
        self.augment = augment

        self.tile_path = Path(tile_path)

        # self._get_npy_data()
        self._get_tif_data()

        print(self._len)

    def _get_npy_data(self, search="**/*.npy"):
        for f in Path(self.tile_path).glob(search):
            print(f.stem)
            self.data[f"{f.stem}"] = np.load(f, mmap_mode="c").astype(np.float32)
            if "hr" in f.stem.lower():
                self.hr_key = f"{f.stem}"
                self._len = len(self.data[f"{f.stem}"])
            if "lr" in f.stem.lower():
                self.lr_key = f"{f.stem}"
        logging.getLogger("data").info(f"Found {self.data.keys()} in {self.tile_path}")

    def _get_tif_data(self, search="**/*.tif"):
        self.hr_key = "hr"
        self.lr_key = "lr"
        self.data[self.hr_key] = []
        self.data[self.lr_key] = []

        for f in Path(self.tile_path).glob(search):
            self.data[f"{f.parts[-2].lower()}"].append(
                tifffile.imread(f).astype(np.float32)
            )

        self._len = len(self.data[self.hr_key])
        logging.getLogger("data").info(
            f"Found {self._len} files in {self.data.keys()} in {self.tile_path}"
        )

    def _min_max_norm(self, tile):
        return (tile - self.min) / (self.max - self.min)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        tile_hr = self.data[self.hr_key][index].astype(np.float32)
        tile_lr = self.data[self.lr_key][index].astype(np.float32)
        tile_hr = torch.as_tensor(tile_hr, dtype=torch.float32).unsqueeze(0)
        tile_lr = torch.as_tensor(tile_lr, dtype=torch.float32).unsqueeze(0)

        if self.augment:
            if torch.rand(1) < 0.5:
                tile_hr = tile_hr.flip(-1)  # hflip
                tile_lr = tile_lr.flip(-1)
            if torch.rand(1) < 0.5:
                tile_hr = tile_hr.flip(-2)  # vflip
                tile_lr = tile_lr.flip(-2)
            if torch.rand(1) < 0.5:
                tile_hr = tile_hr.rot90(1, [-2, -1])  # rotate CHW Tensor CCW
                tile_lr = tile_lr.rot90(1, [-2, -1])  # Tensor is not batched yet

        return {"hr": tile_hr, "lr": tile_lr}


class MyDset(BaseDataset):
    """Quick tiff version of base dataset for ESRGAN RDN test with normalised data"""

    def __init__(self, tile_path, augment: bool = False):
        self.hr_key = "hr"
        self.scales = []
        self.set_scale((4, 4))

        super().__init__(tile_path, augment)

    def set_scale(self, scale: tuple):
        """Set scale factor to load next iteration"""
        if scale[0] != scale[1]:
            raise NotImplementedError
        self.curr_scale = scale[0]
        self.lr_key = "lr"


class ArbsrDset(BaseDataset):
    """Read multiple LR and HR data with different geophysical scales.

    These data are float32 single channel, and their resolution are fixed
    at n times different to "hr". 1 / [2, 2.5,  3,  4,  5]
    For tiles, this means HR 240, LR [120, 96, 80, 60, 48] pixels in each dim.
    Train, Val, and Test data are organised into folders appended by "x.y",
    indicating their decimal scale factor.

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

        # Old WA Tile Dataset
        # self.max = 14051.333
        # self.min = -4403.574
        # self.mean = -26.282
        # self.std = 224.668

        # # P738 LR training tiles, rounded to larger magnitude
        # self.max = 213
        # self.min = -315
        # self.mean = -38
        # self.std = 40

    def set_scale(self, scale: tuple):
        """Set scale factor to load next iteration"""
        if scale[0] != scale[1]:
            raise NotImplementedError
        self.curr_scale = scale[0]
        self.lr_key = f"{self.curr_scale:.1f}"

    def _get_npy_data(self, search="*.npy"):
        for f in Path(self.tile_path).glob(search):
            scale = f"{f.stem.split('_')[-1].replace('-', '.')}"
            self.data[scale] = np.load(f, mmap_mode="c").astype(np.float32)
            self.scales.append(scale)

        self._len = len(self.data["1.0"])

        if not self.tile_path.exists():
            raise FileNotFoundError(f"{self.tile_path} does not exist!")
        logging.getLogger("data").info(f"Found {self.data.keys()} in {self.tile_path}")
        if not self.data.get("1.0").any():
            raise FileNotFoundError(f"Could not load HR data from {self.tile_path}")
