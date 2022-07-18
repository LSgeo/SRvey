import logging
import time

import numpy as np
import verde as vd
import torch
from torch.utils.data import DataLoader, Subset

import srvey.cfg as cfg
from mlnoddy.datasets import NoddyDataset


def build_dataloaders():
    """Returns dataloaders for Training, Validation, and image previews"""

    train_dataset = HRLRNoddyDataset(
        model_dir=cfg.train_tiles_path, **cfg.dataset_config
    )
    val_dataset = HRLRNoddyDataset(model_dir=cfg.val_tiles_path, **cfg.dataset_config)
    preview_dataset = Subset(val_dataset, cfg.preview_indices)

    logging.getLogger("Data").info(
        f"| Training samples: {len(train_dataset)}"
        f" | Validation samples: {len(val_dataset)}"
        f" | Number of previews: {len(preview_dataset)} |"
    )

    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=cfg.pin_memory,
        batch_size=cfg.trn_batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=bool(cfg.num_workers),
        shuffle=cfg.shuffle,
        drop_last=False,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        pin_memory=cfg.pin_memory,
        batch_size=cfg.val_batch_size,
        num_workers=0,
        # persistent_workers=bool(cfg.num_workers),
        shuffle=False,
    )
    preview_dataloader = DataLoader(
        preview_dataset,
        pin_memory=cfg.pin_memory,
        batch_size=min(cfg.val_batch_size, len(preview_dataset)),
    )

    return train_dataloader, validation_dataloader, preview_dataloader


def subsample(parameters: dict, scale=1, *rasters):
    input_cell_size = 20
    """Run a mock-survey on a geophysical raster.
    Designed for use with Noddy forward models, as part of a Pytorch dataset.

    Args:
        parameters:
            line_spacing: in meters, spacing between parallel lines
            sample_spacing: in meters, spacing between points along line
            heading: "NS" for columns as lines, "EW" for rows as lines
        scale: multiplier for low resolution (vs high resolution)
        input_cell_size: ground truth cell size, 20 m for Noddy models
        *rasters: input Tensor forward model

    The Noddyverse dataset is a suite of 1 Million 200x200x200 petrophysical
    voxels, at a designated size of 20 m per pixel. Forward models in the
    Noddyverse (https://doi.org/10.5194/essd-14-381-2022) are generated
    as per below:
        Geophysical forward models were calculated using a Fourier domain
        formulation using reflective padding to minimise (but not remove)
        boundary effects. The forward gravity and magnetic field calculations
        assume a flat top surface with a 100 m sensor elevation above this
        surface and the Earth's magnetic field with vertical inclination,
        zero declination and an intensity of 50000nT.

    Note that every single cell of the forward model has a calculated forward
    model, i.e. they are 200x200, with no interpolation (for a 20 m cell size)

    We simulate an airborne survey, by selecting rows (flight lines) of pixels
    at every n m. We can (but not by default) also subsample along rows (ss).

    """
    cs = input_cell_size
    ss = int(parameters.get("sample_spacing") / cs)
    ls = int(parameters.get("line_spacing") * scale / cs)

    if parameters.get("heading").upper() in ["EW", "E", "W"]:
        ls, ss = ss, ls  # swap convention to emulate survey direction

    x, y = np.meshgrid(np.arange(200), np.arange(200), indexing="xy")
    x = cs * x[::ls, ::ss]
    y = cs * y[::ls, ::ss]
    zs = [raster[::ls, ::ss] for raster in rasters]

    return x, y, zs


def grid(x, y, zs, ls: int = 20, cs_fac: int = 4):
    in_cs: int = 20
    """Grid a subsampled noddy forward model.

    params:
        x, y: x, y coordinates
        z: geophysical response value
        line_spacing: sample line spacing, to calculate target cell_size
        cs_fac: line spacing to cell size factor, typically 4 or 5
        name: data_variable name
        input_cell_size: Input model cell size, 20m for Noddyverse

    See docstring for subsample() for further notes.
    #TODO Grid the full extent, or crop to useful extent.
    """
    for rz in zs:
        gridder = vd.ScipyGridder("cubic", extra_args={"fill_value": 0}).fit((x, y), rz)

        yield gridder.grid(
            region=[0, in_cs * 200, 0, in_cs * 200],
            spacing=ls / cs_fac,
            dims=["x", "y"],
            data_names="forward",
        ).get("forward").values.astype(np.float32)


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers.

    https://github.com/jaewon-lee-b/lte/blob/main/utils.py#L104
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        return ret.view(-1, ret.shape[-1])
    else:
        return ret


def to_cell_samples(grid):
    """Convert the image to coord-*C pairs.
    grid: Tensor, (C, H, W)
    https://github.com/jaewon-lee-b/lte/blob/main/utils.py#L122

    Returns:
        coordinate array (shape = H*W, 2) and values in make_coord range
        c_vals: H*W list of tuples of channels value(s) at coordinate n

    i.e. return two lists describing the index and channel(s) value(s) at
         index of input grid.


    """
    if len(grid.shape) != 3:  # C,H,W, data are not batched yet.
        raise ValueError(f"Grid should be of shape C,H,W by now. Got {grid.shape}")
    coord = make_coord(grid.shape[-2:])
    c_vals = grid.view(grid.shape[-3], -1).permute(1, 0)
    return coord, c_vals


class HRLRNoddyDataset(NoddyDataset):
    def __init__(self, **kwargs):
        self.scale = kwargs.get("scale", 2)
        self.random_heading = not bool(kwargs.get("heading"))
        self.sp = {
            "line_spacing": kwargs.get("line_spacing", 20),
            "sample_spacing": kwargs.get("sample_spacing", 20),
            "heading": kwargs.get("heading", None),  # Default will be random
        }
        super().__init__(**kwargs)

    def _process(self, index):
        dt0 = time.perf_counter()
        super()._process(index)

        hls = self.sp["line_spacing"]
        lls = hls * self.scale
        if self.random_heading:
            if torch.rand(1) < 0.5:
                self.sp["heading"] = "NS"
            else:
                self.sp["heading"] = "EW"

        hr_x, hr_y, _hr_zs = subsample(self.sp, 1, *self.data["gt_grid"])
        lr_x, lr_y, _lr_zs = subsample(self.sp, self.scale, *self.data["gt_grid"])
        _hr_grids = [
            torch.from_numpy(g).unsqueeze(0) for g in grid(hr_x, hr_y, _hr_zs, ls=hls)
        ]
        _lr_grids = [
            torch.from_numpy(g).unsqueeze(0) for g in grid(lr_x, lr_y, _lr_zs, ls=lls)
        ]

        if cfg.dataset_config["load_magnetics"] and cfg.dataset_config["load_gravity"]:
            self.data["hr_grid"] = torch.stack(_hr_grids, dim=0)
            self.data["lr_grid"] = torch.stack(_lr_grids, dim=0)
            raise NotImplementedError("Haven't designed network for this yet")
        else:
            self.data["hr_grid"] = _hr_grids[0]
            self.data["lr_grid"] = _lr_grids[0]

        self.data["hr_coord"], self.data["hr_vals"] = to_cell_samples(
            self.data["hr_grid"].contiguous()
        )

        self.data["hr_cell"] = torch.ones_like(self.data["hr_coord"])
        self.data["hr_cell"][:, 0] *= 2 / self.data["hr_grid"].shape[-2]
        self.data["hr_cell"][:, 1] *= 2 / self.data["hr_grid"].shape[-1]

        self.data["Sample processing time"] = torch.tensor(
            time.perf_counter() - dt0, dtype=torch.float16
        ) 
         # used for metric

        # LTE Sample_q not implemented, but I think it subsamples random amount of pixels from the full suite
        # https://github.com/jaewon-lee-b/lte/blob/94bca2bf5777b76edbad46e899a1c2243e3751d4/datasets/wrappers.py#L64

        # # self.data at this point may look like (some optional):
        # {
        #     "label": encode_label(self.parent),
        #     "geo":   torch.from_numpy($2d_array_of_layer_n)),
        #     "gt_grid":    torch.stack(gt_data, dim=0),
        #     "lr_grid":    torch.stack(lr_grids, dim=0),
        #     "hr_grid":    torch.stack(hr_grids, dim=0),
        #     "hr_coord": coordinate array of for hr values
        #     "hr_cell":  array like hr_coord but = [[coord 2/h], [coord2/w]]

        #     "data_time": float of how long processing/dataset get took,
        # }
