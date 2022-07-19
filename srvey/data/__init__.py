import logging
import time

import numpy as np
import verde as vd
import torch
from torch.utils.data import DataLoader, Subset

import srvey.cfg as cfg
from mlnoddy.datasets import NoddyDataset

import matplotlib.pyplot as plt


def build_dataloaders():
    """Returns dataloaders for Training, Validation, and image previews"""

    train_dataset = HRLRNoddyDataset(model_dir=cfg.train_path, **cfg.dataset_config)
    val_dataset = HRLRNoddyDataset(model_dir=cfg.val_path, **cfg.dataset_config)
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


def line_sample(parameters: dict, raster):
    """Run a mock airborne survey on a raster.

    Sample every nth "line_spacing" lines.
    Designed for use with Noddy forward models, as part of a Pytorch
    dataset.

    Args:
        parameters:
            line_spacing: in indices, separation between parallel lines
            heading: "NS" for columns as lines, "EW" for rows as lines
        rasters: input Tensor forward model

    The Noddyverse dataset (https://doi.org/10.5194/essd-14-381-2022)
    is a suite of 1 Million 200x200x200 petrophysical voxels, at a
    designated size of 20 m per pixel. Forward models in the Noddyverse
    were generated as per below:
        >Geophysical forward models were calculated using a Fourier domain
        formulation using reflective padding to minimise (but not remove)
        boundary effects. The forward gravity and magnetic field
        calculations assume a flat top surface with a 100 m sensor
        elevation above this surface and the Earth's magnetic field with
        vertical inclination, zero declination and an intensity of 50000nT.

    Note that every single cell of the forward model has a calculated
    forward model, i.e. they are 200x200, with no interpolation (20 m cell
    size)

    We simulate an airborne survey, by selecting every nth row (flight line)
    of pixels. We can (but not by default) instead subsample along rows
    (ss), using the heading parameter.

    """

    # In pixel indice units
    ss = parameters.get("sample_spacing")  # ss is typically 1
    hls = parameters.get("hr_line_spacing")
    lls = parameters.get("lr_line_spacing")

    if parameters.get("heading").upper() in ["EW", "E", "W"]:
        raise NotImplementedError("Not implemented for hls and lls")
        # hls, ss = ss, hls  # swap convention to emulate survey direction

    row_coords = np.arange(raster.shape[-2])[::ss]
    hr_col_coords = np.arange(raster.shape[-1])[::hls]
    lr_col_coords = np.arange(raster.shape[-1])[::lls]

    hr_coord = np.dstack(
        (
            np.tile(row_coords, len(hr_col_coords)),
            np.repeat(hr_col_coords, len(row_coords)),
        )
    ).squeeze()

    lr_coord = np.dstack(
        (
            np.tile(row_coords, len(lr_col_coords)),
            np.repeat(lr_col_coords, len(row_coords)),
        )
    ).squeeze()

    # raster is C,H,W at this point. hr_val is equivalent to upstream "rgb"
    hr_val = raster[:, hr_coord[:, -2], hr_coord[:, -1]].T
    lr_val = raster[:, lr_coord[:, -2], lr_coord[:, -1]].T

    return hr_coord, hr_val, lr_coord, lr_val


def point_sample(fwd_model, num_points=1000):
    """Get pixel coordinates and values from input raster

    TODO: To handle raw spatial data, you would need to normalise spatial
    coordinates weirdly? Like, reproject on the fly to a -1,1 coordsys and back.
    At which point, just use the raster as is anyway. BUT we are talking point
    samples - not a grid. So I guess you would need to reproject to a local
    projection anyway, with minimal distortion... Good thing Noddy synthetic
    models are perfectly unrealistic!

    """
    # option 1 - Square only, uint8 - ok for Noddy 200x200
    hr_coord = torch.randint(
        low=0,
        high=fwd_model.shape[0],
        size=(num_points, 2),
        dtype=torch.uint8,
    )
    hr_vals = fwd_model[hr_coord[:, 0], hr_coord[:, 1]]
    # option 2 - not yet working, any shape
    # indices = list(np.ndindex(fwd_model.shape))
    # coord = rng.choice(indices, num_points, replace=False, shuffle=False)
    # vals = fwd_model.take(coord)
    # option 3 - fast, only one sample
    # np.unravel_index(rng.choice(np.flatnonzero(fwd_model)), fwd_model.shape)
    return hr_coord, hr_vals


# def _grid_old_attempt_(wesn = (0, 199, 0, 199)):
#     """Left to demonstrate vd.grid coordinates possible usage""""
#     # easting, northing = vd.grid_coordinates(
#     #     region=wesn,
#     #     spacing=(ss, ls), #NS, EW spacing. We may swap conventions above.
#     #     pixel_register=True,  # Return grid coords as pixel centers - better array shape
#     #     adjust="region",  # Ensure line spacing does not change (geophysical resolution)
#     # )
#     return NotImplementedError


def grid(hr_coord, hr_val, lr_coord, lr_val, hls, lls):
    cs_fac: int = 4
    in_cs: int = 20
    """Grid a subsampled noddy forward model.

    We define the gridding method using Verde ScipyGridder,
    fit the sampled data using .fit(),
    and create a grid with the extent provided by .grid(coordinates=...)

    The grid target cell size is determined by the step in arange, and should
    be approx 1/4 or 1/5 of the target line spacing. 
    e.g. if the lr grid ls is 4, the step should be 1, or 4/5.

    params:
        coord: tensor of [x, y] coordinates
        val: geophysical response value at corresponding list coord
        hls: sample line spacing, to calculate target cell_size
        lls: lr Line Spacing
    cs_fac: line spacing to cell size factor, typically 4 or 5
    in_cs: Input model cell size, 20m for Noddyverse

    """

    scale = hls / lls
    hr_step = hls / cs_fac
    lr_step = lls / cs_fac
    d = 100 # this tries to limit grids from extending into NAN territory
    crop_d = 48
    w0 = torch.randint(low=0, high=(d - crop_d), size=(1,), dtype=torch.uint8).numpy()
    s0 = torch.randint(low=0, high=(d - crop_d), size=(1,), dtype=torch.uint8).numpy()

    for coord, val, step in [[hr_coord, hr_val, hr_step], [lr_coord, lr_val, lr_step]]:
        w, e, s, n = np.array((w0, w0 + crop_d, s0, s0 + crop_d)) / scale
        gridder = vd.ScipyGridder("cubic")  # , extra_args={"fill_value": 0})
        gridder = gridder.fit((coord[:, 1], coord[:, 0]), val.squeeze())
        grid = gridder.grid(
            data_names="forward",
            coordinates=(
                np.meshgrid(
                    np.arange(w, e, step=step),
                    np.arange(s, n, step=step),
                )
            ),
        )

        ## DEBUG
        # plt.imshow(grid.get("forward").values.astype(np.float32), origin="lower")
        # plt.title(f"scale: {scale:0.1f}, ls={step*20} m, {e=}, {n=}")
        # plt.colorbar()
        # plt.savefig(f"test_{step*20}.png")
        # plt.close()

        yield np.expand_dims(grid.get("forward").values.astype(np.float32), 0)


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


# def to_cell_samples(grid):
#     """Convert the image to coord-*C pairs.
#     grid: Tensor, (C, H, W)
#     https://github.com/jaewon-lee-b/lte/blob/main/utils.py#L122

#     Returns:
#         coordinate array (shape = H*W, 2) and values in make_coord range
#         c_vals: H*W list of tuples of channels value(s) at coordinate n
#     i.e. return two lists describing the index and channel(s) value(s) at
#          index of input grid.
#     """

#     if len(grid.shape) != 3:  # C,H,W, data are not batched yet.
#         raise ValueError(f"Grid should be of shape C,H,W by now. Got {grid.shape}")
#     coord = make_coord(grid.shape[-2:])
#     c_vals = grid.view(grid.shape[-3], -1).permute(1, 0)
#     return coord, c_vals


class HRLRNoddyDataset(NoddyDataset):
    def __init__(self, **kwargs):
        self.lr_line_spacing = torch.randint(
            low=kwargs.get("lr_line_spacing_min", 2),
            high=kwargs.get("lr_line_spacing_max", 4) + 1,  # randint high is exclusive
            size=(1,),
            dtype=torch.int32,
        ).item()
        self.sp = {
            "hr_line_spacing": kwargs.get("hr_line_spacing", 2),
            "lr_line_spacing": self.lr_line_spacing,
            "sample_spacing": kwargs.get("sample_spacing", 1),
            "heading": kwargs.get("heading", "NS"),
        }
        self.random_heading = not bool(self.sp.get("heading"))
        super().__init__(**kwargs)

    def _process(self, index):
        dt0 = time.perf_counter()
        super()._process(index)

        hls = self.sp["hr_line_spacing"]
        lls = self.sp["lr_line_spacing"]
        if self.random_heading:
            if torch.rand(1) < 0.5:
                self.sp["heading"] = "NS"
            else:
                self.sp["heading"] = "EW"
        
        ###
        # We have the Noddyverse 200x200 forward model.
        # We need to sample and grid it to:
        # a HR cropped grid at 48x48, and it's coord / val
        # a LR cropped grid at 24x24 (for 2x scale) and its coord/val

        # Different to LTE/SWINIR, we sample the coords and vals, THEN grid.
        # It is tricky, because we want the grid to be 48x48 / scale,
        # and the coordinates / values to match the grid.
        # So, we need to either sample at the target size, or crop the grid 
        # and then drop the coord/vals that are outside the grid.

        # Alternatively, we could grid at a rather low resolution over a
        # larger extent - but this would not use realistic line spacings.

        ###

        def _crop(hr_grid, lr_grid, hr_coord, hr_val, lr_coord, lr_val):
            """When we crop grid we have to re-sample the coord vals. Ugh."""

            c = cfg.encoder_spec["img_size"]
            c0 = hr_grid.shape[-1]

            hr_grid = hr_grid[:, :, c0:c0+c, c0:c0+c]





        hr_coord, hr_val, lr_coord, lr_val = line_sample(
            self.sp,
            self.data["gt_grid"][0].numpy(),
        )
        
        hr_grid, lr_grid = grid(hr_coord, hr_val, lr_coord, lr_val, hls, lls)

        self.data["hr_coord"].shape = torch.from_numpy(hr_coord).to(dtype=torch.float32)
        self.data["lr_coord"].shape = torch.from_numpy(lr_coord).to(dtype=torch.float32)
        self.data["hr_val"].shape = torch.from_numpy(hr_val).to(dtype=torch.float32)
        self.data["lr_val"].shape = torch.from_numpy(lr_val).to(dtype=torch.float32)
        self.data["hr_grid"].shape = torch.from_numpy(hr_grid).to(dtype=torch.float32)
        self.data["lr_grid"].shape = torch.from_numpy(lr_grid).to(dtype=torch.float32)
     
        ## DEBUG
        assert not torch.isnan(self.data["hr_grid"]).any()
        assert not torch.isnan(self.data["lr_grid"]).any()
        assert not torch.isnan(self.data["hr_coord"]).any()
        assert not torch.isnan(self.data["lr_coord"]).any()
        assert not torch.isnan(self.data["hr_val"]).any()
        assert not torch.isnan(self.data["lr_val"]).any()

        if cfg.dataset_config["load_magnetics"] and cfg.dataset_config["load_gravity"]:
            #     _hr_grids = [torch.from_numpy(g).unsqueeze(0) for g in grid(hr_coord.numpy(), hr_vals.numpy(), ls=hls)]
            #     _lr_grids = [torch.from_numpy(g).unsqueeze(0) for g in grid(lr_coord, lr_val, ls=lls)]
            #     self.data["hr_grid"] = torch.stack(_hr_grids, dim=0)
            #     self.data["lr_grid"] = torch.stack(_lr_grids, dim=0)
            raise NotImplementedError("Haven't designed network for this yet")

        # OLD method point sample from existing grid
        # hr_coord, hr_vals = to_cell_samples(hr_grid.contiguous())

        self.data["hr_cell"] = torch.ones_like(
            self.data["hr_coord"], dtype=torch.float32
        )
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
