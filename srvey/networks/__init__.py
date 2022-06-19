""" srvey.networks

Here I collect network agnostic "network-y" code. 
For example, save_previews should be usable in any network architecture.
Pending a refactor for it to use kwargs as raster count arbitrary saver.
And to make it not do model things. Do that in the model...
"""
print(f"Loading {__name__}")

import torch
import numpy as np


def save_previews(
    session, log_to_disk: bool = True, log_to_comet: bool = False, **kwargs
):
    # """Convert current batch to images and log to comet.ml"""

    # from srvey import cfg

    # self.model.eval()  # ensure no training occurs
    # with torch.no_grad():
    #     sr = self.model(self.batch["lr"].to(self.device, non_blocking=True))
    #     data = [["SR", sr.detach().cpu().numpy()]]
    #     if self.curr_epoch == self.start_epoch:  # Log LR and HR input and target once
    #         data.append(["LR", self.batch["lr"].detach().cpu().numpy()])
    #         data.append(["HR", self.batch["hr"].detach().cpu().numpy()])

    # for i, (name, batch) in enumerate(data):  # For each resolution data
    #     v = 0  # Reset tile index for each Resolution batch
    #     for j, d in enumerate(batch):  # For each tensor data in the batch
    #         # if log_to_comet:
    #         self.exp.log_image(
    #             (255 * (d - d.min()) / (d.max() - d.min())).astype(np.uint8),
    #             name=f"Tile_{cfg.preview_indices[v]}_{name}",
    #             image_scale=1,
    #             step=self.curr_iteration,
    #         )
    #         v += 1  # Track tile within batch
    pass
