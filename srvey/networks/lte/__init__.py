"""LTE
As implemented by https://github.com/jaewon-lee-b/lte
Local Texture Estimator for Implicit Representation Function (CVPR 2022)

I (Luke Smith) hardcode SWINIR and MLP, the top performing model from their work

"""
print(f'Loading {__name__}')

import torch

from srvey.networks import save_previews
from srvey.networks.lte.network import LTE
from srvey.networks.lte.swinir import SwinIR
