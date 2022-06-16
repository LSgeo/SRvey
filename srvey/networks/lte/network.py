import torch
import torch.nn as nn

import lte


class model(nn.Module):
    def __init__(self) -> None:
        pass

    def load_pretrained_model():
        pass


class mlp(nn.Module):
    """As per https://github.com/jaewon-lee-b/lte/blob/main/models/mlp.py
    Default (https://github.com/jaewon-lee-b/lte/blob/main/configs/train-div2k/train_swinir-lte.yaml):
        out_dim: 3
        hidden_list: [256, 256, 256]
    """

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
