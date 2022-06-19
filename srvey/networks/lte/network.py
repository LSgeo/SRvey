print(f"Loading {__name__}")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import srvey.cfg as cfg


def make_coord(shape, ranges=None, flatten=True):
    """https://github.com/jaewon-lee-b/lte/blob/main/utils.py#L104"""
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class LTE(nn.Module):
    """From https://github.com/jaewon-lee-b/lte/blob/main/models/lte.py"""

    def __init__(self):
        from srvey.networks.lte.swinir import SwinIR

        super().__init__()
        self.encoder = SwinIR(*cfg.encoder_spec)
        self.coef = nn.Conv2d(
            self.encoder.out_dim, cfg.imnet_spec["hidden_dim"], 3, padding=1
        )
        self.freq = nn.Conv2d(
            self.encoder.out_dim, cfg.imnet_spec["hidden_dim"], 3, padding=1
        )
        self.phase = nn.Linear(2, cfg.imnet_spec["hidden_dim"] // 2, bias=False)

        self.imnet = mlp(*cfg.imnet_spec)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = (
            make_coord(inp.shape[-2:], flatten=False)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(inp.shape[0], 2, *inp.shape[-2:])
        )

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat(
                    (torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1
                )

                inp = torch.mul(q_coef, q_freq)

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(
            self.inp,
            coord.flip(-1).unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


# class model(LTE):
#     def __init__():
#         super().__init__()


class mlp(nn.Module):
    """https://github.com/jaewon-lee-b/lte/blob/main/models/mlp.py"""

    def __init__(self):
        super().__init__()
        layers = []
        lastv = cfg.imnet_spec["in_dim"]
        for hidden in cfg.imnet_spec["hidden_list"]:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, cfg.imnet_spec["out_dim"]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
