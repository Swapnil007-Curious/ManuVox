import itertools

import torch
from torch import nn
from torch.nn import functional as F

HAND_PAIRS = list(itertools.combinations(range(21), 2))  # 210 point pairs of one hand.
FINGER_POINTS = (0, 4, 8, 12, 16, 20)  # Wrist and the five fingertips.
COORDS = 2 * N_LANDMARKS  # 118 coordinates per frame.


class FeatureBlock(nn.Module):
    """(B, 32, 120) frames -> (B, 32, 588) standardised features, inside the model.

    Features: coordinates, velocity and acceleration of all 59 points, the 210 pairwise distances
    of the dominant (right-block) hand, and 24 distances from its wrist and fingertips to the
    nose, mouth centre and shoulders. A point is missing when x == y == 0, and anything derived
    from a missing point is 0. Living inside the model means the ONNX export carries it too.
    """

    def __init__(self):
        super().__init__()
        self.output_dim = 3 * COORDS + len(HAND_PAIRS) + len(FINGER_POINTS) * 4
        first, second = zip(*HAND_PAIRS)
        self.register_buffer("pair_a", torch.tensor(first) + 21, persistent=False)
        self.register_buffer("pair_b", torch.tensor(second) + 21, persistent=False)
        self.register_buffer("fingers", torch.tensor(FINGER_POINTS) + 21, persistent=False)
        face = [NOSE, 42 + 9, 42 + 10, L_SHOULDER, R_SHOULDER]  # Nose, mouth corners, shoulders.
        self.register_buffer("targets", torch.tensor(face), persistent=False)
        self.register_buffer("mean", torch.zeros(self.output_dim))
        self.register_buffer("std", torch.ones(self.output_dim))

    def raw(self, x):
        b, t = x.shape[:2]
        p = x[..., :COORDS].reshape(b, t, N_LANDMARKS, 2)
        ok = ((p[..., 0] != 0) | (p[..., 1] != 0)).to(p.dtype)
        step = (p[:, 1:] - p[:, :-1]) * (ok[:, 1:] * ok[:, :-1]).unsqueeze(-1)
        velocity = torch.cat([torch.zeros_like(step[:, :1]), step], dim=1)
        second = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) * (ok[:, 2:] * ok[:, 1:-1] * ok[:, :-2]).unsqueeze(-1)
        acceleration = torch.cat([torch.zeros_like(second[:, :2]), second], dim=1)
        shape = distances(
            p.index_select(2, self.pair_a),
            p.index_select(2, self.pair_b),
            ok.index_select(2, self.pair_a) * ok.index_select(2, self.pair_b),
        )
        tg, tg_ok = p.index_select(2, self.targets), ok.index_select(2, self.targets)
        mouth = 0.5 * (tg[:, :, 1:2] + tg[:, :, 2:3])
        anchors = torch.cat([tg[:, :, :1], mouth, tg[:, :, 3:]], dim=2)
        anchors_ok = torch.cat([tg_ok[:, :, :1], tg_ok[:, :, 1:2] * tg_ok[:, :, 2:3], tg_ok[:, :, 3:]], dim=2)
        hand, hand_ok = p.index_select(2, self.fingers), ok.index_select(2, self.fingers)
        to_body = distances(
            hand.unsqueeze(3), anchors.unsqueeze(2), hand_ok.unsqueeze(3) * anchors_ok.unsqueeze(2)
        )
        return torch.cat(
            [
                x[..., :COORDS],
                velocity.reshape(b, t, COORDS),
                acceleration.reshape(b, t, COORDS),
                shape,
                to_body.reshape(b, t, -1),
            ],
            dim=-1,
        )

    def forward(self, x):
        return (self.raw(x) - self.mean) / self.std.clamp_min(1e-6)


class SimpleFeatures(nn.Module):
    """Coordinates and velocity only, (B, 32, 236): the cheaper input of the temporal CNN."""

    def __init__(self):
        super().__init__()
        self.output_dim = 2 * COORDS
        self.register_buffer("mean", torch.zeros(self.output_dim))
        self.register_buffer("std", torch.ones(self.output_dim))

    def raw(self, x):
        b, t = x.shape[:2]
        p = x[..., :COORDS].reshape(b, t, N_LANDMARKS, 2)
        ok = ((p[..., 0] != 0) | (p[..., 1] != 0)).to(p.dtype)
        step = (p[:, 1:] - p[:, :-1]) * (ok[:, 1:] * ok[:, :-1]).unsqueeze(-1)
        velocity = torch.cat([torch.zeros_like(step[:, :1]), step], dim=1)
        return torch.cat([x[..., :COORDS], velocity.reshape(b, t, COORDS)], dim=-1)

    def forward(self, x):
        return (self.raw(x) - self.mean) / self.std.clamp_min(1e-6)


def distances(a, b, valid):
    """Euclidean distance over the last axis, 0 where ``valid`` is 0."""
    return torch.sqrt(((a - b) ** 2).sum(dim=-1)) * valid


@torch.no_grad()
def fit_standardisation(block, x_train, chunk=4096):
    """Set the block's mean and std from every frame of the train split (float64 sums)."""
    total = torch.zeros(block.output_dim, dtype=torch.float64, device=x_train.device)
    squares = torch.zeros_like(total)
    for start in range(0, len(x_train), chunk):
        values = block.raw(x_train[start : start + chunk].float()).reshape(-1, block.output_dim).double()
        total += values.sum(dim=0)
        squares += (values**2).sum(dim=0)
    count = x_train.shape[0] * x_train.shape[1]
    mean = total / count
    block.mean.copy_(mean)
    block.std.copy_((squares / count - mean**2).clamp_min(0).sqrt().clamp_min(1e-6))
