# models/mlp_clip.py

import torch
import torch.nn as nn
import numpy as np


class MLPEncoder(nn.Module):
    """Small MLP  x -> embedding, with an optional linear skip path.

    Inputs are standardised with fixed per-coordinate statistics (set once from
    training data via `set_input_stats`) so the GELU layers see O(1) values
    whatever the domain's scale.  With skip=True the output is
    MLP(x) + Linear(x), so the model contains the linear encoder.
    """

    def __init__(self, in_dim, out_dim, hidden=256, depth=2, skip=True):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if skip else None
        self.register_buffer('mean', torch.zeros(in_dim))
        self.register_buffer('std', torch.ones(in_dim))

    def set_input_stats(self, x):
        self.mean.copy_(x.mean(0).to(self.mean))
        self.std.copy_(x.std(0).clamp_min(1e-8).to(self.std))

    def forward(self, x):
        z = (x - self.mean) / self.std
        out = self.mlp(z)
        if self.skip is not None:
            out = out + self.skip(z)
        return out


class MLPCLIP(nn.Module):
    """Two MLP encoders with a learnable logit scale (same interface as LinearCLIP).

    Parameters
    ----------
    embed_dim        : int   — shared embedding dimension
    u_dimension      : int   — input dimension of u
    v_dimension      : int   — input dimension of v
    hidden           : int   — hidden width of each MLP
    depth            : int   — number of hidden layers
    skip             : bool  — add a linear skip path (model contains the linear one)
    init_logit_scale : float — initial value of log(logit_scale)
    """

    def __init__(
        self,
        embed_dim: int,
        u_dimension: int,
        v_dimension: int,
        hidden: int = 256,
        depth: int = 2,
        skip: bool = True,
        init_logit_scale: float = np.log(1 / 0.07),
    ):
        super().__init__()
        self.u_encoder = MLPEncoder(u_dimension, embed_dim, hidden, depth, skip)
        self.v_encoder = MLPEncoder(v_dimension, embed_dim, hidden, depth, skip)
        self.logit_scale = nn.Parameter(torch.ones(1) * init_logit_scale)

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def set_input_stats(self, u, v):
        self.u_encoder.set_input_stats(u)
        self.v_encoder.set_input_stats(v)

    def encode_u(self, u):
        return self.u_encoder(u.type(self.dtype))

    def encode_v(self, v):
        return self.v_encoder(v.type(self.dtype))

    def forward(self, u, v):
        u_features = self.encode_u(u)
        v_features = self.encode_v(v)
        return u_features, v_features, self.logit_scale.exp()
