# models/linear_clip.py

import torch
import torch.nn as nn
import numpy as np


class LinearCLIP(nn.Module):
    """Two linear encoders with a learnable logit scale.

    Encodes u and v into a shared embedding space.  No bias terms so that
    the learned weight matrices directly correspond to the theoretical A*.

    Parameters
    ----------
    embed_dim        : int   — shared embedding dimension
    u_dimension      : int   — input dimension of u
    v_dimension      : int   — input dimension of v
    bias             : bool  — whether to use bias (default False)
    init_logit_scale : float — initial value of log(logit_scale)
    """

    def __init__(
        self,
        embed_dim: int,
        u_dimension: int,
        v_dimension: int,
        bias: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
    ):
        super().__init__()
        self.u_encoder = nn.Linear(u_dimension, embed_dim, bias=bias)
        self.v_encoder = nn.Linear(v_dimension, embed_dim, bias=bias)
        self.logit_scale = nn.Parameter(torch.ones(1) * init_logit_scale) # changing the logit_scale - not sure if i should be doing this 

    @property
    def dtype(self):
        return self.u_encoder.weight.dtype

    def encode_u(self, u):
        return self.u_encoder(u.type(self.dtype))

    def encode_v(self, v):
        return self.v_encoder(v.type(self.dtype))

    def cross_term(self):
        """Return logit_scale * W_u^T @ W_v.  Shape (dim_u, dim_v)."""
        return self.logit_scale.exp() * self.u_encoder.weight.T @ self.v_encoder.weight

    def forward(self, u, v):
        u_features = self.encode_u(u)
        v_features = self.encode_v(v)
        return u_features, v_features, self.logit_scale.exp()
