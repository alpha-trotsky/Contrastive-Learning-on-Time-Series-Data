from torch import nn
import torch
import numpy as np
 # taken directly from the paper i believe?
class LinearCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 u_dimension: int,
                 v_dimension: int,
                 bias=False,
                 init_logit_scale: float = np.log(1 / 0.07),
                 ):
        super().__init__()
        self.u_encoder = nn.Linear(u_dimension, embed_dim, bias=bias)
        self.v_encoder = nn.Linear(v_dimension, embed_dim, bias=bias)
        self.logit_scale = nn.Parameter(torch.ones(1) * init_logit_scale)

    @property
    def dtype(self):
        return self.u_encoder.weight.dtype

    def encode_u(self, u):
        return self.u_encoder(u.type(self.dtype))

    def encode_v(self, v):
        return self.v_encoder(v.type(self.dtype))

    def cross_term(self):
        return self.logit_scale.exp() * self.u_encoder.weight.T @ self.v_encoder.weight

    def forward(self, u, v):
        u_features = self.encode_u(u)
        v_features = self.encode_v(v)
        return u_features, v_features, self.logit_scale.exp()