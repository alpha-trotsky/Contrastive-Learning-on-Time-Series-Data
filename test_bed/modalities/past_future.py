import torch
from modalities.base import PairSampler


class PastFutureModality(PairSampler):
    def __init__(self, gen, past_index, future_index, sigma_u=0.0, sigma_v=0.0):
        self.gen = gen
        self.past_index = torch.as_tensor(past_index, dtype=torch.long)
        self.future_index = torch.as_tensor(future_index, dtype=torch.long)
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v

    @property
    def dim_u(self):
          return len(self.past_index)

    @property
    def dim_v(self):
      return len(self.future_index)


    def sample_pair(self, N):
        xi = self.gen.sample_coefficients(N)               # (N, dim_true)
        field = self.gen.coefficients_to_field(xi)         # (N, dim_true)
        u = field[:, self.past_index]                      # (N, dim_u)
        v = field[:, self.future_index]                    # (N, dim_v)
        if self.sigma_u > 0:
            u = u + self.sigma_u * torch.randn_like(u)
        if self.sigma_v > 0:
            v = v + self.sigma_v * torch.randn_like(v)
        return u, v
    

    def C_uu(self):
        Phi_p = self.gen.coeff_to_field_map()[self.past_index, :]
        Cov = Phi_p @ Phi_p.T
        if self.sigma_u > 0:
            Cov = Cov + self.sigma_u ** 2 * torch.eye(self.dim_u, dtype=Cov.dtype)
        return Cov

    def C_vv(self):
        Phi_f = self.gen.coeff_to_field_map()[self.future_index, :]
        Cov = Phi_f @ Phi_f.T
        if self.sigma_v > 0:
            Cov = Cov + self.sigma_v ** 2 * torch.eye(self.dim_v, dtype=Cov.dtype)
        return Cov

    def C_uv(self):
        Phi = self.gen.coeff_to_field_map()
        Phi_p = Phi[self.past_index, :]
        Phi_f = Phi[self.future_index, :]
        return Phi_p @ Phi_f.T