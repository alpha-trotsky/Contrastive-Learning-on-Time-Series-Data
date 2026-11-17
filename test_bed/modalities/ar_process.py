"""Gaussian past/future windows of a stationary process with a chosen
correlation between subsequent time steps.

The series is an equal-weight sum of independent AR(1) components with lag-one
correlations phis = (phi_1, ..., phi_k), plus optional white observation noise:

    cov(x_s, x_t) = mean_j phi_j^{|s - t|}  (+ sigma^2 if s == t).

The past is the first past_len steps of a window, the future the next
future_len.  The cross-covariance has rank k (the dimension of the Markov
state), so exactly k canonical correlations between past and future are
non-zero.  With one component and no noise the only one equals phi.

Used as the Gaussian base of modalities.nonlinear.StudentTPastFuture.
"""
import torch
from modalities.base import PairSampler


class ARPastFuture(PairSampler):
    def __init__(self, phis, past_len=50, future_len=50, sigma=0.0):
        self.phis = [float(p) for p in phis]
        self.past_len, self.future_len, self.sigma = past_len, future_len, sigma
        n = past_len + future_len
        lag = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs().double()
        C = sum(torch.where(lag == 0, torch.ones_like(lag), torch.tensor(p, dtype=torch.float64) ** lag)
                for p in self.phis) / len(self.phis)
        self.C = C + sigma ** 2 * torch.eye(n, dtype=torch.float64)
        self.L = torch.linalg.cholesky(self.C)

    dim_u = property(lambda self: self.past_len)
    dim_v = property(lambda self: self.future_len)

    def sample_pair(self, N):
        x = torch.randn(N, self.C.shape[0], dtype=torch.float64) @ self.L.T
        return x[:, :self.past_len], x[:, self.past_len:]

    def C_uu(self):
        return self.C[:self.past_len, :self.past_len]

    def C_vv(self):
        return self.C[self.past_len:, self.past_len:]

    def C_uv(self):
        return self.C[:self.past_len, self.past_len:]
