# models/mlp_regressor.py
"""Non-contrastive baselines with nonlinear (or linear) networks.

MeanRegressor      u -> E[v|u], trained with MSE.  Forecast = point, or point +
                   Gaussian residual noise with one fixed covariance.
GaussianRegressor  u -> (mean, per-step log-variance), trained with Gaussian
                   NLL.  The band width depends on the past; the shape is always
                   Gaussian.  Samples reuse the residual correlation across steps
                   (estimated from standardised training residuals), so only the
                   per-step widths are learned.  This is the "best nonlinear
                   Gaussian model" a contrastive forecaster has to beat.

depth=0 gives the linear version of each.
"""
import torch
import torch.nn as nn

from models.mlp_clip import MLPEncoder


class MeanRegressor(nn.Module):
    def __init__(self, u_dimension, v_dimension, hidden=256, depth=2, skip=True):
        super().__init__()
        self.net = MLPEncoder(u_dimension, v_dimension, hidden, depth, skip and depth > 0)
        self.register_buffer('v_mean', torch.zeros(v_dimension))
        self.register_buffer('v_std', torch.ones(v_dimension))

    def set_input_stats(self, u, v):
        self.net.set_input_stats(u)
        self.v_mean.copy_(v.mean(0).to(self.v_mean))
        self.v_std.copy_(v.std(0).clamp_min(1e-8).to(self.v_std))

    def predict(self, u):
        """Predicted mean in data units."""
        return self.net(u.float()) * self.v_std + self.v_mean

    def loss(self, u, v):
        z = (v.float() - self.v_mean) / self.v_std
        return ((self.net(u.float()) - z) ** 2).mean()


class GaussianRegressor(nn.Module):
    def __init__(self, u_dimension, v_dimension, hidden=256, depth=2, skip=True):
        super().__init__()
        self.d = v_dimension
        self.net = MLPEncoder(u_dimension, 2 * v_dimension, hidden, depth, skip and depth > 0)
        self.register_buffer('v_mean', torch.zeros(v_dimension))
        self.register_buffer('v_std', torch.ones(v_dimension))

    def set_input_stats(self, u, v):
        self.net.set_input_stats(u)
        self.v_mean.copy_(v.mean(0).to(self.v_mean))
        self.v_std.copy_(v.std(0).clamp_min(1e-8).to(self.v_std))

    def _out(self, u):
        o = self.net(u.float())
        return o[:, :self.d], o[:, self.d:].clamp(-15, 10)

    def predict(self, u):
        """(mean, per-step std) in data units."""
        mu, log_var = self._out(u)
        return mu * self.v_std + self.v_mean, (0.5 * log_var).exp() * self.v_std

    def loss(self, u, v):
        z = (v.float() - self.v_mean) / self.v_std
        mu, log_var = self._out(u)
        return 0.5 * (log_var + (z - mu) ** 2 / log_var.exp()).mean()
