# modalities/field_coeff.py

import torch
from modalities.base import PairSampler


class FieldCoeffModality(PairSampler):
    """Data modality pairing noisy field observations u with spectral coefficients v.

    Observation model
    -----------------
    Given raw KL coefficients xi ~ N(0, I) of dimension dim_true:

        u = field[u_index] + N(0, sigma^2 * I_{dim_field})
        v = xi[:dim_coeff]                                   (noiseless)

    where field = xi @ Phi.T  (Phi from the signal generator).

    Parameters
    ----------
    gen       : SignalGenerator — underlying signal generator
    u_index   : array-like of int — indices into [0, dim_true) selecting
                                    which grid points form u
    dim_coeff : int              — number of leading KL coefficients in v
    sigma     : float            — observation noise std on u (default 0.0)
    """

    def __init__(self, gen, u_index, dim_coeff, sigma=0.0):
        self.gen = gen
        self.u_index = torch.as_tensor(u_index, dtype=torch.long)
        self.dim_coeff = dim_coeff
        self.sigma = sigma
        self.dim_field = len(self.u_index)

        assert (self.u_index < gen.dim_true).all(), "u_index out of [0, dim_true)"
        assert dim_coeff <= gen.dim_true, "dim_coeff must be <= dim_true"

    # ------------------------------------------------------------------
    # PairSampler interface
    # ------------------------------------------------------------------

    @property
    def dim_u(self):
        return self.dim_field

    @property
    def dim_v(self):
        return self.dim_coeff

    # ------------------------------------------------------------------
    # Noise helper
    # ------------------------------------------------------------------

    def _add_noise(self, u):
        """Return u + N(0, sigma^2 I) (new tensor; no in-place mutation)."""
        if self.sigma > 0.0:
            return u + self.sigma * torch.randn_like(u)
        return u

    # ------------------------------------------------------------------
    # Pair sampling
    # ------------------------------------------------------------------

    def sample_pair(self, N):
        """Draw N (u, v) pairs from the joint observation model.

        Returns
        -------
        u : (N, dim_field)  — noisy field at u_index locations
        v : (N, dim_coeff)  — truncated KL coefficients (noiseless)
        """
        xi = self.gen.sample_coefficients(N)                # (N, dim_true)
        field = self.gen.coefficients_to_field(xi)          # (N, dim_true)
        u = self._add_noise(field[:, self.u_index])         # (N, dim_field)
        v = xi[:, :self.dim_coeff]                          # (N, dim_coeff)
        return u, v

    # ------------------------------------------------------------------
    # Marginal moments of u
    # ------------------------------------------------------------------

    def cov_field(self):
        """Marginal covariance of u: Phi_u @ Phi_u.T + sigma^2 I.

        Returns (dim_field, dim_field).
        """
        Phi_u = self.gen.coeff_to_field_map()[self.u_index, :]  # (dim_field, dim_true)
        Cov = Phi_u @ Phi_u.T                                    # (dim_field, dim_field)
        if self.sigma > 0.0:
            Cov = Cov + self.sigma ** 2 * torch.eye(self.dim_field, dtype=Cov.dtype)
        return Cov

    # ------------------------------------------------------------------
    # Conditional moments: field / u  given  xi_r = xi[:dim_coeff]
    # ------------------------------------------------------------------

    def cond_mean_field(self, xi, truncate=False):
        """Posterior mean of the field given the leading coefficients xi_r.

        E[field | xi_r][n] = sum_{k < dim_coeff} Phi[n, k] * xi_r[k]

        Parameters
        ----------
        xi       : (dim_coeff,) or (N, dim_coeff)
        truncate : if True, return values only at u_index locations
        """
        Phi = self.gen.coeff_to_field_map()         # (dim_true, dim_true)
        Ar = Phi[:, :self.dim_coeff]                # (dim_true, dim_coeff)
        batched = xi.ndim == 2
        mu = (xi @ Ar.T) if batched else (Ar @ xi)
        if truncate:
            return mu[:, self.u_index] if batched else mu[self.u_index]
        return mu

    def cond_cov_field(self, truncate=False):
        """Posterior covariance of the field given the leading coefficients xi_r.

        Cov[field | xi_r] = Phi[:, dim_coeff:] @ Phi[:, dim_coeff:].T
        """
        Phi = self.gen.coeff_to_field_map()         # (dim_true, dim_true)
        tail = Phi[:, self.dim_coeff:]               # (dim_true, dim_true - dim_coeff)
        Cov_full = tail @ tail.T                     # (dim_true, dim_true)
        if not truncate:
            return Cov_full
        Cov_u = Cov_full[self.u_index][:, self.u_index]
        if self.sigma > 0.0:
            Cov_u = Cov_u + self.sigma ** 2 * torch.eye(self.dim_field, dtype=Cov_u.dtype)
        return Cov_u

    # ------------------------------------------------------------------
    # Conditional moments: xi_r = xi[:dim_coeff]  given  u
    # ------------------------------------------------------------------

    def cond_mean_coeff(self, u):
        """Posterior mean of xi_r given u: Ar.T @ Sigma_uu^{-1} @ u."""
        Phi = self.gen.coeff_to_field_map()
        Ar = Phi[self.u_index, :self.dim_coeff]     # (dim_field, dim_coeff)
        Sigma_uu = self.cov_field()                  # (dim_field, dim_field)
        K = torch.linalg.solve(Sigma_uu, Ar)        # (dim_field, dim_coeff)
        batched = u.ndim == 2
        return (u @ K) if batched else (K.T @ u)

    def cond_cov_coeff(self):
        """Posterior covariance of xi_r given u: I_r - Ar.T @ Sigma_uu^{-1} @ Ar."""
        Phi = self.gen.coeff_to_field_map()
        Ar = Phi[self.u_index, :self.dim_coeff]              # (dim_field, dim_coeff)
        Sigma_uu = self.cov_field()                          # (dim_field, dim_field)
        Sigma_rr = self.gen.cov_coeff()[:self.dim_coeff, :self.dim_coeff]
        return Sigma_rr - Ar.T @ torch.linalg.solve(Sigma_uu, Ar)

    # ------------------------------------------------------------------
    # Conditional sampling (diagnostic utilities)
    # ------------------------------------------------------------------

    def sample_cond_field(self, N, xi):
        """Sample full-grid noiseless fields conditioned on xi_r."""
        assert xi.shape[-1] == self.dim_coeff
        d = self.gen.dim_true
        xi_padded = torch.zeros(N, d, dtype=xi.dtype)
        xi_padded[:, :self.dim_coeff] = xi.unsqueeze(0).expand(N, -1)
        xi_padded[:, self.dim_coeff:] = torch.randn(
            N, d - self.dim_coeff, dtype=xi.dtype
        )
        return self.gen.coefficients_to_field(xi_padded)

    def sample_cond_coeff(self, N, u):
        """Sample xi_r conditioned on u via Cholesky decomposition."""
        mu = self.cond_mean_coeff(u)           # (dim_coeff,)
        cov = self.cond_cov_coeff()            # (dim_coeff, dim_coeff)
        L = torch.linalg.cholesky(cov)
        eps = torch.randn(N, self.dim_coeff, dtype=mu.dtype)
        return mu.unsqueeze(0) + eps @ L.T

    # ------------------------------------------------------------------
    # PairSampler analytical triple
    # ------------------------------------------------------------------

    def C_uu(self):
        return self.cov_field()

    def C_vv(self):
        return torch.eye(self.dim_coeff, dtype=torch.float64)

    def C_uv(self):
        """E[u v^T] = Phi_u[:, :dim_coeff].  Shape (dim_field, dim_coeff)."""
        Phi = self.gen.coeff_to_field_map()
        return Phi[self.u_index, :self.dim_coeff]
