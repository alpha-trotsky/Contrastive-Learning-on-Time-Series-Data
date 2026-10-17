"""Non-Gaussian / nonlinear past->future domains with EXACT oracles.

Each domain has a strength knob whose zero is a Gaussian process, so every
result can be read as a curve anchored at the validated Gaussian theory.  Each
exposes  oracle_samples(u, M)  -- exact draws from the true p(v | u) -- so the
scoring layer can report excess-over-oracle (half the energy distance to the
truth), and conditional calibration against the true spread.

  StudentTPastFuture   elliptical t-process.  Same covariances and the same
                       LINEAR conditional mean as the GP; heavy tails and a
                       heteroscedastic conditional spread.  Tests the
                       covariance/uncertainty claims in isolation.
                       knob: df (inf -> GP).
  WarpedPastFuture     Gaussian copula: u = g(x_u), v = g(x_v) with a monotone
                       sinh-arcsinh warp g.  NONLINEAR conditional mean, skew,
                       tails; PMI is that of the latent GP (invariant to g), so a
                       nonlinear encoder has a known target (g^{-1}).
                       knob: warp strength (0 -> GP).
  DoubleWellPastFuture Markov SDE  dx = [-theta x + lam*kappa (x - x^3)] dt + s dW.
                       Genuinely nonlinear dynamics: regime switching and
                       BIMODAL futures.  knob: lam (0 -> OU, a Gaussian process).
"""
import math
import torch

from modalities.base import PairSampler


def _gaussian_conditional(C_uu, C_vv, C_uv):
    K = torch.linalg.solve(C_uu, C_uv)
    C_vgu = C_vv - C_uv.T @ K
    return K, C_vgu


# ---------------------------------------------------------------------------
# Tier 1 -- Student-t process
# ---------------------------------------------------------------------------

class StudentTPastFuture(PairSampler):
    """(u, v) = c * (u_g, v_g) / sqrt(w),  w ~ Gamma(df/2, rate=df/2),
    with (u_g, v_g) from a Gaussian PastFuture modality and c^2 = (df-2)/df so
    the covariances equal the Gaussian ones exactly (the Gaussian-theory A* is
    therefore unchanged for every df -- that is the null hypothesis)."""

    def __init__(self, base, df):
        assert df > 2
        self.base, self.df = base, float(df)
        self.c2 = 1.0 if math.isinf(self.df) else (self.df - 2) / self.df

    dim_u = property(lambda self: self.base.dim_u)
    dim_v = property(lambda self: self.base.dim_v)
    C_uu = lambda self: self.base.C_uu()
    C_vv = lambda self: self.base.C_vv()
    C_uv = lambda self: self.base.C_uv()

    def _w(self, N, shape_df):
        if math.isinf(shape_df):
            return torch.ones(N, 1, dtype=torch.float64)
        return torch.distributions.Gamma(shape_df / 2, shape_df / 2).sample((N, 1)).double()

    def sample_pair(self, N):
        u, v = self.base.sample_pair(N)
        r = math.sqrt(self.c2) / self._w(N, self.df).sqrt()
        return u * r, v * r

    def oracle_samples(self, u, M):
        """v | u ~ t_{df+du}( uK, (df + q)/(df + du) * c^2 C_{v|u} ),  q = u^T (c^2 C_uu)^{-1} u."""
        u = u.double()
        K, C_vgu = _gaussian_conditional(self.C_uu(), self.C_vv(), self.C_uv())
        N, du = u.shape
        L = torch.linalg.cholesky(self.c2 * C_vgu)
        z = torch.randn(N, M, self.dim_v, dtype=torch.float64) @ L.T
        if math.isinf(self.df):
            return (u @ K)[:, None, :] + z
        q = (u * torch.linalg.solve(self.c2 * self.C_uu(), u.T).T).sum(1)       # (N,)
        scale = ((self.df + q) / (self.df + du)).sqrt()[:, None, None]
        w = torch.distributions.Gamma((self.df + du) / 2, (self.df + du) / 2).sample((N, M, 1)).double()
        return (u @ K)[:, None, :] + scale * z / w.sqrt()

    def oracle_spread(self, u):
        """Per-query conditional scale factor (for conditional-calibration bins)."""
        if math.isinf(self.df):
            return torch.ones(u.shape[0], dtype=torch.float64)
        q = (u.double() * torch.linalg.solve(self.c2 * self.C_uu(), u.double().T).T).sum(1)
        return ((self.df + q) / (self.df + self.dim_u)).sqrt()


# ---------------------------------------------------------------------------
# Tier 2 -- warped GP (Gaussian copula)
# ---------------------------------------------------------------------------

class WarpedPastFuture(PairSampler):
    """u = g(x_u), v = g(x_v), (x_u, x_v) from a Gaussian PastFuture modality
    (observation noise lives in the latent, so the oracle is exact).

    g(x) = s * sinh((asinh(x/s) + eps) / delta),  s = latent pointwise std.
    strength k in [0, 1] sets  eps = 0.75 k  (skew),  delta = 1 - 0.5 k  (tails),
    then the output is re-centred / re-scaled to the latent's mean 0, std s so
    only the SHAPE changes with k.  k = 0 is the identity (the GP).
    """

    def __init__(self, base, strength):
        self.base, self.k = base, float(strength)
        self.eps, self.delta = 0.75 * self.k, 1.0 - 0.5 * self.k
        self.s = math.sqrt(base.C_uu().diag().mean().item())
        # affine normalisation of the warp, by quadrature over x ~ N(0, s^2)
        x = torch.linspace(-8, 8, 20001, dtype=torch.float64) * self.s
        p = torch.exp(-0.5 * (x / self.s) ** 2)
        p = p / p.sum()
        y = self._raw(x)
        self.m = (p * y).sum().item()
        self.sd = ((p * (y - self.m) ** 2).sum().sqrt() / self.s).item()

    dim_u = property(lambda self: self.base.dim_u)
    dim_v = property(lambda self: self.base.dim_v)

    def _raw(self, x):
        s = self.s
        return s * torch.sinh((torch.asinh(x / s) + self.eps) / self.delta)

    def g(self, x):
        return (self._raw(x) - self.m) / self.sd

    def g_inv(self, y):
        s = self.s
        r = y * self.sd + self.m
        return s * torch.sinh(self.delta * torch.asinh(r / s) - self.eps)

    def sample_pair(self, N):
        xu, xv = self.base.sample_pair(N)
        return self.g(xu), self.g(xv)

    def oracle_samples(self, u, M):
        K, C_vgu = _gaussian_conditional(self.base.C_uu(), self.base.C_vv(), self.base.C_uv())
        xu = self.g_inv(u.double())
        L = torch.linalg.cholesky(C_vgu)
        z = torch.randn(u.shape[0], M, self.dim_v, dtype=torch.float64) @ L.T
        return self.g((xu @ K)[:, None, :] + z)

    def oracle_spread(self, u):
        return self.oracle_samples(u, 200).std(1).mean(1)


# ---------------------------------------------------------------------------
# Tier 3 -- double-well SDE
# ---------------------------------------------------------------------------

class DoubleWellPastFuture(PairSampler):
    """x_{t+1} = x_t + f(x_t) dt + s sqrt(dt) eps,  f(x) = -theta x + lam*kappa*(x - x^3).

    Observed every `stride` Euler steps; u = past_len observations (+ N(0, sigma_u^2)),
    v = the next future_len observations (+ N(0, sigma_v^2)).  Initial state from
    a long burn-in, so windows are stationary draws.

    Oracle: bootstrap particle filter over the past (exact as n_particles -> inf;
    1-D state so a few thousand particles is plenty), then forward simulation.
    """

    def __init__(self, lam, past_len=50, future_len=50, theta=1.0, kappa=4.0, s=0.7,
                 dt=0.01, stride=5, sigma_u=0.05, sigma_v=0.05, burn_in=2000, n_particles=2000):
        self.lam, self.theta, self.kappa, self.sig = float(lam), theta, kappa, s
        self.dt, self.stride, self.burn_in = dt, stride, burn_in
        self.past_len, self.future_len = past_len, future_len
        self.sigma_u, self.sigma_v, self.n_particles = sigma_u, sigma_v, n_particles

    dim_u = property(lambda self: self.past_len)
    dim_v = property(lambda self: self.future_len)

    def drift(self, x):
        return -self.theta * x + self.lam * self.kappa * (x - x ** 3)

    def _advance(self, x, n_obs):
        """Simulate n_obs observation intervals from states x (...,).  -> (..., n_obs)."""
        out = []
        for _ in range(n_obs):
            for _ in range(self.stride):
                x = x + self.drift(x) * self.dt + self.sig * math.sqrt(self.dt) * torch.randn_like(x)
            out.append(x)
        return torch.stack(out, -1), x

    def _stationary(self, N):
        x = torch.randn(N, dtype=torch.float64)
        for _ in range(self.burn_in):
            x = x + self.drift(x) * self.dt + self.sig * math.sqrt(self.dt) * torch.randn_like(x)
        return x

    def sample_pair(self, N):
        x0 = self._stationary(N)
        path, _ = self._advance(x0, self.past_len + self.future_len)
        u = path[:, :self.past_len] + self.sigma_u * torch.randn(N, self.past_len, dtype=torch.float64)
        v = path[:, self.past_len:] + self.sigma_v * torch.randn(N, self.future_len, dtype=torch.float64)
        return u, v

    @torch.no_grad()
    def oracle_samples(self, u, M):
        u = u.double()
        N, P = u.shape[0], self.n_particles
        x = self._stationary(N * P).view(N, P)
        logw = torch.zeros(N, P, dtype=torch.float64)
        for t in range(self.past_len):
            x, _ = self._advance(x, 1)
            x = x[..., 0]
            logw = logw - 0.5 * ((u[:, t:t + 1] - x) / self.sigma_u) ** 2
            w = torch.softmax(logw, 1)
            ess = 1.0 / (w ** 2).sum(1)
            # resample rows whose ESS dropped below half
            low = ess < P / 2
            if low.any():
                idx = torch.multinomial(w[low], P, replacement=True)
                x[low] = torch.gather(x[low], 1, idx)
                logw[low] = 0.0
        w = torch.softmax(logw, 1)
        idx = torch.multinomial(w, M, replacement=True)
        x_last = torch.gather(x, 1, idx)                                     # (N, M)
        fut, _ = self._advance(x_last, self.future_len)                     # (N, M, F)
        return fut + self.sigma_v * torch.randn_like(fut)

    def oracle_spread(self, u):
        return self.oracle_samples(u, 200).std(1).mean(1)
