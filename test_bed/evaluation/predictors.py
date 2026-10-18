"""Adapters that turn any forecasting engine into predictive samples (N, M, d).

The contrastive adapter is the important one.  A contrastive model never
outputs v; it outputs a score s(u, v).  Its model distribution is (paper eq. 39
/ 46, any encoder):

        nu(v | u)  ∝  exp(s(u, v)) mu_v(v).

Given a bank {v_b} ~ mu_v (training futures), this is sampled exactly as the
bank grows by sampling-importance-resampling with weights softmax_b s(u, v_b).
This needs only black-box evaluations of s, so it works unchanged for linear
or nonlinear encoders, Gaussian or non-Gaussian data, synthetic or real series.
The effective sample size of the weights is reported as a reliability check.
"""
import torch


def gaussian_samples(mean, cov, M, gen=None):
    """mean (N, d), cov (d, d) or (N, d, d)  ->  (N, M, d)."""
    L = torch.linalg.cholesky(cov + 1e-10 * torch.eye(cov.shape[-1], dtype=cov.dtype))
    z = torch.randn(mean.shape[0], M, mean.shape[1], dtype=mean.dtype, generator=gen)
    if L.dim() == 2:
        return mean[:, None, :] + z @ L.T
    return mean[:, None, :] + torch.einsum('nij,nmj->nmi', L, z)


def point_samples(pred):
    """A point forecast is a one-sample ensemble."""
    return pred[:, None, :]


def climatology_samples(bank, N, M, gen=None):
    """Ignore u; draw from the marginal of v (the no-skill reference)."""
    idx = torch.randint(0, bank.shape[0], (N, M), generator=gen)
    return bank[idx]


@torch.no_grad()
def sir_samples(score_fn, u, bank, M, chunk=256, gen=None):
    """Sample nu(v|u) ∝ exp(score(u, v)) mu_v by importance resampling a bank.

    score_fn(u (n, du), bank (B, dv)) -> (n, B) logits.
    Returns samples (N, M, dv) and the per-query effective sample size (N,).
    """
    out, ess = [], []
    for s in range(0, u.shape[0], chunk):
        logits = score_fn(u[s:s + chunk], bank).double()
        w = torch.softmax(logits, dim=1)
        ess.append(1.0 / (w ** 2).sum(1))
        idx = torch.multinomial(w, M, replacement=True, generator=gen)
        out.append(bank[idx])
    return torch.cat(out), torch.cat(ess)


# ---------------------------------------------------------------------------
# Score functions for the CLIP models in this repo (what the loss trains)
# ---------------------------------------------------------------------------

def clip_score_fn(model, inner_product, normalize=False):
    """Model logit s(u, v): cosine/inner-product (optionally on L2-normalised
    embeddings, as in real CLIP) or L2 tilting."""
    @torch.no_grad()
    def f(u, v):
        gu, gv = model.encode_u(u.float()), model.encode_v(v.float())
        s = model.logit_scale.exp()
        if inner_product:
            if normalize:
                gu = torch.nn.functional.normalize(gu, dim=-1)
                gv = torch.nn.functional.normalize(gv, dim=-1)
            return s * gu @ gv.T
        return -0.5 * s * torch.cdist(gu, gv) ** 2
    return f


# ---------------------------------------------------------------------------
# Regression baselines (models/mlp_regressor.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def regressor_samples(model, u, u_bank, v_bank, M):
    """Predictive samples (N, M, d) of a regression baseline, with the noise
    estimated on the bank (held-out pairs, not training data):
      MeanRegressor      predicted mean + Gaussian residual noise, one fixed covariance
      GaussianRegressor  predicted mean + per-step widths from the model, with the
                         residual correlation across steps (standardised residuals)
    """
    from models.mlp_regressor import GaussianRegressor
    if isinstance(model, GaussianRegressor):
        mu, sd = (x.double() for x in model.predict(u))
        mu_b, sd_b = (x.double() for x in model.predict(u_bank))
        R = torch.corrcoef(((v_bank - mu_b) / sd_b).T)
        return mu[:, None, :] + sd[:, None, :] * gaussian_samples(torch.zeros_like(mu), R, M)
    pred = model.predict(u).double()
    r = v_bank - model.predict(u_bank).double()
    return gaussian_samples(pred, r.T @ r / r.shape[0], M)
