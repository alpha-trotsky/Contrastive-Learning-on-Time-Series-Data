"""Black-box probabilistic forecast scoring.

Replaces the cosine self/random similarity as the headline "prediction vs
ground truth" metric.  Every forecaster -- closed-form Gaussian, a contrastive
model, an MSE regressor, an oracle, a true black box -- is reduced to the same
object: M predictive SAMPLES of v for each conditioning u, shape (N, M, d).
A point forecaster is just M = 1.  All scores below compare those samples to
the single realised ground truth y = v, so they are equally valid on synthetic
data (where an oracle also exists) and on real data (where it does not).

Scores (lower is better unless noted):

  energy_score    ES(P, y) = E||X - y|| - 1/2 E||X - X'||.  Strictly proper for
                  multivariate forecasts; reduces to ||x_hat - y|| (Euclidean
                  error) for a point forecast, so point and probabilistic
                  engines sit on ONE scale.  Its expected excess over the
                  oracle is half the energy distance to the true conditional.
  crps            the 1-D energy score, per coordinate / horizon step.
  variogram_score sensitive to the dependence structure across coordinates
                  (which ES detects weakly).
  mse_mean        squared error of the predictive mean (the old forecast_mse,
                  but against the realised v, not a closed-form target).
  coverage        empirical coverage of central per-coordinate intervals at
                  several nominal levels (calibration), plus mean width
                  (sharpness).  Over-dispersed models over-cover.
  pit             probability integral transform values; flat histogram iff
                  marginally calibrated.

All estimators use the "fair" (unbiased in M) form so ensembles of different
size are comparable.
"""
import torch

LEVELS = (0.5, 0.8, 0.9, 0.95)


def energy_score(samples, y):
    """samples (N, M, d), y (N, d)  ->  (N,) fair energy score."""
    N, M, d = samples.shape
    t1 = torch.linalg.vector_norm(samples - y[:, None, :], dim=-1).mean(1)
    if M == 1:
        return t1
    t2 = torch.cdist(samples, samples).sum((1, 2)) / (M * (M - 1))
    return t1 - 0.5 * t2


def crps(samples, y):
    """Per-coordinate fair CRPS.  samples (N, M, d), y (N, d)  ->  (N, d)."""
    N, M, d = samples.shape
    t1 = (samples - y[:, None, :]).abs().mean(1)
    if M == 1:
        return t1
    xs, _ = samples.sort(dim=1)
    i = torch.arange(1, M + 1, dtype=samples.dtype, device=samples.device)[None, :, None]
    # sum_{i,j} |x_i - x_j| = 2 sum_i (2i - M - 1) x_(i)
    pair = 2 * ((2 * i - M - 1) * xs).sum(1)
    return t1 - 0.5 * pair / (M * (M - 1))


def variogram_score(samples, y, p=0.5, max_lag=None, chunk=32):
    """Variogram score of order p over coordinate pairs (i<j, |i-j| <= max_lag),
    pair weights 1/|i-j|.  samples (N, M, d), y (N, d)  ->  (N,)."""
    N, M, d = samples.shape
    ii, jj = torch.triu_indices(d, d, offset=1)
    lag = (jj - ii).to(samples.dtype)
    if max_lag is not None:
        keep = lag <= max_lag
        ii, jj, lag = ii[keep], jj[keep], lag[keep]
    w = 1.0 / lag
    out = []
    for s in range(0, N, chunk):
        X, Y = samples[s:s + chunk], y[s:s + chunk]
        ex = (X[:, :, ii] - X[:, :, jj]).abs().pow(p).mean(1)          # (n, P)
        ob = (Y[:, ii] - Y[:, jj]).abs().pow(p)                         # (n, P)
        out.append(((ob - ex) ** 2 * w).sum(1))
    return torch.cat(out)


def pit(samples, y):
    """Randomised PIT per coordinate.  -> (N, d) in [0, 1]."""
    below = (samples < y[:, None, :]).float().mean(1)
    equal = (samples == y[:, None, :]).float().mean(1)
    return below + equal * torch.rand_like(below)


def coverage(samples, y, levels=LEVELS):
    """Central-interval coverage and width per nominal level (pooled over N, d)."""
    out = {}
    for lv in levels:
        q = torch.tensor([(1 - lv) / 2, (1 + lv) / 2], dtype=samples.dtype, device=samples.device)
        lo, hi = torch.quantile(samples, q, dim=1)                      # (N, d) each
        out[lv] = (((y >= lo) & (y <= hi)).float().mean().item(), (hi - lo).mean().item())
    return out


def score_forecast(samples, y):
    """All scores for one forecaster.  Returns a flat dict of floats."""
    samples, y = samples.double(), y.double()
    M = samples.shape[1]
    mean = samples.mean(1)
    res = {
        'energy': energy_score(samples, y).mean().item(),
        'crps': crps(samples, y).mean().item(),
        'variogram': variogram_score(samples, y).mean().item(),
        'mse_mean': ((mean - y) ** 2).mean().item(),
    }
    if M > 1:
        cov = coverage(samples, y)
        for lv, (c, w) in cov.items():
            res[f'cov{int(lv * 100)}'] = c
            res[f'width{int(lv * 100)}'] = w
        # calibration error: mean |empirical - nominal| coverage over levels
        res['cal_err'] = sum(abs(c - lv) for lv, (c, _) in cov.items()) / len(cov)
    return res


def add_skills(table, clim='climatology', oracle='oracle'):
    """Add skill scores relative to climatology (1 - S/S_clim, higher is better)
    and the excess over the oracle (S - S_oracle >= 0 in expectation)."""
    for name, r in table.items():
        for k in ('energy', 'crps', 'variogram', 'mse_mean'):
            if clim in table:
                r[f'{k}_skill'] = 1 - r[k] / table[clim][k]
            if oracle in table:
                r[f'{k}_excess'] = r[k] - table[oracle][k]
    return table
