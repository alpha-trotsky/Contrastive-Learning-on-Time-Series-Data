"""Sanity-check the non-Gaussian domains and measure their 'Gaussian gap'.

For each tier and knob value:
  * oracle calibration (coverage ~ nominal) -> the exact sampler is right;
  * ES of the best linear-Gaussian predictor N(uK_emp, C_{v|u,emp}) fitted on
    empirical second moments (what the Gaussian theory 'thinks' is optimal)
    minus ES of the oracle  ->  how much non-Gaussian structure there is for a
    model to capture.  Zero at the Gaussian end of each knob.

Usage (from test_bed/):  python -m analysis.domain_check
"""
import json
import math
from pathlib import Path

import torch
import yaml

from experiment import build_signal, build_modality
from signals.gaussian_process import GaussianProcess1D
from modalities.past_future import PastFutureModality
from modalities.nonlinear import StudentTPastFuture, WarpedPastFuture, DoubleWellPastFuture
from evaluation.predictors import gaussian_samples
from evaluation.scoring import score_forecast


def base_gp():
    """The linear test bed exactly as in config.yaml (held fixed: every knob's
    zero point IS this field setup)."""
    cfg = yaml.safe_load(open('config.yaml'))
    return build_modality(cfg['modality'], build_signal(cfg['signal']))


def linear_gaussian(u_tr, v_tr):
    """Best linear-Gaussian predictor from empirical moments."""
    mu_u, mu_v = u_tr.mean(0), v_tr.mean(0)
    U, V = u_tr - mu_u, v_tr - mu_v
    n = U.shape[0]
    Cuu, Cuv, Cvv = U.T @ U / n, U.T @ V / n, V.T @ V / n
    K = torch.linalg.solve(Cuu + 1e-9 * torch.eye(Cuu.shape[0], dtype=Cuu.dtype), Cuv)
    return lambda u: ((u - mu_u) @ K + mu_v), Cvv - Cuv.T @ K


def check(mod, n_train=100000, N=500, M=400):
    u_tr, v_tr = mod.sample_pair(n_train)
    u, y = mod.sample_pair(N)
    mean_fn, C = linear_gaussian(u_tr, v_tr)
    o = score_forecast(mod.oracle_samples(u, M), y)
    g = score_forecast(gaussian_samples(mean_fn(u), C, M), y)
    return {'oracle_es': o['energy'], 'lingauss_es': g['energy'],
            'gauss_gap': g['energy'] - o['energy'], 'gauss_gap_rel': (g['energy'] - o['energy']) / o['energy'],
            'oracle_cov90': o['cov90'], 'oracle_cal_err': o['cal_err'],
            'lingauss_cov90': g['cov90'], 'lingauss_cal_err': g['cal_err'],
            'mse_oracle_mean': o['mse_mean'], 'mse_lingauss_mean': g['mse_mean']}


def main():
    torch.manual_seed(0)
    out = {}
    base = base_gp()
    for df in [math.inf, 30, 10, 5, 3]:
        out[f't df={df}'] = check(StudentTPastFuture(base, df))
        print(f't df={df}', {k: round(v, 4) for k, v in out[f"t df={df}"].items()})
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out[f'warp k={k}'] = check(WarpedPastFuture(base, k))
        print(f'warp k={k}', {kk: round(v, 4) for kk, v in out[f"warp k={k}"].items()})
    for lam in [0.0, 0.5, 1.0, 1.5]:
        out[f'dwell lam={lam}'] = check(DoubleWellPastFuture(lam, n_particles=1000), n_train=20000, N=200, M=300)
        print(f'dwell lam={lam}', {kk: round(v, 4) for kk, v in out[f"dwell lam={lam}"].items()})
    Path('results/domains').mkdir(parents=True, exist_ok=True)
    json.dump(out, open('results/domains/domain_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
