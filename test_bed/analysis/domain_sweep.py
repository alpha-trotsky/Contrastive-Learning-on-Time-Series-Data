"""Goal 1 sweep: train the existing linear models on each non-Gaussian domain and
score them against that domain's perfect forecaster (oracle).

One run = (domain, dial, model).  Models: cosine CLIP (two-sided conditional,
inner product), L2 one-sided on p(v|u), MSE regression.  Contrastive models are
scored through their own score function only (SIR over a bank of training
futures); MSE as a point forecast and with Gaussian residual noise.

Usage (from test_bed/):
    python -m analysis.domain_sweep --domain t --dial 3 --model cosine
"""
import argparse
import json
import math
import zlib
from pathlib import Path

import torch
import yaml

from experiment import build_signal, build_modality, build_model, build_loss, resolve_theory_target
from modalities.nonlinear import StudentTPastFuture, WarpedPastFuture, DoubleWellPastFuture
from evaluation.predictors import (gaussian_samples, point_samples, climatology_samples,
                                   sir_samples, clip_score_fn)
from evaluation.scoring import score_forecast, add_skills
from evaluation.theory_match import theory_match_whitened_error

LOSSES = {
    'cosine': {'type': 'conditional', 'inner_product': True, 'lambda_u': 0.5, 'lambda_v': 0.5},
    'l2':     {'type': 'one_sided_u', 'inner_product': False, 'lambda_u': 1.0, 'lambda_v': 0.0},
    'mse':    {'type': 'mse'},
}


def build_domain(name, dial, cfg):
    if name == 'dwell':
        return DoubleWellPastFuture(dial, n_particles=1000)
    base = build_modality(cfg['modality'], build_signal(cfg['signal']))   # config.yaml, unchanged
    if name == 't':
        return StudentTPastFuture(base, dial)
    if name == 'warp':
        return WarpedPastFuture(base, dial)
    raise ValueError(name)


def make_data(mod, n, chunk=10000):
    us, vs = zip(*(mod.sample_pair(min(chunk, n - i)) for i in range(0, n, chunk)))
    return torch.cat(us), torch.cat(vs)


def cond_coverage(samples, y, spread, level=0.9, bins=3):
    """90% coverage within terciles of the oracle's predicted spread
    (low / mid / high uncertainty cases).  Marginal coverage can hide a model
    that is too wide on easy cases and too narrow on hard ones."""
    q = torch.tensor([(1 - level) / 2, (1 + level) / 2], dtype=samples.dtype)
    lo, hi = torch.quantile(samples, q, dim=1)
    per_query = ((y >= lo) & (y <= hi)).double().mean(1)
    edges = torch.quantile(spread, torch.linspace(0, 1, bins + 1, dtype=spread.dtype))
    out = []
    for b in range(bins):
        m = (spread >= edges[b]) & (spread <= edges[b + 1])
        out.append(per_query[m].mean().item())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['t', 'warp', 'dwell'], required=True)
    p.add_argument('--dial', type=float, required=True)
    p.add_argument('--model', choices=LOSSES, required=True)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--n_train', type=int, default=200000)
    p.add_argument('--n_test', type=int, default=500)
    p.add_argument('--M', type=int, default=300)
    p.add_argument('--bank', type=int, default=50000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/domain_sweep')
    a = p.parse_args()
    torch.set_num_threads(1)

    cfg = yaml.safe_load(open('config.yaml'))
    cfg['loss'] = LOSSES[a.model]
    # same data for every model at a given (domain, dial, seed)
    torch.manual_seed(1000 * a.seed + zlib.crc32(f'{a.domain}{a.dial}'.encode()) % 1000)
    mod = build_domain(a.domain, a.dial, cfg)
    u_tr, v_tr = make_data(mod, a.n_train)
    u_te, y = make_data(mod, a.n_test)
    oracle = mod.oracle_samples(u_te, a.M)
    spread = mod.oracle_spread(u_te)

    torch.manual_seed(a.seed)
    model = build_model(cfg['model'], mod)
    loss_fn = build_loss(cfg['loss'], model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for step in range(a.steps):
        idx = torch.randint(0, a.n_train, (a.batch,))
        u, v = u_tr[idx], v_tr[idx]
        gu, gv, s = model(u.float(), v.float())
        loss = loss_fn(gu, gv, s, u, v)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    bank = v_tr[:a.bank]
    S = {'oracle': oracle, 'climatology': climatology_samples(bank, a.n_test, a.M)}
    extra = {'final_loss': loss.item(), 'logit_scale': model.logit_scale.exp().item()}
    if a.model == 'mse':
        W = (model.u_encoder.weight.T @ model.v_encoder.weight).double().detach()
        pred = u_te @ W
        S['model_point'] = point_samples(pred)
        r = v_tr[:a.bank] - u_tr[:a.bank] @ W
        S['model'] = gaussian_samples(pred, r.T @ r / a.bank, a.M)
    else:
        S['model'], ess = sir_samples(clip_score_fn(model, LOSSES[a.model]['inner_product']),
                                      u_te, bank.float(), a.M)
        extra['ess_median'] = ess.median().item()
        extra['ess_min'] = ess.min().item()
    if a.domain == 't':
        # same covariances as the GP, so the Gaussian theory's A* still applies
        target = resolve_theory_target(cfg['loss'])
        w = theory_match_whitened_error(model, mod, target)
        extra['whitened_theory_err'] = None if w is None else w.item()

    table = add_skills({k: score_forecast(s, y) for k, s in S.items()})
    for k in ('oracle', 'model'):
        table[k]['cov90_by_spread'] = cond_coverage(S[k].double(), y.double(), spread.double())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.domain}_{a.dial:g}_{a.model}_s{a.seed}'
    json.dump({'args': vars(a), 'scores': table, 'extra': extra}, open(out / f'{tag}.json', 'w'), indent=1)
    m, o = table['model'], table['oracle']
    print(f"{tag}: energy model {m['energy']:.4f} oracle {o['energy']:.4f} clim {table['climatology']['energy']:.4f} "
          f"| cov90 {m['cov90']:.3f} by-spread {[round(x, 2) for x in m['cov90_by_spread']]} | {extra}")


if __name__ == '__main__':
    main()
