"""Bank-size sweep: does the forecast improve or degrade as the bank of real
futures grows?  (Theory note, Part I: bounded scores should improve with the
bank; an unbounded score on heavy tails should collapse onto ever more extreme
futures.)

One run = (domain, dial, model, encoder, seed).  The model is trained exactly as
in analysis/domain_sweep.py (same data, same seeds), then frozen.  The forecast
is rebuilt from held-out banks of increasing size; for each size we average over
several disjoint banks and record the energy score against the oracle, the
largest bank weight and the effective sample size.

Extra model 'cosine_tanh': the unnormalised dot product (two-sided, as 'cosine')
with a tanh on the future encoder's output.  The score is then bounded in the
future for every past, without normalising anything.

Usage (from test_bed/):
    python -m analysis.bank_sweep --domain t --dial 3 --model cosine --encoder mlp
"""
import argparse
import json
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from experiment import build_model, build_loss
from analysis.domain_sweep import LOSSES, build_domain, make_data
from evaluation.predictors import sir_samples, clip_score_fn
from evaluation.scoring import score_forecast

SWEEP_LOSSES = dict(LOSSES, cosine_tanh=LOSSES['cosine'])
BANK_SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000, 200000]


class BoundedFuture(nn.Module):
    """Wrap a CLIP model so the future embedding is tanh(g(v)): bounded, not normalised."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def logit_scale(self):
        return self.base.logit_scale

    def set_input_stats(self, u, v):
        if hasattr(self.base, 'set_input_stats'):
            self.base.set_input_stats(u, v)

    def encode_u(self, u):
        return self.base.encode_u(u)

    def encode_v(self, v):
        return torch.tanh(self.base.encode_v(v))

    def forward(self, u, v):
        return self.encode_u(u), self.encode_v(v), self.logit_scale.exp()


@torch.no_grad()
def bank_stats(score_fn, u, bank, M):
    """Forecast samples from one bank, plus weight-concentration diagnostics."""
    S, ess = sir_samples(score_fn, u, bank, M)
    wmax, top = [], []
    for s in range(0, u.shape[0], 256):
        w = torch.softmax(score_fn(u[s:s + 256], bank).double(), dim=1)
        m = w.max(1)
        wmax.append(m.values)
        top.append(m.indices)
    return S, ess, torch.cat(wmax), torch.cat(top)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['t', 'warp', 'dwell'], default='t')
    p.add_argument('--dial', type=float, default=3)
    p.add_argument('--model', choices=[k for k in SWEEP_LOSSES if k not in ('mse', 'gauss')], required=True)
    p.add_argument('--encoder', choices=['linear', 'mlp'], default='linear')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--n_train', type=int, default=200000)
    p.add_argument('--n_test', type=int, default=500)
    p.add_argument('--M', type=int, default=300)
    p.add_argument('--max_bank', type=int, default=max(BANK_SIZES))
    p.add_argument('--reps', type=int, default=4, help='disjoint banks per size (fewer if they do not fit)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/bank_sweep')
    a = p.parse_args()
    torch.set_num_threads(1)

    cfg = yaml.safe_load(open('config.yaml'))
    cfg['loss'] = SWEEP_LOSSES[a.model]
    # identical data to analysis.domain_sweep at the same (domain, dial, seed)
    torch.manual_seed(1000 * a.seed + zlib.crc32(f'{a.domain}{a.dial}'.encode()) % 1000)
    mod = build_domain(a.domain, a.dial, cfg)
    u_tr, v_tr = make_data(mod, a.n_train)
    u_te, y = make_data(mod, a.n_test)
    oracle = mod.oracle_samples(u_te, a.M)
    _, v_bk = make_data(mod, a.max_bank)

    torch.manual_seed(a.seed)
    if a.encoder == 'mlp':
        cfg['model'].update(type='mlp_clip', hidden=a.hidden, depth=a.depth, skip=True)
    model = build_model(cfg['model'], mod)
    if a.model == 'cosine_tanh':
        model = BoundedFuture(model)
    loss_fn = build_loss(cfg['loss'], model)
    if hasattr(model, 'set_input_stats'):
        model.set_input_stats(u_tr.float(), v_tr.float())
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

    L = SWEEP_LOSSES[a.model]
    score_fn = clip_score_fn(model, L['inner_product'], L.get('normalize', False))
    oracle_es = score_forecast(oracle, y)['energy']
    rows = []
    for B in BANK_SIZES:
        if B > a.max_bank:
            break
        reps = max(1, min(a.reps, a.max_bank // B))
        es, ess_med, wmax_mean, frac_collapsed = [], [], [], []
        for r in range(reps):
            bank = v_bk[r * B:(r + 1) * B].float()
            S, ess, wmax, top = bank_stats(score_fn, u_te, bank, a.M)
            es.append(score_forecast(S, y)['energy'])
            ess_med.append(ess.median().item())
            wmax_mean.append(wmax.mean().item())
            frac_collapsed.append((wmax > 0.5).double().mean().item())
        # where do collapsed forecasts land?  percentile of the winning future's size in the bank
        norms = bank.double().norm(dim=1)
        rank = (norms[None, :] < norms[top][:, None]).double().mean(1)
        collapsed = wmax > 0.5
        mean = lambda xs: sum(xs) / len(xs)
        rows.append({'bank': B, 'reps': reps,
                     'excess_pct': 100 * (mean(es) - oracle_es) / oracle_es,
                     'excess_pct_reps': [100 * (e - oracle_es) / oracle_es for e in es],
                     'ess_median': mean(ess_med), 'wmax_mean': mean(wmax_mean),
                     'frac_wmax_gt_half': mean(frac_collapsed),
                     'collapsed_top_norm_pct': rank[collapsed].mean().item() if collapsed.any() else None,
                     'all_top_norm_pct': rank.mean().item()})
        print(f"B={B:>6}  excess {rows[-1]['excess_pct']:+7.2f}%  ESS med {rows[-1]['ess_median']:9.1f}  "
              f"max w {rows[-1]['wmax_mean']:.3f}  frac(max w>0.5) {rows[-1]['frac_wmax_gt_half']:.3f}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.domain}_{a.dial:g}_{a.model}_{a.encoder}_s{a.seed}'
    json.dump({'args': vars(a), 'oracle_energy': oracle_es, 'logit_scale': model.logit_scale.exp().item(),
               'final_loss': loss.item(), 'rows': rows}, open(out / f'{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
