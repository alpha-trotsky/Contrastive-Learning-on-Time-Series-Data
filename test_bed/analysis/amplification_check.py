"""Per-past check of the tail theorems on a trained score.

Trains one model exactly as analysis/tail_sweep.py does (heavy-tailed domain,
KL loss), then, past by past,

  1. measures the tail amplification  A_bar(u) = mean over directions of
     exp(s(u, c v_j)) / Z(u)  with the ray probe (bank futures scaled by c), and
     compares it with the population over-forecast factor of the events
     {|v| > t_p}, estimated with a bank of 10^6 futures.  The tail theorem
     says that the two agree as t_p grows;
  2. groups the pasts by A_bar(u) and records the over-forecast factor as a
     function of the bank size N in each group.  The exposure theorem says
     that each group follows min(N, A).

Usage (from test_bed/):
    python -m analysis.amplification_check --score tanh --encoder linear
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from experiment import build_model
from analysis import tail_sweep as T
from analysis.domain_sweep import build_domain

SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
PS = [1e-2, 1e-3, 1e-4]


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--score', default='tanh')
    p.add_argument('--encoder', default='linear')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--n_bank', type=int, default=1000000)
    p.add_argument('--out', default='results/amplification')
    a = p.parse_args()
    torch.set_num_threads(1)
    ns = argparse.Namespace(data='t', dial=3.0, seed=a.seed, n_train=200000, n_test=500, M=300, max_bank=1000)
    d = T.load_data(ns)
    u_tr, v_tr, u_te = d['u_tr'], d['v_tr'], d['u_te']
    loss_mode = T.DEFAULT_LOSS[a.score]

    torch.manual_seed(a.seed)
    cfg = yaml.safe_load(open('config.yaml'))
    mcfg = cfg['model']
    if a.encoder == 'mlp':
        mcfg.update(type='mlp_clip', hidden=256, depth=2, skip=True)
    model = build_model(mcfg, T._Dims(d['dim_u'], d['dim_v']))
    if a.score == 'tk':
        model.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))
    if hasattr(model, 'set_input_stats'):
        model.set_input_stats(u_tr.float(), v_tr.float())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(a.steps):
        idx = torch.randint(0, u_tr.shape[0], (256,))
        gu = model.encode_u(u_tr[idx].float())
        loss = T.kl_loss(T.logits_fn(model, a.score, gu, model.encode_v(v_tr[idx].float())), loss_mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    score_fn = T.make_score_fn(model, a.score)

    # a large bank drawn from the exact marginal of the future
    mod = build_domain('t', 3.0, cfg)
    torch.manual_seed(12345 + a.seed)
    L = torch.linalg.cholesky(mod.c2 * mod.C_vv())
    bank = ((torch.randn(a.n_bank, mod.dim_v, dtype=torch.float64) @ L.T) / mod._w(a.n_bank, mod.df).sqrt()).float()
    norms = bank.double().norm(dim=1)
    thr = d['thr']                                               # marginal quantiles for PS
    tail = torch.stack([norms > t for t in thr], 1).double()     # (N, 3)

    n = u_te.shape[0]
    # weights on the tail sets for nested banks: the first N futures, and disjoint blocks for the smaller sizes
    tw = {N: torch.zeros(n, len(PS), dtype=torch.float64) for N in SIZES if N <= a.n_bank}
    reps = {N: max(1, min(20, a.n_bank // N)) for N in tw}
    logZ = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, n, 20):
            lg = score_fn(u_te[s:s + 20], bank).double()                       # (20, N)
            logZ[s:s + 20] = torch.logsumexp(lg, 1) - math.log(a.n_bank)
            for N in tw:
                acc = 0
                for r in range(reps[N]):
                    blk = slice(r * N, (r + 1) * N)
                    w = torch.softmax(lg[:, blk], 1)
                    acc = acc + w @ tail[blk]
                tw[N][s:s + 20] = acc / reps[N]
        # ray probe: directions of ordinary bank futures and of the most extreme ones
        dirs = bank[:2000]
        top = bank[norms.topk(2000).indices]
        amp = {}
        for name, D in (('all', dirs), ('extreme', top / top.norm(dim=1, keepdim=True) * dirs.norm(dim=1).median())):
            for c in (10, 100, 1000):
                la = score_fn(u_te, D * c).double() - logZ[:, None]
                amp[f'{name}_{c}'] = torch.logsumexp(la, 1) - math.log(la.shape[1])   # log A_bar(u)

    Nmax = max(tw)
    of_pop = tw[Nmax] / torch.tensor(PS, dtype=torch.float64)                   # (n, 3) population over-forecast factor
    logA = amp['all_1000']
    res = {'args': vars(a), 'sizes': list(tw), 'reps': reps, 'ps': PS,
           'kappa': model.logit_scale.exp().item(),
           'spearman_logA_vs_OF': {str(pv): spearman(logA, of_pop[:, k]) for k, pv in enumerate(PS)},
           'spearman_c100_vs_c1000': spearman(amp['all_100'], amp['all_1000']),
           'spearman_all_vs_extreme_dirs': spearman(amp['all_1000'], amp['extreme_1000']),
           'log10_A': (logA / math.log(10)).tolist(),
           'log10_A_c10': (amp['all_10'] / math.log(10)).tolist(),
           'of_pop': of_pop.tolist(),
           'mean_A_capped1e6': logA.clamp(max=math.log(1e6)).exp().mean().item(),
           'mean_of_pop': of_pop.mean(0).tolist()}
    # groups of pasts by amplification
    edges = [-math.inf, 0.0, 1.0, 2.0, 3.0, math.inf]                          # log10 A
    l10 = logA / math.log(10)
    groups = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (l10 > lo) & (l10 <= hi)
        if m.sum() == 0:
            continue
        groups.append({'log10A_range': [lo, hi], 'n': int(m.sum()),
                       'A_geomean': math.exp(logA[m].mean().item()),
                       'A_mean': logA[m].exp().mean().item(),
                       'of_by_N': {str(pv): [tw[N][m, k].mean().item() / pv for N in tw] for k, pv in enumerate(PS)}})
    res['groups'] = groups
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.score}_{a.encoder}_s{a.seed}'
    json.dump(res, open(out / f'{tag}.json', 'w'))
    print(tag, 'kappa', res['kappa'], 'spearman', res['spearman_logA_vs_OF'], 'c100 vs c1000', res['spearman_c100_vs_c1000'],
          'dirs', res['spearman_all_vs_extreme_dirs'])
    print('mean A (capped 1e6)', res['mean_A_capped1e6'], 'mean OF at N =', Nmax, res['mean_of_pop'])
    for g in groups:
        print(f"log10 A in ({g['log10A_range'][0]}, {g['log10A_range'][1]}]  n={g['n']}  A geo-mean {g['A_geomean']:.3g}  mean {g['A_mean']:.3g}")
        for pv in ('0.001', '0.0001'):
            print(f"   p={pv}: OF by N {list(tw)} = " + ' '.join(f'{x:.3g}' for x in g['of_by_N'][pv]))


if __name__ == '__main__':
    main()
