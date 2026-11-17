"""Tail sweep: train one score, freeze it, and measure how its bank forecast
treats extreme futures as the bank grows.

Extends analysis/bank_sweep.py (same data, same seeds, same training loop for
the KL losses) with

  * two more score families
      tanh   k <f(u), tanh(g(v))>            bounded, saturates along rays
      tk     -(b/2) log(1 + k |f-g|^2 / b)   Student-t kernel: decays like
                                             -b log|v|  (b learnable, b -> inf is L2)
      tk2    tk + (b_u/2) log(1+|G0 u|^2) + (b_v/2) log(1+|H0 v|^2)
                                             log-quadratic family: contains log r of an
                                             elliptical t-process with LINEAR maps
  * energy-score training.  --train es replaces the KL (InfoNCE) loss by the
    energy score of the model's own bank forecast, built from a bank of K
    training futures drawn afresh at every step; --train hyb adds it to the KL
    loss with weight --lam.  This is the energy-distance loss of
    Baptista-Stuart-Tran Rem. 3.5 for the forecasting direction, estimated
    with a K-future bank.
  * tail diagnostics for the frozen score
      tail weight   mean bank weight on {|v| > t_p}, t_p the (1-p) marginal
                    quantile of |v|, for every bank size.  Divided by p this is
                    the over-forecast factor of the extreme event (1 = calibrated
                    on average; Theorem: at most N, and it saturates at the
                    tail amplification).
      ray probe     s(u, c v_j) - log Z(u) for c = 1 .. 1000: the tail
                    amplification of the score, measured with no tail data.

Usage (from test_bed/):
    python -m analysis.tail_sweep --score tanh --encoder mlp --seed 0
    python -m analysis.tail_sweep --score tanh --encoder mlp --train hyb --lam 3
    python -m analysis.tail_sweep --data btc --score l2 --loss joint
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from experiment import build_model
from analysis.domain_sweep import build_domain, make_data
from evaluation.scoring import energy_score

BANK_SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000, 200000]
TAIL_P = [1e-2, 1e-3, 1e-4]
RAY_C = [1, 3, 10, 100, 1000]
# default KL loss per score: the variants used in the note's tables
DEFAULT_LOSS = {'dot': 'twoway', 'tanh': 'twoway', 'ncos': 'twoway', 'l2': 'oneway', 'tk': 'oneway', 'tk2': 'oneway'}


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def logits_fn(model, kind, gu, gv):
    """Score matrix s(u_i, v_j) from embeddings gu (n, k), gv (m, k)."""
    k = model.logit_scale.exp()
    if kind == 'dot':
        return k * gu @ gv.T
    if kind == 'tanh':
        return k * gu @ torch.tanh(gv).T
    if kind == 'ncos':
        return k * F.normalize(gu, dim=-1) @ F.normalize(gv, dim=-1).T
    eu = ev = None
    if kind == 'tk2':          # last columns of the embeddings carry G0 u and H0 v
        gu, eu = gu[:, :model.k], gu[:, model.k:]
        gv, ev = gv[:, :model.k], gv[:, model.k:]
    d2 = (gu * gu).sum(-1, keepdim=True) - 2 * gu @ gv.T + (gv * gv).sum(-1)[None, :]
    d2 = d2.clamp_min(0)
    if kind == 'l2':
        return -0.5 * k * d2
    if kind in ('tk', 'tk2'):
        b = 1 + model.log_beta.exp()
        s = -0.5 * b * torch.log1p(k * d2 / b)
        if kind == 'tk2':
            s = s + 0.5 * model.log_bu.exp() * torch.log1p((eu * eu).sum(-1))[:, None] \
                  + 0.5 * model.log_bv.exp() * torch.log1p((ev * ev).sum(-1))[None, :]
        return s
    raise ValueError(kind)


def make_score_fn(model, kind):
    @torch.no_grad()
    def f(u, v):
        return logits_fn(model, kind, model.encode_u(u.float()), model.encode_v(v.float()))
    return f


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def kl_loss(logits, mode):
    """InfoNCE-type losses on an in-batch score matrix (rows = pasts)."""
    n = logits.shape[0]
    lab = torch.arange(n)
    if mode == 'oneway':
        return F.cross_entropy(logits, lab)
    if mode == 'twoway':
        return 0.5 * F.cross_entropy(logits, lab) + 0.5 * F.cross_entropy(logits.T, lab)
    if mode == 'joint':
        off = ~torch.eye(n, dtype=torch.bool)
        return -(logits.diagonal().mean() - torch.logsumexp(logits[off], dim=0) + math.log(n))
    raise ValueError(mode)


def es_loss(logits, d_pos, D):
    """Energy score of the bank forecast, one per past.

    logits (B, K) scores of K bank futures, d_pos (B, K) = |v_k - y_i|,
    D (K, K) = |v_k - v_l|.  ES_i = sum_k w_ik d_pos_ik - 1/2 sum_kl w_ik w_il D_kl.
    """
    W = torch.softmax(logits, dim=1)
    return ((W * d_pos).sum(1) - 0.5 * ((W @ D) * W).sum(1)).mean()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data(a):
    """-> dict(u_tr, v_tr, u_te, y, v_bk, oracle or None, ref_norms)."""
    if a.data == 't':
        cfg = yaml.safe_load(open('config.yaml'))
        # identical data to analysis.bank_sweep / domain_sweep at the same (dial, seed)
        torch.manual_seed(1000 * a.seed + zlib.crc32(f't{a.dial}'.encode()) % 1000)
        mod = build_domain('t', a.dial, cfg)
        u_tr, v_tr = make_data(mod, a.n_train)
        u_te, y = make_data(mod, a.n_test)
        oracle = mod.oracle_samples(u_te, a.M)
        _, v_bk = make_data(mod, a.max_bank)
        # marginal quantiles of |v| and the oracle's exceedance probabilities
        # 1e6 futures drawn directly from the exact marginal (cheaper than the field sampler)
        L = torch.linalg.cholesky(mod.c2 * mod.C_vv())
        ref = ((torch.randn(1000000, mod.dim_v, dtype=torch.float64) @ L.T) / mod._w(1000000, mod.df).sqrt()).norm(dim=1)
        thr = torch.stack([torch.kthvalue(ref, int(round((1 - p) * ref.numel()))).values for p in TAIL_P])
        big = torch.cat([mod.oracle_samples(u_te, 2000).norm(dim=-1) for _ in range(2)], 1)   # (n_test, 4000)
        p_true = torch.stack([(big > t).double().mean(1) for t in thr], 1)                # (n_test, 3)
        return dict(u_tr=u_tr, v_tr=v_tr, u_te=u_te, y=y, v_bk=v_bk, oracle=oracle, thr=thr,
                    p_true=p_true, dim_u=mod.dim_u, dim_v=mod.dim_v)
    from modalities.heavy_real import load, DEFAULTS
    kw = dict(DEFAULTS[a.data])
    d = load(a.data, **kw)
    g = torch.Generator().manual_seed(a.seed)
    te = torch.randperm(d['u_test'].shape[0], generator=g)[:a.n_test]
    v_bk = d['v_bank'][torch.randperm(d['v_bank'].shape[0], generator=g)]
    ref = torch.cat([d['v_train'], d['v_bank']]).norm(dim=1)
    thr = torch.stack([torch.kthvalue(ref, int(round((1 - p) * ref.numel()))).values for p in TAIL_P])
    return dict(u_tr=d['u_train'], v_tr=d['v_train'], u_te=d['u_test'][te], y=d['v_test'][te], v_bk=v_bk,
                oracle=None, thr=thr, p_true=None, dim_u=d['u_train'].shape[1], dim_v=d['v_train'].shape[1])


class LogQuadratic(nn.Module):
    """Wrap a CLIP model with a past-only and a future-only linear map (score 'tk2')."""

    def __init__(self, base, du, dv):
        super().__init__()
        self.base, self.k = base, None
        self.hu = nn.Linear(du, du, bias=False)
        self.hv = nn.Linear(dv, dv, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))
        self.log_bu = nn.Parameter(torch.tensor(0.0))
        self.log_bv = nn.Parameter(torch.tensor(0.0))

    @property
    def logit_scale(self):
        return self.base.logit_scale

    def set_input_stats(self, u, v):
        if hasattr(self.base, 'set_input_stats'):
            self.base.set_input_stats(u, v)

    def encode_u(self, u):
        g = self.base.encode_u(u)
        self.k = g.shape[1]
        return torch.cat([g, self.hu(u)], 1)

    def encode_v(self, v):
        g = self.base.encode_v(v)
        self.k = g.shape[1]
        return torch.cat([g, self.hv(v)], 1)


class _Dims:
    def __init__(self, du, dv):
        self.dim_u, self.dim_v = du, dv


# ---------------------------------------------------------------------------
# evaluation of a frozen score
# ---------------------------------------------------------------------------

@torch.no_grad()
def bank_eval(score_fn, u, y, bank, thr, M, chunk=125):
    """One bank: forecast samples' energy score and weight diagnostics."""
    norms = bank.double().norm(dim=1)
    tail = torch.stack([(norms > t) for t in thr], 1).double()          # (N, 3)
    order = norms.argsort()
    rank = torch.empty_like(norms)
    rank[order] = torch.arange(norms.numel(), dtype=norms.dtype) / max(norms.numel() - 1, 1)
    es, ess, wmax, toprank, tw, lz = [], [], [], [], [], []
    for s in range(0, u.shape[0], chunk):
        logits = score_fn(u[s:s + chunk], bank).double()
        lz.append(torch.logsumexp(logits, 1) - math.log(bank.shape[0]))
        w = torch.softmax(logits, dim=1)
        ess.append(1.0 / (w ** 2).sum(1))
        m = w.max(1)
        wmax.append(m.values)
        toprank.append(rank[m.indices])
        tw.append(w @ tail)
        idx = torch.multinomial(w, M, replacement=True)
        es.append(energy_score(bank[idx].double(), y[s:s + chunk].double()))
    cat = torch.cat
    return dict(es=cat(es), ess=cat(ess), wmax=cat(wmax), toprank=cat(toprank), tw=cat(tw), logZ=cat(lz))


@torch.no_grad()
def ray_probe(score_fn, u, dirs, logZ):
    """log tail amplification along rays: s(u_i, c v_j) - log Z(u_i).

    For each scale c: the log of the direction-averaged amplification
    (log-mean-exp over the direction sample), and the largest one."""
    out = {}
    for c in RAY_C:
        la = score_fn(u, dirs * c).double() - logZ[:, None]              # (n, J)
        lmean = torch.logsumexp(la, 1) - math.log(la.shape[1])
        out[str(c)] = {
            'log10_mean_amp_median': (lmean.median() / math.log(10)).item(),
            'log10_mean_amp_q90': (torch.quantile(lmean, 0.9) / math.log(10)).item(),
            'mean_amp_mean_capped': lmean.clamp(max=math.log(1e12)).exp().mean().item(),
            'frac_pasts_mean_amp_gt1': (lmean > 0).double().mean().item(),
            'frac_pairs_amp_gt1': (la > 0).double().mean().item(),
            'log10_max_amp_median': (la.max(1).values.median() / math.log(10)).item(),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='t', help="'t' (Student-t process) or a modalities.heavy_real name")
    p.add_argument('--dial', type=float, default=3)
    p.add_argument('--score', choices=list(DEFAULT_LOSS), required=True)
    p.add_argument('--loss', choices=['oneway', 'twoway', 'joint'], default=None)
    p.add_argument('--train', choices=['kl', 'es', 'hyb'], default='kl')
    p.add_argument('--lam', type=float, default=1.0, help='weight of the energy-score term in --train hyb')
    p.add_argument('--es_bank', type=int, default=1024, help='bank futures per step in the energy-score term')
    p.add_argument('--clip', type=float, default=0.0, help='gradient-norm clip (0 = off)')
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
    p.add_argument('--reps', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/tail_sweep')
    p.add_argument('--save_model', action='store_true')
    p.add_argument('--force', action='store_true', help='rerun even if the result file exists')
    a = p.parse_args()
    a.loss = a.loss or DEFAULT_LOSS[a.score]
    torch.set_num_threads(1)
    t0 = time.time()
    out = Path(a.out)
    tag = f"{a.data}{a.dial:g}_{a.score}-{a.loss}_{a.train}" if a.data == 't' else f"{a.data}_{a.score}-{a.loss}_{a.train}"
    if a.train == 'hyb':
        tag += f'{a.lam:g}'
    tag += f'_{a.encoder}_s{a.seed}'
    if (out / f'{tag}.json').exists() and not a.force:
        print(f'{tag}: exists, skipping')
        return

    d = load_data(a)
    u_tr, v_tr = d['u_tr'], d['v_tr']
    n_train = u_tr.shape[0]

    torch.manual_seed(a.seed)
    cfg = yaml.safe_load(open('config.yaml'))['model']
    if a.encoder == 'mlp':
        cfg.update(type='mlp_clip', hidden=a.hidden, depth=a.depth, skip=True)
    model = build_model(cfg, _Dims(d['dim_u'], d['dim_v']))
    if a.score == 'tk':
        model.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))
    if a.score == 'tk2':
        model = LogQuadratic(model, d['dim_u'], d['dim_v'])
    if hasattr(model, 'set_input_stats'):
        model.set_input_stats(u_tr.float(), v_tr.float())
    # energy scores are reported in units of the climatological spread E|v - v'|
    es_unit = torch.cdist(v_tr[:2000], v_tr[2000:4000]).mean().item()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = []
    for step in range(a.steps):
        idx = torch.randint(0, n_train, (a.batch,))
        u, v = u_tr[idx].float(), v_tr[idx].float()
        gu = model.encode_u(u)
        loss = torch.zeros(())
        if a.train in ('kl', 'hyb'):
            l_kl = kl_loss(logits_fn(model, a.score, gu, model.encode_v(v)), a.loss)
            loss = loss + l_kl
        if a.train in ('es', 'hyb'):
            vb = v_tr[torch.randint(0, n_train, (a.es_bank,))].float()
            l_es = es_loss(logits_fn(model, a.score, gu, model.encode_v(vb)),
                           torch.cdist(v, vb), torch.cdist(vb, vb)) / es_unit
            loss = loss + (a.lam if a.train == 'hyb' else 1.0) * l_es
        opt.zero_grad()
        loss.backward()
        if a.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            hist.append({'step': step, 'loss': loss.item(),
                         'kl': l_kl.item() if a.train != 'es' else None,
                         'es': l_es.item() if a.train != 'kl' else None})
    model.eval()
    t_train = time.time() - t0

    score_fn = make_score_fn(model, a.score)
    u_te, y, v_bk, thr = d['u_te'], d['y'], d['v_bk'], d['thr']
    max_bank = min(a.max_bank, v_bk.shape[0])
    if d['oracle'] is not None:
        ref_es = energy_score(d['oracle'].double(), y.double())
        ref_name = 'oracle'
    else:   # no oracle on real data: reference = climatology (uniform weights on a 20k bank)
        cl = v_bk[torch.randint(0, v_bk.shape[0], (u_te.shape[0], a.M))]
        ref_es = energy_score(cl.double(), y.double())
        ref_name = 'climatology'
    ref_mean = ref_es.mean().item()
    calm = u_te.norm(dim=1) <= u_te.norm(dim=1).median()      # the calmer half of the test pasts
    rows, last = [], None
    for B in BANK_SIZES + ([max_bank] if max_bank not in BANK_SIZES else []):
        if B > max_bank:
            continue
        reps = max(1, min(a.reps, max_bank // B))
        R = [bank_eval(score_fn, u_te, y, v_bk[r * B:(r + 1) * B], thr, a.M) for r in range(reps)]
        last = R[0]
        mean = lambda f: sum(f(r) for r in R) / len(R)
        es_reps = [r['es'].mean().item() for r in R]
        row = {'bank': B, 'reps': reps,
               'excess_pct': 100 * (sum(es_reps) / reps - ref_mean) / ref_mean,
               'excess_pct_reps': [100 * (e - ref_mean) / ref_mean for e in es_reps],
               'ess_median': mean(lambda r: r['ess'].median().item()),
               'wmax_mean': mean(lambda r: r['wmax'].mean().item()),
               'frac_wmax_gt_half': mean(lambda r: (r['wmax'] > 0.5).double().mean().item()),
               'toprank_mean': mean(lambda r: r['toprank'].mean().item()),
               'tail_weight': [mean(lambda r: r['tw'][:, k].mean().item()) for k in range(len(TAIL_P))],
               'tail_weight_calm': [mean(lambda r: r['tw'][calm, k].mean().item()) for k in range(len(TAIL_P))]}
        rows.append(row)
        print(f"B={B:>6}  excess {row['excess_pct']:+8.2f}%  ESS med {row['ess_median']:9.1f}  max w {row['wmax_mean']:.3f}  "
              f"over-forecast x{row['tail_weight'][1] / TAIL_P[1]:8.2f} (p=1e-3)", flush=True)
    # ray probe, with log Z(u) from the largest bank
    J = min(1000, v_bk.shape[0])
    probe = ray_probe(score_fn, u_te, v_bk[-J:].float(), last['logZ'])

    out.mkdir(parents=True, exist_ok=True)
    res = {'args': vars(a), 'tag': tag, 'ref': ref_name, 'ref_energy': ref_mean, 'es_unit': es_unit,
           'logit_scale': model.logit_scale.exp().item(),
           'beta': (1 + model.log_beta.exp()).item() if a.score in ('tk', 'tk2') else None,
           'beta_uv': [model.log_bu.exp().item(), model.log_bv.exp().item()] if a.score == 'tk2' else None,
           'tail_p': TAIL_P, 'thr': thr.tolist(),
           'p_true_mean': d['p_true'].mean(0).tolist() if d['p_true'] is not None else None,
           'p_true_calm': d['p_true'][calm].mean(0).tolist() if d['p_true'] is not None else None,
           'rows': rows, 'ray_probe': probe, 'hist': hist,
           'time_train': t_train, 'time_total': time.time() - t0}
    json.dump(res, open(out / f'{tag}.json', 'w'), indent=1)
    if a.save_model:
        (out / 'models').mkdir(exist_ok=True)
        torch.save(model.state_dict(), out / 'models' / f'{tag}.pt')
    print(f'{tag}: train {t_train:.0f}s total {time.time() - t0:.0f}s  kappa {res["logit_scale"]:.2f}  '
          f'ray probe c=1000: {probe["1000"]}')


if __name__ == '__main__':
    main()
