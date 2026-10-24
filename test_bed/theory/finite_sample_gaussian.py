"""Finite-sample study of the linear-Gaussian setting of Baptista-Stuart-Tran.

The population results (notes/score_tails, Sec. 3.2) say which loss is exact
with infinite data.  Here we train on n samples instead and ask how far the
fitted forecast is from the truth.

Setting.  u, v in R^d, each N(0, I), with C_uv = rho * I (d independent
copies of a scalar pair with correlation rho; the model does not know this).

Scores.  Linear encoders give quadratic scores
    s(u, v) = -1/2 u'Ru + u'Av - 1/2 v'Sv
    dot   R = S = 0
    l2    R = G'G, A = G'H, S = H'H      (-1/2 |Gu - Hv|^2, G, H in R^{2d x d})
    quad  R, S PSD and A free            (dot + past-only + future-only terms)

Empirical losses.  The measures in each loss are replaced by empirical ones on
the n training pairs (full batch, every pair is a negative):
    1way   -mean_i s_ii + mean_i log mean_j exp s_ij        (InfoNCE)
    2way   1/2 (1way + the same with i and j swapped)        (CLIP)
    joint  -mean_i s_ii + log mean_ij exp s_ij

Evaluation.  The fitted score is scored by the exact forecast error
    J(s) = E_u KL( mu_{v|u} || nu^s_{v|u} )   (nats; 0 = perfect)
using the population marginal of v, i.e. the limit of a large bank.  No
sampling noise enters the evaluation.  J at the population minimiser of each
loss is its bias; E J(fitted) - bias is the price of finite n.

Each training set is shared by all (family, loss) pairs, so differences between
losses are paired.  Fits start at the population minimiser of the same
(family, loss), so this measures statistical error, not optimisation failure.

Usage (from test_bed/):
    python -m theory.finite_sample_gaussian            # full grid
    python -m theory.finite_sample_gaussian --quick    # smoke test
"""
import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

OUT = Path('results/theory')
FAMILIES = ['dot', 'l2', 'quad']
LOSSES = ['1way', '2way', 'joint']


# ---------------------------------------------------------------------------
# Exact Gaussian quantities
# ---------------------------------------------------------------------------

def logdet(M):
    L, info = torch.linalg.cholesky_ex(0.5 * (M + M.T))
    if info.item() != 0:
        return torch.tensor(float('nan'))
    return 2 * torch.log(torch.diagonal(L)).sum()


def cond_kl(P, M, K, C_cond, C_in):
    """E_x KL( N(Kx, C_cond) || N(Mx, P^{-1}) ),  x ~ N(0, C_in)."""
    d = C_cond.shape[0]
    E = K - M
    return 0.5 * (torch.trace(P @ C_cond) + torch.trace(E.T @ P @ E @ C_in)
                  - d - logdet(P) - logdet(C_cond))


class Gaussian:
    def __init__(self, d, rho):
        self.d, self.rho = d, rho
        I = torch.eye(d)
        self.C_uu, self.C_vv, self.C_uv = I, I, rho * I
        self.K = rho * I                      # E[v|u] = K u and E[u|v] = K v
        self.C_cond = (1 - rho ** 2) * I      # Cov[v|u] = Cov[u|v]
        self.C = torch.cat([torch.cat([I, rho * I], 1), torch.cat([rho * I, I], 1)], 0)
        self.chol = torch.linalg.cholesky(self.C)
        self.mi = -0.5 * d * math.log(1 - rho ** 2)

    def sample(self, n, gen):
        z = torch.randn(n, 2 * self.d, generator=gen) @ self.chol.T
        return z[:, :self.d], z[:, self.d:]

    def kl_v_given_u(self, R, A, S):
        P = torch.eye(self.d) + S
        return cond_kl(P, torch.linalg.solve(P, A.T), self.K, self.C_cond, self.C_uu)

    def kl_u_given_v(self, R, A, S):
        P = torch.eye(self.d) + R
        return cond_kl(P, torch.linalg.solve(P, A), self.K, self.C_cond, self.C_vv)

    def kl_joint(self, R, A, S):
        Q = torch.cat([torch.cat([R, -A], 1), torch.cat([-A.T, S], 1)], 0)
        L = torch.eye(2 * self.d) + Q
        return 0.5 * (torch.trace(L @ self.C) - 2 * self.d - logdet(L) - logdet(self.C))

    def population(self, loss, R, A, S):
        if loss == '1way':
            return self.kl_v_given_u(R, A, S)
        if loss == '2way':
            return 0.5 * (self.kl_v_given_u(R, A, S) + self.kl_u_given_v(R, A, S))
        if loss == 'joint':
            return self.kl_joint(R, A, S)
        raise ValueError(loss)


# ---------------------------------------------------------------------------
# Score families
# ---------------------------------------------------------------------------

def init_params(fam, d, seed):
    gen = torch.Generator().manual_seed(seed)
    if fam == 'dot':
        return [0.5 * torch.eye(d) + 0.05 * torch.randn(d, d, generator=gen)]
    if fam == 'l2':
        return [0.3 * torch.randn(2 * d, d, generator=gen) for _ in range(2)]
    if fam == 'quad':
        return [0.1 * torch.randn(d, d, generator=gen) for _ in range(3)]
    raise ValueError(fam)


def unpack(fam, p):
    if fam == 'dot':
        d = p[0].shape[0]
        return torch.zeros(d, d), p[0], torch.zeros(d, d)
    if fam == 'l2':
        G, H = p
        return G.T @ G, G.T @ H, H.T @ H
    if fam == 'quad':
        N, A, M = p
        return N @ N.T, A, M @ M.T
    raise ValueError(fam)


# ---------------------------------------------------------------------------
# Empirical losses
# ---------------------------------------------------------------------------

def empirical(loss, U, V, R, A, S):
    n = U.shape[0]
    s = (-0.5 * ((U @ R) * U).sum(1)[:, None] + U @ A @ V.T
         - 0.5 * ((V @ S) * V).sum(1)[None, :])
    pos = s.diagonal().mean()
    logn = math.log(n)
    if loss == '1way':
        return -pos + (torch.logsumexp(s, 1) - logn).mean()
    if loss == '2way':
        return -pos + 0.5 * ((torch.logsumexp(s, 1) - logn).mean()
                             + (torch.logsumexp(s, 0) - logn).mean())
    if loss == 'joint':
        return -pos + torch.logsumexp(s.flatten(), 0) - 2 * logn
    raise ValueError(loss)


def minimise(fn, params, iters=500):
    params = [p.clone().requires_grad_() for p in params]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=50,
                            tolerance_grad=1e-10, tolerance_change=1e-14,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        val = fn(params)
        val.backward()
        return val
    opt.step(closure)
    return [p.detach() for p in params]


def population_fit(g, fam, loss, restarts=4):
    best = None
    for r in range(restarts):
        try:
            p = minimise(lambda q: g.population(loss, *unpack(fam, q)), init_params(fam, g.d, r))
        except RuntimeError:
            continue
        with torch.no_grad():
            val = g.population(loss, *unpack(fam, p)).item()
        if math.isfinite(val) and (best is None or val < best[0]):
            best = (val, p)
    p = best[1]
    with torch.no_grad():
        bias = g.kl_v_given_u(*unpack(fam, p)).item()
    return p, bias


# ---------------------------------------------------------------------------
# One training set: fit every (family, loss) and score it
# ---------------------------------------------------------------------------

def run_one(task):
    torch.set_num_threads(1)
    d, rho, n, rep, starts = task
    g = Gaussian(d, rho)
    gen = torch.Generator().manual_seed(1_000_003 * rep + 7919 * n + int(1000 * rho) + 31 * d)
    U, V = g.sample(n, gen)
    out = {}
    for fam in FAMILIES:
        for loss in LOSSES:
            p = minimise(lambda q: empirical(loss, U, V, *unpack(fam, q)), starts[(fam, loss)])
            with torch.no_grad():
                err = g.kl_v_given_u(*unpack(fam, p)).item()
            out[f'{fam}/{loss}'] = err if math.isfinite(err) else None
    return d, rho, n, rep, out


def reps_for(n, base):
    return max(base // 4, int(base * min(1.0, 256 / n) ** 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--tag', default='finite_sample_gaussian')
    ap.add_argument('--settings', default='scalar,dim',
                    help='scalar: d=1 over rho; dim: rho=0.5 over d')
    args = ap.parse_args()

    if args.quick:
        settings = [(1, 0.5)]
        ns = [64, 256]
        base = 8
    else:
        settings = []
        if 'scalar' in args.settings:
            settings += [(1, r) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
        if 'dim' in args.settings:
            settings += [(d, 0.5) for d in [2, 4, 8]]
        ns = [32, 64, 128, 256, 512, 1024, 2048]
        base = args.reps

    # one file per (d, rho), so an interrupted sweep resumes where it stopped
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for d, rho in settings:
            path = out_dir / f'd{d}_rho{rho}.json'
            if path.exists():
                print(f'skip {path} (exists)', flush=True)
                continue
            g = Gaussian(d, rho)
            pop, starts = {}, {}
            for fam in FAMILIES:
                for loss in LOSSES:
                    p, bias = population_fit(g, fam, loss)
                    starts[(fam, loss)] = p
                    pop[f'{fam}/{loss}'] = bias
            tasks = [(d, rho, n, rep, starts) for n in ns if n > 2 * d
                     for rep in range(reps_for(n, base))]
            raw = []
            for k, (_, _, n, rep, out) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
                raw.append({'n': n, 'rep': rep, 'err': out})
                if (k + 1) % 200 == 0:
                    print(f'd={d} rho={rho}: {k + 1}/{len(tasks)}  {time.time() - t0:.0f}s', flush=True)
            path.write_text(json.dumps({'d': d, 'rho': rho, 'mutual_information': g.mi,
                                        'population_bias': pop, 'runs': raw}))
            print(f'wrote {path}  {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
