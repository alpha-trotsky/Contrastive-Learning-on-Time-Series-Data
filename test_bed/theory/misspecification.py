"""Population-level checks for docs/theory_bounded_misspecified.tex.

Two experiments, both exact or nearly so (no training, no sampling noise in part 1):

1. Misspecification on Gaussian data.  For linear encoders every score is a
   quadratic  s(u, v) = -1/2 u'Ru + u'Av - 1/2 v'Sv,  so the model joint is
   Gaussian with precision  D^{-1} + Q,  D = blockdiag(C_uu, C_vv),
   Q = [[R, -A], [-A', S]].  We minimise each population loss in closed form
   (all KLs between Gaussians) over each score family and report how much of
   the available information I(u; v) the resulting v|u forecast loses:

       loss fraction = E_u KL(mu_{v|u} || nu_{v|u}) / I(u; v)

   0 = perfect forecaster, 1 = no better than the marginal ("climatology").

   Families:  dot   R = S = 0                      (unnormalised cosine)
              L2    Q = [G, -H]'[G, -H]  (PSD)      (-1/2 |Gu - Hv|^2)
              quad  R, S, A free                    (dot + u-only + v-only terms)
   Losses:    1-way (v|u), 2-way (lambda_u = lambda_v), joint.

2. Collapse of the bank forecast.  Self-normalised weights exp(s(v_j)) over N
   bank futures, for a linear score on Gaussian vs Student-t futures and an
   L2 score on Student-t futures.  Reports the largest weight as N grows.

Usage (from test_bed/):  python -m theory.misspecification
"""
import math
from pathlib import Path

import torch
import yaml

torch.set_default_dtype(torch.float64)

OUT = Path('results/theory')


# ---------------------------------------------------------------------------
# Gaussian KL helpers
# ---------------------------------------------------------------------------

def logdet(M):
    """log det of a positive-definite matrix; +inf-safe: nan if M is not PD,
    which LBFGS's line search treats as a failed step."""
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
    def __init__(self, C_uu, C_uv, C_vv):
        self.C_uu, self.C_uv, self.C_vv = C_uu, C_uv, C_vv
        self.nu, self.nv = C_uu.shape[0], C_vv.shape[0]
        self.K_vu = torch.linalg.solve(C_uu, C_uv).T          # E[v|u] = K_vu u
        self.K_uv = torch.linalg.solve(C_vv, C_uv.T).T        # E[u|v] = K_uv v
        self.C_vgu = C_vv - C_uv.T @ torch.linalg.solve(C_uu, C_uv)
        self.C_ugv = C_uu - C_uv @ torch.linalg.solve(C_vv, C_uv.T)
        self.C = torch.cat([torch.cat([C_uu, C_uv], 1), torch.cat([C_uv.T, C_vv], 1)], 0)
        self.D_inv = torch.block_diag(torch.linalg.inv(C_uu), torch.linalg.inv(C_vv))
        # I(u; v) = E_u KL(mu_{v|u} || mu_v)
        self.mi = 0.5 * (logdet(C_vv) - logdet(self.C_vgu))

    # model conditionals for s = -1/2 u'Ru + u'Av - 1/2 v'Sv
    def kl_v_given_u(self, R, A, S):
        P = torch.linalg.inv(self.C_vv) + S
        return cond_kl(P, torch.linalg.solve(P, A.T), self.K_vu, self.C_vgu, self.C_uu)

    def kl_u_given_v(self, R, A, S):
        P = torch.linalg.inv(self.C_uu) + R
        return cond_kl(P, torch.linalg.solve(P, A), self.K_uv, self.C_ugv, self.C_vv)

    def kl_joint(self, R, A, S):
        Q = torch.cat([torch.cat([R, -A], 1), torch.cat([-A.T, S], 1)], 0)
        L = self.D_inv + Q
        n = L.shape[0]
        return 0.5 * (torch.trace(L @ self.C) - n - logdet(L) - logdet(self.C))


# ---------------------------------------------------------------------------
# Score families, parametrised so that every normaliser is finite
# ---------------------------------------------------------------------------

def family(name, g, k):
    nu, nv = g.nu, g.nv
    if name == 'dot':
        # start at the Thm 5.1 minimiser; A is unconstrained for conditionals,
        # the joint needs ||C_uu^{1/2} A C_vv^{1/2}|| < 1 (checked by logdet -> nan)
        A0 = torch.linalg.solve(g.C_uu, g.C_uv) @ torch.linalg.inv(g.C_vv)
        params = [(0.5 * A0).clone().requires_grad_()]

        def unpack(p):
            return torch.zeros(nu, nu), p[0], torch.zeros(nv, nv)
    elif name == 'l2':
        params = [(0.1 * torch.randn(k, nu)).requires_grad_(),
                  (0.1 * torch.randn(k, nv)).requires_grad_()]

        def unpack(p):
            G, H = p
            return G.T @ G, G.T @ H, H.T @ H
    elif name == 'quad':
        params = [torch.zeros(nu, nu, requires_grad=True),
                  torch.zeros(nu, nv, requires_grad=True),
                  torch.zeros(nv, nv, requires_grad=True)]

        def unpack(p):
            Ru, A, Sv = p
            return 0.5 * (Ru + Ru.T), A, 0.5 * (Sv + Sv.T)
    else:
        raise ValueError(name)
    return params, unpack


def objective(g, loss, R, A, S):
    if loss == '1way':
        return g.kl_v_given_u(R, A, S)
    if loss == '2way':
        return 0.5 * (g.kl_v_given_u(R, A, S) + g.kl_u_given_v(R, A, S))
    if loss == 'joint':
        return g.kl_joint(R, A, S)
    raise ValueError(loss)


def fit(g, fam, loss, k=None, restarts=3, iters=400):
    best = None
    for r in range(restarts):
        torch.manual_seed(r)
        params, unpack = family(fam, g, k or (g.nu + g.nv))
        opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            val = objective(g, loss, *unpack(params))
            val.backward()
            return val
        opt.step(closure)
        with torch.no_grad():
            val = objective(g, loss, *unpack(params)).item()
            fc = g.kl_v_given_u(*unpack(params)).item()
        if math.isfinite(val) and (best is None or val < best[0]):
            best = (val, fc)
    return best[1] / g.mi.item()


def scalar(rho):
    one = torch.ones(1, 1)
    return Gaussian(one, rho * one, one)


def goal1_gp():
    """Covariances of the goal-1 Gaussian domain (config.yaml), in closed form."""
    from signals.gaussian_process import GaussianProcess1D
    from modalities.past_future import PastFutureModality
    cfg = yaml.safe_load(open('config.yaml'))
    sig, m = cfg['signal'], cfg['modality']
    gen = GaussianProcess1D(sig['alpha'], sig['tau'], sig['dim_true'], sig['zero_mean'])
    past = list(range(m['start'], m['start'] + m['past_len']))
    future = list(range(past[-1] + 1, past[-1] + 1 + m['future_len']))
    mod = PastFutureModality(gen, past, future, m['sigma_u'], m['sigma_v'])
    return Gaussian(mod.C_uu(), mod.C_uv(), mod.C_vv())


GRID = [('dot', '1way'), ('dot', '2way'), ('dot', 'joint'),
        ('l2', '1way'), ('l2', '2way'), ('l2', 'joint'),
        ('quad', '1way'), ('quad', '2way'), ('quad', 'joint')]


def part1():
    rows = []
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        g = scalar(rho)
        rows.append((f'scalar, rho = {rho}', g.mi.item(), [fit(g, f, l) for f, l in GRID]))
        print(rows[-1])
    g = goal1_gp()
    rows.append(('goal-1 GP (50 -> 50)', g.mi.item(),
                 [fit(g, f, l, restarts=1, iters=3000) for f, l in GRID]))
    print(rows[-1])
    return rows


# ---------------------------------------------------------------------------
# Part 2: collapse of the bank forecast
# ---------------------------------------------------------------------------

def student_t(n, df, gen):
    z = torch.randn(n, generator=gen)
    w = torch.distributions.Gamma(df / 2, df / 2).sample((n,))
    return z / torch.sqrt(w) * math.sqrt((df - 2) / df)   # unit variance


def part2(reps=200, a=1.0, df=3.0):
    """Mean of the largest self-normalised weight, and ESS / N."""
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    cases = {
        'linear score, Gaussian futures': lambda v: a * v,
        'linear score, Student-t futures': lambda v: a * v,
        'L2 score, Student-t futures': lambda v: -0.5 * (v - a) ** 2,
    }
    rows = []
    for N in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
        row = []
        for name, s in cases.items():
            wmax, ess = 0.0, 0.0
            for _ in range(reps):
                v = torch.randn(N, generator=gen) if 'Gaussian' in name else student_t(N, df, gen)
                w = torch.softmax(s(v), 0)
                wmax += w.max().item() / reps
                ess += (1.0 / (w ** 2).sum()).item() / N / reps
            row.append((wmax, ess))
        rows.append((N, row))
        print(N, row)
    return list(cases), rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = part1()
    names, coll = part2()
    lines = ['# Population-level checks for the theory note', '',
             'Generated by `python -m theory.misspecification`.', '',
             '## 1. Information lost by the v|u forecast at the population optimum', '',
             'Entries: E_u KL(mu_{v|u} || nu_{v|u}) / I(u; v). 0 = perfect, 1 = climatology.', '',
             '| setting | I(u;v) nats | ' + ' | '.join(f'{f} {l}' for f, l in GRID) + ' |',
             '|' + '---|' * (2 + len(GRID))]
    for name, mi, vals in rows:
        lines.append(f'| {name} | {mi:.3f} | ' + ' | '.join(f'{max(x, 0.0):.3f}' for x in vals) + ' |')
    lines += ['', '## 2. Largest bank weight (mean over 200 banks) and ESS / N', '',
              '| N | ' + ' | '.join(names) + ' |', '|' + '---|' * (1 + len(names))]
    for N, row in coll:
        lines.append(f'| {N:.0e} | ' + ' | '.join(f'{w:.3f} (ESS/N {e:.4f})' for w, e in row) + ' |')
    (OUT / 'misspecification.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
