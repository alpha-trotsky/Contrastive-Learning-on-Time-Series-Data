"""Embedding dimension and correlation: exact population-level computations.

Data: elliptical Student-t past/future windows (nu degrees of freedom) built on
a Gaussian base with covariance blocks C_uu, C_uv, C_vv.

For each base and each embedding dimension d this script evaluates the best
model whose score sees the future only through a rank-d linear map (the top d
canonical directions).  That model is known in closed form: it forecasts the d
embedded coordinates exactly and leaves the remaining coordinates at their
climatological law given the embedded ones.  Reported:

  * the information lost (the floor of the one-way loss), split into the part
    that depends on the correlations and the part that comes from the shared
    scale:   sum_{i>d} -1/2 log(1 - rho_i^2)  +  c(nu,du,dv) - c(nu,du,d);
  * by simulation: the probability the model gives to {|v| > t}, where t is the
    1 - p quantile of the true conditional (1 = calibrated), and the energy
    distance to the true conditional in % of the oracle's energy score.

Also checks the closed form for the mutual information of a multivariate t
against Monte Carlo.

Usage (from test_bed/):  python -m theory.dimension [--only NAME]
"""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch import digamma, lgamma

OUT = Path('results/dimension')
DS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 50]
PS = [1e-1, 1e-2, 1e-3]


def T(x):
    return torch.tensor(float(x), dtype=torch.float64)


def c_scale(nu, du, dv):
    """Mutual information of an uncorrelated (du + dv)-dimensional Student-t."""
    if du == 0 or dv == 0 or math.isinf(nu):
        return 0.0
    D = du + dv
    g = lambda m: lgamma(T((nu + m) / 2))
    e = lambda m: (nu + m) / 2 * (digamma(T((nu + m) / 2)) - digamma(T(nu / 2)))
    return (g(D) + g(0) - g(du) - g(dv) - e(D) + e(du) + e(dv)).item()


def isqrt(C):
    e, V = torch.linalg.eigh(C)
    return V @ torch.diag(e.rsqrt()) @ V.T, V @ torch.diag(e.sqrt()) @ V.T


def canonical(Cuu, Cuv, Cvv):
    Wu, _ = isqrt(Cuu)
    Wv, Sv = isqrt(Cvv)
    U, rho, Vh = torch.linalg.svd(Wu @ Cuv @ Wv)
    return rho.clamp(0, 1 - 1e-12), U.T @ Wu, Sv @ Vh.T          # rho, u -> a, b -> v


def mi_check(nu=3.0, du=3, dv=2, n=2000000, seed=0):
    """Monte Carlo mutual information of a correlated t vs I_Gauss + c_scale."""
    g = torch.Generator().manual_seed(seed)
    D = du + dv
    A = torch.randn(D, D, generator=g, dtype=torch.float64)
    S = A @ A.T + 0.5 * torch.eye(D, dtype=torch.float64)
    z = torch.randn(n, D, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(S).T
    w = torch.distributions.Gamma(nu / 2, nu / 2).sample((n, 1)).double()
    z = z / w.sqrt()

    def logp(x, Sig):
        m = x.shape[1]
        q = (x * torch.linalg.solve(Sig, x.T).T).sum(1)
        return (lgamma(T((nu + m) / 2)) - lgamma(T(nu / 2)) - m / 2 * math.log(nu * math.pi)
                - 0.5 * torch.logdet(Sig) - (nu + m) / 2 * torch.log1p(q / nu))
    lr = logp(z, S) - logp(z[:, :du], S[:du, :du]) - logp(z[:, du:], S[du:, du:])
    rho, _, _ = canonical(S[:du, :du], S[:du, du:], S[du:, du:])
    exact = (-0.5 * torch.log1p(-rho ** 2)).sum().item() + c_scale(nu, du, dv)
    return {'mc': lr.mean().item(), 'mc_se': (lr.std() / math.sqrt(n)).item(), 'closed_form': exact}


def bases():
    from experiment import build_signal, build_modality
    from modalities.ar_process import ARPastFuture
    cfg = yaml.safe_load(open('config.yaml'))
    out = {'field': build_modality(cfg['modality'], build_signal(cfg['signal']))}
    for phi in (0.0, 0.3, 0.6, 0.9, 0.99):
        out[f'ar_{phi:g}'] = ARPastFuture([phi])
    out['armix4'] = ARPastFuture([0.99, 0.9, 0.6, 0.3])
    return out


def run(name, base, nu=3.0, n_past=100, n_tail=200000, n_es=1000, seed=0):
    torch.manual_seed(seed)
    c2 = (nu - 2) / nu
    Cuu, Cuv, Cvv = c2 * base.C_uu(), c2 * base.C_uv(), c2 * base.C_vv()
    du, dv = Cuu.shape[0], Cvv.shape[0]
    rho, to_a, to_v = canonical(Cuu, Cuv, Cvv)
    k = min(du, dv)
    rho_v = torch.zeros(dv, dtype=torch.float64)
    rho_v[:k] = rho[:k]
    ig = -0.5 * torch.log1p(-rho_v ** 2)
    lag1 = (Cvv[0, 1] / Cvv[0, 0]).item()
    res = {'name': name, 'nu': nu, 'du': du, 'dv': dv, 'lag1': lag1, 'rho': rho_v.tolist(),
           'I_gauss': ig.sum().item(), 'I_scale': c_scale(nu, du, dv), 'ds': [d for d in DS if d <= dv], 'rows': []}

    # test pasts
    L = torch.linalg.cholesky(torch.cat([torch.cat([Cuu, Cuv], 1), torch.cat([Cuv.T, Cvv], 1)]))
    z = torch.randn(n_past, du + dv, dtype=torch.float64) @ L.T
    z = z / torch.distributions.Gamma(nu / 2, nu / 2).sample((n_past, 1)).double().sqrt()
    a = z[:, :du] @ to_a.T                                        # (n_past, du) canonical coordinates
    sa = ((nu + (a * a).sum(1)) / (nu + du)).sqrt()
    mean_b = torch.zeros(n_past, dv, dtype=torch.float64)
    mean_b[:, :k] = rho[:k] * a[:, :k]
    sd_b = (1 - rho_v ** 2).sqrt()

    def truth(i, n):
        w = torch.distributions.Gamma((nu + du) / 2, (nu + du) / 2).sample((n, 1)).double()
        return mean_b[i] + sa[i] * sd_b * torch.randn(n, dv, dtype=torch.float64) / w.sqrt()

    def model(i, n, d):
        b = truth(i, n)
        if d < dv:
            w = torch.distributions.Gamma((nu + d) / 2, (nu + d) / 2).sample((n, 1)).double()
            s = ((nu + (b[:, :d] ** 2).sum(1, keepdim=True)) / (nu + d)).sqrt()
            b[:, d:] = s * torch.randn(n, dv - d, dtype=torch.float64) / w.sqrt()
        return b

    acc = {d: {'tail': torch.zeros(n_past, len(PS)), 'ed': torch.zeros(n_past)} for d in res['ds']}
    es0 = torch.zeros(n_past)
    for i in range(n_past):
        nt = (truth(i, n_tail) @ to_v.T).norm(dim=1)
        thr = torch.quantile(nt[:100000], 1 - torch.tensor(PS[:1], dtype=torch.float64)).tolist()
        thr += [torch.kthvalue(nt, int(round((1 - p) * n_tail))).values.item() for p in PS[1:]]
        x = truth(i, n_es) @ to_v.T
        x2 = truth(i, n_es) @ to_v.T
        exx = torch.cdist(x, x2).mean()
        es0[i] = 0.5 * exx
        for d in res['ds']:
            nm = (model(i, n_tail, d) @ to_v.T).norm(dim=1)
            acc[d]['tail'][i] = torch.tensor([(nm > t).double().mean().item() for t in thr])
            y, y2 = model(i, n_es, d) @ to_v.T, model(i, n_es, d) @ to_v.T
            acc[d]['ed'][i] = 2 * torch.cdist(x, y).mean() - exx - torch.cdist(y, y2).mean()
    for d in res['ds']:
        lost_g = ig[d:].sum().item()
        lost_s = c_scale(nu, du, dv) - c_scale(nu, du, d)
        of = acc[d]['tail'] / torch.tensor(PS)
        res['rows'].append({'d': d, 'lost_gauss': lost_g, 'lost_scale': lost_s,
                            'of_mean': of.mean(0).tolist(), 'of_median': of.median(0).values.tolist(),
                            'energy_excess_pct': (100 * acc[d]['ed'].mean() / (2 * es0.mean())).item()})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f'population_{name}.json', 'w'))
    print(f"== {name}: lag-1 correlation {lag1:.3f}; rho = {[round(x, 3) for x in rho_v[:5].tolist()]}; "
          f"I = {res['I_gauss']:.3f} (correlation) + {res['I_scale']:.3f} (scale) nats")
    print('   d | lost: corr  scale | prob. of the true 1e-1/1e-2/1e-3 conditional tail event, / p (median past) | energy excess %')
    for r in res['rows']:
        print(f"  {r['d']:2d} | {r['lost_gauss']:.3f}  {r['lost_scale']:.3f} | "
              + '  '.join(f'{x:8.2f}' for x in r['of_median']) + f" | {r['energy_excess_pct']:+.2f}")
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--n_past', type=int, default=100)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.only in (None, 'check'):
        chk = [mi_check(3.0, 3, 2), mi_check(5.0, 4, 4, seed=1)]
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(chk, open(OUT / 'mi_check.json', 'w'))
        print('mutual information of a multivariate t, Monte Carlo vs closed form:', chk)
    for name, base in bases().items():
        if a.only in (None, name):
            run(name, base, n_past=a.n_past)
