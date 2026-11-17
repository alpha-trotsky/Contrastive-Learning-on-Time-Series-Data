"""Numerical sanity checks of the auxiliary results of notes/score_tails, Sec. 4.3.

Each check prints PASS/FAIL.  They are not proofs; they guard against algebra
slips in the constants.

  1. Lemma (energy distance charges mass times distance):
         D_E(p, q) >= (c0 t q(H_{t,e}) - 2 E_p|X|)_+^2 / (2t)
     on random discrete p, q in R^d.
  2. Proposition (log-quadratic form of log r on an elliptical t), and the
     lemma that log r(u, t theta) + d_u log t converges.
  3. Proposition (KL-optimal bounded score = clipped log ratio): on a discrete
     space, a numerical minimiser of KL(p || rho mu) under m <= rho <= M
     coincides with clip(r / lambda, m, M).
  4. Theorem (exposure), part (iii):  E q_N(T) <= (4b + 8 chi^2) pi.

Usage (from test_bed/):  python -m theory.check_lemmas
"""
import math

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

C0 = 2 - math.sqrt(2)
rng = np.random.default_rng(0)


def energy_distance(x, wx, y, wy):
    d = lambda a, b: np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 2 * wx @ d(x, y) @ wy - wx @ d(x, x) @ wx - wy @ d(y, y) @ wy


def check_energy_lemma(trials=2000):
    worst = np.inf
    for _ in range(trials):
        d = rng.integers(1, 6)
        n, m = rng.integers(2, 12, size=2)
        x = rng.standard_normal((n, d)) * rng.uniform(0.1, 2)
        y = rng.standard_normal((m, d)) * rng.uniform(0.1, 2)
        # push a few q-atoms far out
        far = rng.random(m) < 0.4
        y[far] *= rng.uniform(5, 200)
        wx, wy = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m))
        e = rng.standard_normal(d)
        e /= np.linalg.norm(e)
        t = rng.uniform(0.5, 50)
        qH = wy[(np.linalg.norm(y, axis=1) >= t) & (y @ e >= 0)].sum()
        bound = max(C0 * t * qH - 2 * wx @ np.linalg.norm(x, axis=1), 0) ** 2 / (2 * t)
        worst = min(worst, energy_distance(x, wx, y, wy) - bound)
    print(f"1. energy lemma: min(D_E - bound) over {trials} trials = {worst:.3e}  ->  {'PASS' if worst > -1e-9 else 'FAIL'}")


def mvt_logpdf(x, S, nu):
    d = len(x)
    q = x @ np.linalg.solve(S, x)
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * math.log(nu * math.pi)
            - 0.5 * np.linalg.slogdet(S)[1] - 0.5 * (nu + d) * math.log1p(q / nu))


def check_logquad(nu=3.0, du=4, dv=3):
    d = du + dv
    Araw = rng.standard_normal((d, d))
    S = Araw @ Araw.T + 0.5 * np.eye(d)
    Suu, Svv = S[:du, :du], S[du:, du:]
    w, V = np.linalg.eigh(np.linalg.inv(S) / nu)
    R = (V * np.sqrt(w)) @ V.T                       # symmetric square root of Sigma^-1 / nu
    G, H = R[:, :du], -R[:, du:]
    sq = lambda M: (lambda w, V: (V / np.sqrt(w)) @ V.T)(*np.linalg.eigh(M))   # M^(-1/2)
    G0, H0 = sq(nu * Suu), sq(nu * Svv)
    diffs = []
    for _ in range(200):
        u, v = rng.standard_normal(du) * 3, rng.standard_normal(dv) * 3
        z = np.concatenate([u, v])
        logr = mvt_logpdf(z, S, nu) - mvt_logpdf(u, Suu, nu) - mvt_logpdf(v, Svv, nu)
        form = (-0.5 * (nu + d) * math.log1p(np.sum((G @ u - H @ v) ** 2))
                + 0.5 * (nu + du) * math.log1p(np.sum((G0 @ u) ** 2))
                + 0.5 * (nu + dv) * math.log1p(np.sum((H0 @ v) ** 2)))
        diffs.append(logr - form)
    spread = np.ptp(diffs)
    print(f"2a. log-quadratic form of log r: spread of (log r - formula) = {spread:.2e}  ->  {'PASS' if spread < 1e-8 else 'FAIL'}")
    u = rng.standard_normal(du)
    th = rng.standard_normal(dv)
    th /= np.linalg.norm(th)
    vals = []
    for t in (1e3, 1e4, 1e5, 1e6):
        z = np.concatenate([u, t * th])
        vals.append(mvt_logpdf(z, S, nu) - mvt_logpdf(u, Suu, nu) - mvt_logpdf(t * th, Svv, nu) + du * math.log(t))
    steps = np.abs(np.diff(vals))                     # O(1/t) convergence: each decade shrinks the step ~10x
    conv = steps[-1] < 1e-3 and np.all(steps[1:] < 0.2 * steps[:-1])
    print(f"2b. log r(u, t theta) + d_u log t at t = 1e3..1e6: {np.round(vals, 6)}  ->  {'PASS' if conv else 'FAIL'}")


def check_clip(n=40, trials=20):
    ok = True
    for _ in range(trials):
        mu = rng.dirichlet(np.ones(n))
        p = rng.dirichlet(np.ones(n) * 0.3)
        r = p / mu
        m, M = rng.uniform(0.05, 0.9), rng.uniform(1.2, 30)
        clip = lambda lam: np.clip(r / lam, m, M)
        lam = optimize.brentq(lambda l: clip(l) @ mu - 1, 1e-12, 1e12)
        rho_star = clip(lam)
        kl = lambda rho: p @ np.log(r / rho)
        res = optimize.minimize(lambda x: kl(x), np.ones(n), jac=lambda x: -p / x, bounds=[(m, M)] * n,
                                constraints=[{'type': 'eq', 'fun': lambda x: x @ mu - 1, 'jac': lambda x: mu}],
                                method='SLSQP', options={'maxiter': 500, 'ftol': 1e-14})
        ok &= kl(rho_star) <= res.fun + 1e-8
        # and random feasible competitors
        for _ in range(50):
            x = np.clip(rho_star * np.exp(0.3 * rng.standard_normal(n)), m, M)
            for _ in range(200):                      # project onto {x @ mu = 1} within the box
                x = np.clip(x + (1 - x @ mu), m, M)
            if abs(x @ mu - 1) < 1e-10:
                ok &= kl(rho_star) <= kl(x) + 1e-10
    print(f"3. clipped log ratio is the KL-optimal bounded tilt ({trials} random problems)  ->  {'PASS' if ok else 'FAIL'}")


def check_saturation_bound(banks=20000):
    mu = stats.t(3.0)
    pi = 1e-3
    t0 = mu.isf(pi)
    worst = np.inf
    for A in (5.0, 50.0, 500.0):
        r0 = lambda v: np.exp(-0.5 * ((v - 1) / 0.5) ** 2)
        c = np.mean(r0(mu.rvs(size=2000000, random_state=rng)))
        rho = lambda v: (1 - A * pi) * r0(v) / c + A * (v > t0)
        v = mu.rvs(size=4000000, random_state=rng)
        chi2 = np.mean(rho(v) ** 2) - 1
        b = A + (1 - A * pi) * r0(t0) / c
        for N in (2, 10, 100, 1000, 5000):
            v = mu.rvs(size=(banks if N <= 1000 else 2000, N), random_state=rng)
            w = rho(v)
            q = ((w * (v > t0)).sum(1) / w.sum(1)).mean()
            worst = min(worst, (4 * b + 8 * chi2) * pi - q)
    print(f"4. exposure (iii): min over (A, N) of (4b + 8 chi^2) pi - E q_N(T) = {worst:.3e}  ->  {'PASS' if worst > 0 else 'FAIL'}")


if __name__ == '__main__':
    check_energy_lemma()
    check_logquad()
    check_clip()
    check_saturation_bound()
