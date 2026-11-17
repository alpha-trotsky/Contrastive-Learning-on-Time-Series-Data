"""Fixed-score checks of the saturation theorems (notes/score_tails, Sec. 4.3).

No training.  Scalar futures v ~ Student-t(nu), one fixed past.

A. Tail of the tilted model (tail-index theorem).  For a score with
   s(v) + beta log|v| -> l(+-) along the two rays, the model nu^s = e^s mu / Z
   satisfies   nu^s(|v| > t) / (t^-beta mu(|v| > t))  ->  nu/(nu+beta) * mean_theta e^l / Z.
   Checked by quadrature for a saturating score (beta = 0), a Student-t kernel
   score (beta = b) and an L2 score (lighter than every power).

B. Exposure by the bank (bank theorem).  Truth p = r mu with a sharp Gaussian
   tilt r.  Model density ratio  rho = (1 - A pi) r + A 1{v > t0}:  exact in
   the bulk, plus a plateau of height A on the far right tail T = {v > t0},
   mu(T) = pi.  For each bank size N we simulate banks and report
     * the over-forecast factor  E[q_N(T)] / pi  with the bounds of the theorem
           (1 - (1-pi)^N) a / (a + N - 1) / pi   <=  .  <=  (1 - (1-pi)^N) / pi  <=  N
     * the expected excess energy score  1/2 E D_E(p, q_N)  with its lower bound.
   Prediction: the factor grows like N until N ~ A and then stays at A; the
   excess energy has a bump at N ~ A.  A = 0 is the exact score, and the
   unbounded score s = v is included for comparison (no plateau: it grows for ever).

Usage (from test_bed/):  python -m theory.saturation
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import integrate, stats

OUT = Path('results/theory')
NU = 3.0
C0 = 2 - math.sqrt(2)


# ---------------------------------------------------------------------------
# A. tail of the tilted model
# ---------------------------------------------------------------------------

def part_a():
    mu = stats.t(NU)
    a, b = 3.0, 4.0
    scores = {
        'saturating  s = 3 tanh(v)': (lambda v: a * np.tanh(v), 0.0, lambda Z: 0.5 * (math.exp(a) + math.exp(-a)) / Z),
        't-kernel    s = -2 log(1+v^2)': (lambda v: -0.5 * b * np.log1p(v * v), b, lambda Z: NU / (NU + b) / Z),
        'L2          s = -v^2/2': (lambda v: -0.5 * v * v, None, None),
    }
    rows = []
    for name, (s, beta, lim) in scores.items():
        f = lambda v: np.exp(s(v)) * mu.pdf(v)
        Z = integrate.quad(f, -np.inf, np.inf, limit=400)[0]
        row = {'score': name, 'Z': Z, 'beta': beta, 'predicted_limit': lim(Z) if lim else 0.0, 'ratios': {}}
        for t in (10.0, 100.0, 1000.0, 10000.0):
            # integrate in log-radius so the far tail is resolved
            g = lambda x: (f(np.exp(x)) + f(-np.exp(x))) * np.exp(x)
            edges = math.log(t) + np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64])
            mass = sum(integrate.quad(g, lo, hi, epsabs=0, epsrel=1e-11, limit=200)[0]
                       for lo, hi in zip(edges[:-1], edges[1:])) / Z
            row['ratios'][t] = mass / (2 * mu.sf(t) * t ** (-(beta or 0.0)))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# B. exposure by the bank
# ---------------------------------------------------------------------------

class Truth:
    """p = r mu with r(v) proportional to exp(-(v - m)^2 / (2 sd^2))."""

    def __init__(self, m=1.0, sd=0.5):
        self.mu = stats.t(NU)
        self.m, self.sd = m, sd
        self.c = integrate.quad(lambda v: self._r0(v) * self.mu.pdf(v), -np.inf, np.inf)[0]
        x = np.linspace(m - 12 * sd, m + 12 * sd, 40001)
        px = self.r(x) * self.mu.pdf(x)
        self.x, self.dx = x, x[1] - x[0]
        self.px = px / (px.sum() * self.dx)
        self.mean = (self.px * x).sum() * self.dx
        self.m_abs = (self.px * np.abs(x)).sum() * self.dx
        # h(v) = E_p |X - v| on a grid; outside the grid it is |v - mean| exactly
        self.hv = np.linspace(x[0], x[-1], 8001)
        self.h = np.array([(self.px * np.abs(x - v)).sum() * self.dx for v in self.hv])
        cdf = np.cumsum(self.px) * self.dx - 0.5 * self.px * self.dx       # midpoint rule
        self.gini = 2 * (self.px * x * (2 * cdf - 1)).sum() * self.dx      # E_p|X - X'|

    def _r0(self, v):
        return np.exp(-0.5 * ((v - self.m) / self.sd) ** 2)

    def r(self, v):
        return self._r0(v) / getattr(self, 'c', 1.0)

    def dist_to(self, v):
        out = np.abs(v - self.mean)
        inside = (v > self.hv[0]) & (v < self.hv[-1])
        out[inside] = np.interp(v[inside], self.hv, self.h)
        return out


def half_energy_distance(p, v, w):
    """1/2 D_E(p, sum_j w_j delta_{v_j}) for each row of v, w (banks, N)."""
    order = np.argsort(v, axis=1)
    v = np.take_along_axis(v, order, 1)
    w = np.take_along_axis(w, order, 1)
    C = np.cumsum(w, axis=1)
    self_term = 2 * (w * v * (2 * C - w - 1)).sum(1)                      # sum_ij w_i w_j |v_i - v_j|
    cross = (w * p.dist_to(v)).sum(1)
    return cross - 0.5 * self_term - 0.5 * p.gini


def part_b(seed=0):
    rng = np.random.default_rng(seed)
    p = Truth()
    pi = 1e-4
    t0 = p.mu.isf(pi)
    models = {'exact (A = 0)': 0.0, 'plateau A = 30': 30.0, 'plateau A = 1000': 1000.0, 'unbounded s = v': None}
    sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    res = {'pi': pi, 't0': t0, 'm_p': p.m_abs, 'sizes': sizes, 'models': {}}
    for name, A in models.items():
        rows = []
        for N in sizes:
            n_banks = 400000 if N <= 100 else (100000 if N <= 1000 else (8000 if N <= 10000 else (800 if N <= 100000 else 200)))
            chunk = max(1, min(n_banks, int(2e7 // N)))
            tw, ex, wmax = [], [], []
            for s in range(0, n_banks, chunk):
                v = rng.standard_t(NU, size=(min(chunk, n_banks - s), N))
                if A is None:
                    logw = v - v.max(1, keepdims=True)
                    w = np.exp(logw)
                else:
                    w = (1 - A * pi) * p.r(v) + A * (v > t0)
                w = w / w.sum(1, keepdims=True)
                tw.append((w * (v > t0)).sum(1))
                wmax.append(w.max(1))
                ex.append(half_energy_distance(p, v, w))
            tw, ex, wmax = map(np.concatenate, (tw, ex, wmax))
            pk = 1 - (1 - pi) ** N
            row = {'N': N, 'banks': n_banks, 'of': tw.mean() / pi, 'of_se': tw.std() / math.sqrt(n_banks) / pi,
                   'of_upper': pk / pi, 'excess': ex.mean(), 'excess_se': ex.std() / math.sqrt(n_banks),
                   'excess_median': float(np.median(ex)), 'wmax': wmax.mean()}
            if A:
                cond = A / (A + N - 1)
                row['of_lower'] = pk * cond / pi
                row['excess_lower'] = pk * max(C0 * t0 * cond - 2 * p.m_abs, 0.0) ** 2 / (4 * t0)
            rows.append(row)
        res['models'][name] = rows
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = part_a()
    B = part_b()
    json.dump({'A': A, 'B': B}, open(OUT / 'saturation.json', 'w'), indent=1)
    L = ['# Fixed-score checks of the saturation theorems', '',
         'Generated by `python -m theory.saturation`.  Scalar futures, Student-t with 3 degrees of freedom.', '',
         '## A. Tail of the tilted model', '',
         'Ratio  nu^s(|v|>t) / (t^-beta mu(|v|>t))  by quadrature; the theorem predicts the last column.', '',
         '| score | beta | t=10 | t=100 | t=1000 | t=10000 | predicted limit |', '|---|---|---|---|---|---|---|']
    for r in A:
        L.append(f"| {r['score']} | {r['beta'] if r['beta'] is not None else 'any'} | "
                 + ' | '.join(f'{x:.4g}' for x in r['ratios'].values()) + f" | {r['predicted_limit']:.4g} |")
    L += ['', '## B. Exposure by the bank', '',
          f"Tail set T = {{v > {B['t0']:.1f}}}, marginal mass pi = {B['pi']:g}.  "
          'Over-forecast factor = E[q_N(T)] / pi; excess = expected energy score above the truth.', '']
    for name, rows in B['models'].items():
        L += [f'### {name}', '', '| N | over-forecast factor | theorem: lower | theorem: upper | excess energy (mean) | theorem: lower | median | max weight |',
              '|---|---|---|---|---|---|---|---|']
        for r in rows:
            L.append(f"| {r['N']:,} | {r['of']:.3g} ± {r['of_se']:.2g} | {r.get('of_lower', float('nan')):.3g} | {r['of_upper']:.3g} | "
                     f"{r['excess']:.4g} ± {r['excess_se']:.2g} | {r.get('excess_lower', float('nan')):.3g} | {r['excess_median']:.4g} | {r['wmax']:.3g} |")
        L.append('')
    (OUT / 'saturation.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
