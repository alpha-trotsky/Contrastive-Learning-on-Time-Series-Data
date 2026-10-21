"""Aggregate results/domain_sweep/*.json into one markdown table (mean over seeds).

Columns:
  energy vs oracle  : (ES_model - ES_oracle) / ES_oracle   -- 0% = perfect forecaster
  skill             : 1 - ES_model / ES_climatology          -- 0 = no better than ignoring the past
  avg err vs oracle : mean-prediction MSE / oracle's mean-prediction MSE
  band90            : how often the 90% band catches the truth (target 0.90)
  band90 easy/mid/hard : same, split by the oracle's own uncertainty (terciles)
  width90 / oracle  : band width relative to the perfect forecaster's

Usage (from test_bed/):  python -m analysis.summarize_domain_sweep > results/domain_sweep/summary.md
"""
import glob
import json
from collections import defaultdict
from statistics import mean

ORDER = {'t': ['inf', '10', '3'], 'warp': ['0', '0.5', '1'], 'dwell': ['0', '1', '1.5']}
NAMES = {'t': 'Heavy tails (Student-t), dial = df (inf = GP)',
         'warp': 'Bent GP (warp), dial = strength (0 = GP)',
         'dwell': 'Double-well, dial = lambda (0 = OU)'}


def main():
    runs = defaultdict(list)
    for f in glob.glob('results/domain_sweep/*.json'):
        r = json.load(open(f))
        a = r['args']
        runs[(a['domain'], f"{a['dial']:g}", a['model'])].append(r)

    for dom, dials in ORDER.items():
        print(f'\n### {NAMES[dom]}\n')
        print('| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |')
        print('|---|---|---|---|---|---|---|---|---|')
        for d in dials:
            for model in ('cosine', 'l2', 'mse'):
                rs = runs.get((dom, d, model), [])
                if not rs:
                    continue
                m = lambda f: mean(f(r) for r in rs)
                o = lambda r: r['scores']['oracle']
                s = lambda r: r['scores']['model']
                ex = m(lambda r: (s(r)['energy'] - o(r)['energy']) / o(r)['energy'])
                sk = m(lambda r: s(r)['energy_skill'])
                me = m(lambda r: s(r)['mse_mean'] / o(r)['mse_mean'])
                c = m(lambda r: s(r)['cov90'])
                byb = [m(lambda r, i=i: s(r)['cov90_by_spread'][i]) for i in range(3)]
                w = m(lambda r: s(r)['width90'] / o(r)['width90'])
                print(f'| {d} | {model} | {100 * ex:+.1f}% | {sk:.3f} | {me:.2f}x | {c:.3f} | '
                      f'{byb[0]:.2f} / {byb[1]:.2f} / {byb[2]:.2f} | {w:.2f} | {len(rs)} |')
        # point-forecast reference and extras
        extras = []
        for d in dials:
            for model in ('cosine', 'l2'):
                rs = runs.get((dom, d, model), [])
                if rs:
                    e = rs[0]['extra']
                    w = e.get('whitened_theory_err')
                    extras.append(f"{d}/{model}: ESS median {mean(r['extra']['ess_median'] for r in rs):.0f}"
                                  f", min {min(r['extra']['ess_min'] for r in rs):.1f}"
                                  + (f", whitened theory err {mean(r['extra']['whitened_theory_err'] for r in rs):.3f}" if w is not None else ''))
        print('\nSIR diagnostics: ' + '; '.join(extras))


if __name__ == '__main__':
    main()
