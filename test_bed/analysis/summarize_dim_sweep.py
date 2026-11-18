"""Summarize the embedding-dimension / correlation sweep (results/dim_sweep).

Run from test_bed:  python -m analysis.summarize_dim_sweep
Writes results/dim_sweep/summary.md and results/dim_sweep/tables.tex (LaTeX rows).
One seed per configuration.
"""
import glob
import json
from pathlib import Path

OUT = Path('results/dim_sweep')
POP = Path('results/dimension')
DEFAULT_EMBED = 100

MODELS = [  # (score-loss, encoder, label)
    ('tanh-twoway', 'linear', r'inner product, $\tanh$ future'),
    ('ncos-twoway', 'linear', 'normalized cosine'),
    ('l2-oneway', 'linear', r'$L^2$ one-way'),
    ('tk-oneway', 'linear', r'Student-$t$ kernel, one-way'),
    ('tk2-joint', 'linear', 'log-quadratic, joint'),
    ('tanh-twoway', 'mlp', r'MLP, inner product, $\tanh$ future'),
    ('tk-oneway', 'mlp', r'MLP, Student-$t$ kernel, one-way'),
]


def load():
    runs = {}
    for f in sorted(glob.glob(str(OUT / '*.json'))):
        r = json.load(open(f))
        a = r['args']
        data = f"ar{a['phi']}" if a['data'] == 'ar' else f"t{a['dial']:g}"
        key = (data, f"{a['score']}-{a['loss']}", a['encoder'], a['embed'] or DEFAULT_EMBED)
        runs[key] = r
    return runs


def stats(r):
    """info captured (nats), energy above oracle at the largest bank (%), over-forecast factors at p=1e-3, 1e-4."""
    last = max(r['rows'], key=lambda x: x['bank'])
    of = [w / p for w, p in zip(last['tail_weight'], r['tail_p'])]
    return {'info': r['info']['mean'], 'info_se': r['info']['se'], 'es': last['excess_pct'],
            'of3': of[1], 'of4': of[2]}


def available(data):
    if data.startswith('ar'):
        p = json.load(open(POP / f"population_ar_{data[2:]}.json"))
        return p['I_gauss'] + p['I_scale']
    return None


def fmt(x, nd=2):
    return '--' if x is None else f'{x:.{nd}f}'


def main():
    runs = load()
    md, tex = ['# Dimension sweep', '', 'Seed 0. info = nats captured on fresh pairs; ES = energy score above oracle (%) at the largest bank; OF = over-forecast factor.', ''], []
    for data in ('t3', 'ar0.6'):
        dims = sorted({k[3] for k in runs if k[0] == data})
        md += [f'## {data}: by embedding dimension', '', '| model | quantity | ' + ' | '.join(map(str, dims)) + ' |',
               '|---|---|' + '---|' * len(dims)]
        tex.append(f'%% {data}: dims ' + ' '.join(map(str, dims)))
        for sl, enc, label in MODELS:
            got = {d: stats(runs[(data, sl, enc, d)]) for d in dims if (data, sl, enc, d) in runs}
            if len(got) < 2:
                continue
            for q, nd in (('info', 2), ('es', 1), ('of4', 1)):
                md.append(f'| {label} | {q} | ' + ' | '.join(fmt(got[d][q], nd) if d in got else '' for d in dims) + ' |')
            tex.append(f'  {label} & ' + ' & '.join(
                (f"{got[d]['info']:.2f}" if d in got else '') for d in dims) + ' \\\\')
        md.append('')
    phis = sorted({k[0] for k in runs if k[0].startswith('ar')}, key=lambda x: float(x[2:]))
    md += ['## AR(1) Student-t: by lag-one correlation (default embedding 100)', '',
           '| model | ' + ' | '.join(phis) + ' |', '|---|' + '---|' * len(phis)]
    md.append('| available (I_corr + I_scale) | ' + ' | '.join(fmt(available(ph)) for ph in phis) + ' |')
    for sl, enc, label in MODELS:
        md.append(f'| {label} | ' + ' | '.join(
            fmt(stats(runs[(ph, sl, enc, DEFAULT_EMBED)])['info']) if (ph, sl, enc, DEFAULT_EMBED) in runs else ''
            for ph in phis) + ' |')
    (OUT / 'summary.md').write_text('\n'.join(md) + '\n')
    (OUT / 'tables.tex').write_text('\n'.join(tex) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
