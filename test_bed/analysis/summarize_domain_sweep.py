"""Aggregate results/domain_sweep/*.json into markdown tables (mean over seeds).

Columns:
  energy vs oracle  : (ES_model - ES_oracle) / ES_oracle   -- 0% = perfect forecaster
  skill             : 1 - ES_model / ES_climatology          -- 0 = no better than ignoring the past
  avg err vs oracle : mean-prediction MSE / oracle's mean-prediction MSE
  band90            : how often the 90% band catches the truth (target 0.90)
  band90 easy/mid/hard : same, split by the oracle's own uncertainty (terciles)
  width90 / oracle  : band width relative to the perfect forecaster's

Usage (from test_bed/):
    python -m analysis.summarize_domain_sweep                 > results/domain_sweep/summary.md
    python -m analysis.summarize_domain_sweep --encoder mlp   > results/domain_sweep/summary_mlp.md
"""
import argparse
import glob
import json
from collections import defaultdict
from statistics import mean

ORDER = {'t': ['inf', '10', '3'], 'warp': ['0', '0.5', '1'], 'dwell': ['0', '1', '1.5']}
# run key -> display name; 'cosine' / 'l2' are the first sweep's two-sided / one-sided runs
MODELS = ['cosine_1s', 'cosine', 'cosine_joint', 'ncos', 'l2', 'l2_2s', 'l2_joint', 'mse', 'gauss']
CONTRASTIVE = MODELS[:7]
LABEL = {'cosine_1s': 'cosine one-sided', 'cosine': 'cosine two-sided', 'cosine_joint': 'cosine joint',
         'ncos': 'normalised cosine', 'l2': 'L2 one-sided', 'l2_2s': 'L2 two-sided', 'l2_joint': 'L2 joint',
         'mse': 'MSE + noise', 'gauss': 'Gaussian NLL'}
SHORT = {'cosine_1s': 'cos 1s', 'cosine': 'cos 2s', 'cosine_joint': 'cos joint', 'ncos': 'norm cos',
         'l2': 'L2 1s', 'l2_2s': 'L2 2s', 'l2_joint': 'L2 joint', 'mse': 'MSE', 'gauss': 'Gauss NLL'}
NAMES = {'t': 'Heavy tails (Student-t), dial = df (inf = GP)',
         'warp': 'Bent GP (warp), dial = strength (0 = GP)',
         'dwell': 'Double-well, dial = lambda (0 = OU)'}

o = lambda r: r['scores']['oracle']
s_ = lambda r: r['scores']['model']
METRICS = {
    'energy': (lambda r: (s_(r)['energy'] - o(r)['energy']) / o(r)['energy'], lambda x: f'{100 * x:+.1f}%'),
    'hard': (lambda r: s_(r)['cov90_by_spread'][2], lambda x: f'{x:.2f}'),
    'cov90': (lambda r: s_(r)['cov90'], lambda x: f'{x:.2f}'),
    'err': (lambda r: s_(r)['mse_mean'] / o(r)['mse_mean'], lambda x: f'{x:.2f}x'),
}


def load():
    runs = defaultdict(list)
    for f in glob.glob('results/domain_sweep/*.json'):
        r = json.load(open(f))
        a = r['args']
        runs[(a['domain'], f"{a['dial']:g}", a['model'], a.get('encoder', 'linear'))].append(r)
    return runs


def grid(runs, enc, title, metric):
    f, fmt = METRICS[metric]
    models = [m for m in MODELS if any(k[2] == m and k[3] == enc for k in runs)]
    print(f'\n### {title}\n')
    print('| domain | dial | ' + ' | '.join(LABEL[m] for m in models) + ' |')
    print('|---|---|' + '---|' * len(models))
    for dom, dials in ORDER.items():
        for d in dials:
            vals = [runs.get((dom, d, m, enc)) for m in models]
            print(f'| {dom} | {d} | ' + ' | '.join('—' if not rs else fmt(mean(map(f, rs))) for rs in vals) + ' |')


def compare(runs, title, metric):
    """linear -> MLP, one cell per model."""
    f, fmt = METRICS[metric]
    models = [m for m in MODELS if any(k[2] == m and k[3] == 'mlp' for k in runs)]
    print(f'\n### {title} (linear → MLP)\n')
    print('| domain | dial | ' + ' | '.join(SHORT[m] for m in models) + ' |')
    print('|---|---|' + '---|' * len(models))
    for dom, dials in ORDER.items():
        for d in dials:
            cells = []
            for m in models:
                lin, mlp = runs.get((dom, d, m, 'linear')), runs.get((dom, d, m, 'mlp'))
                show = lambda rs: '—' if not rs else fmt(mean(map(f, rs)))
                cells.append(f'{show(lin)} → {show(mlp)}')
            print(f'| {dom} | {d} | ' + ' | '.join(cells) + ' |')


def ess_table(runs, enc):
    models = [m for m in CONTRASTIVE if any(k[2] == m and k[3] == enc for k in runs)]
    print('\n### SIR effective sample size: median / min over test queries (bank of 50k)\n')
    print('| domain | dial | ' + ' | '.join(SHORT[m] for m in models) + ' |')
    print('|---|---|' + '---|' * len(models))
    for dom, dials in ORDER.items():
        for d in dials:
            cells = []
            for m in models:
                rs = runs.get((dom, d, m, enc))
                cells.append('—' if not rs else f"{mean(r['extra']['ess_median'] for r in rs):.0f} / "
                                                f"{min(r['extra']['ess_min'] for r in rs):.1f}")
            print(f'| {dom} | {d} | ' + ' | '.join(cells) + ' |')


def latent_table(runs):
    print('\n### Bent GP: can the true un-bent past be read off the learned features?\n')
    print('Held-out R² of a linear fit of the latent (un-bent) quantity on the learned u-features; '
          '"raw" = the same fit on the raw past.  Latent forecast = the un-bent past\'s best linear '
          'forecast (the part of the past that matters).\n')
    print('| dial | model | encoder | latent past: features / raw | latent forecast: features / raw |')
    print('|---|---|---|---|---|')
    for d in ORDER['warp']:
        for m in CONTRASTIVE:
            for enc in ('linear', 'mlp'):
                rs = runs.get(('warp', d, m, enc))
                if not rs or 'r2_latent_past_features' not in rs[0]['extra']:
                    continue
                g = lambda k: mean(r['extra'][k] for r in rs)
                print(f"| {d} | {LABEL[m]} | {enc} | {g('r2_latent_past_features'):.3f} / {g('r2_latent_past_raw'):.3f} "
                      f"| {g('r2_latent_forecast_features'):.3f} / {g('r2_latent_forecast_raw'):.3f} |")


def detail(runs, enc):
    for dom, dials in ORDER.items():
        print(f'\n### {NAMES[dom]} ({enc})\n')
        print('| dial | model | energy vs oracle | skill | avg err vs oracle | band90 | band90 easy / mid / hard | width90 / oracle | seeds |')
        print('|---|---|---|---|---|---|---|---|---|')
        for d in dials:
            for model in MODELS:
                rs = runs.get((dom, d, model, enc), [])
                if not rs:
                    continue
                m = lambda f: mean(f(r) for r in rs)
                ex = m(METRICS['energy'][0])
                sk = m(lambda r: s_(r)['energy_skill'])
                me = m(METRICS['err'][0])
                c = m(METRICS['cov90'][0])
                byb = [m(lambda r, i=i: s_(r)['cov90_by_spread'][i]) for i in range(3)]
                w = m(lambda r: s_(r)['width90'] / o(r)['width90'])
                print(f'| {d} | {LABEL[model]} | {100 * ex:+.1f}% | {sk:.3f} | {me:.2f}x | {c:.3f} | '
                      f'{byb[0]:.2f} / {byb[1]:.2f} / {byb[2]:.2f} | {w:.2f} | {len(rs)} |')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', choices=['linear', 'mlp'], default='linear')
    enc = p.parse_args().encoder
    runs = load()
    grid(runs, enc, f'Energy vs perfect forecaster, {enc} (0% = perfect)', 'energy')
    grid(runs, enc, f'90% band coverage on the HARD third of cases, {enc} (target 0.90)', 'hard')
    grid(runs, enc, f'90% band coverage overall, {enc} (target 0.90)', 'cov90')
    grid(runs, enc, f'Average-prediction error vs oracle, {enc} (1.00x = perfect)', 'err')
    ess_table(runs, enc)
    if enc == 'mlp':
        compare(runs, 'Energy vs perfect forecaster', 'energy')
        compare(runs, '90% band coverage on the HARD third', 'hard')
        latent_table(runs)
    detail(runs, enc)


if __name__ == '__main__':
    main()
