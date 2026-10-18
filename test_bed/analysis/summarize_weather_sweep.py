"""Tables for the goal 3 weather sweep (results/weather/*/results.json).

Usage (from test_bed/):
    python -m analysis.summarize_weather_sweep --setup A        # -> results/weather/summary_A.md
    python -m analysis.summarize_weather_sweep --setup A --ablation l2_joint_mlp

All numbers are on the TEST period (late Oct - Dec 2020), mean over seeds, at the
checkpoint early stopping picked on the validation period.  Columns:
  skill vs clim / persist   1 - energy / energy of climatology (random bank futures)
                            or of persistence (last value repeated); higher is better
  CRPS                      per horizon step, in the target's own units
  cov50 / cov90 / width90   band coverage (targets 0.50 / 0.90) and 90% width
  cov90 by past-vol         90% coverage per query, in terciles of the past day's
                            volatility of the target (calm / mid / changeable)
  cov90 by NLL width        same, in terciles of the Gaussian-NLL MLP's predicted
                            width (same setup and seed): its own idea of difficulty
  bias                      mean error of the forecast mean
  ESS                       effective number of bank futures per query (median / min)
  stop                      step early stopping picked
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('results/weather')
LABEL = {'cosine_1s': 'cos 1-sided', 'cosine': 'cos 2-sided', 'cosine_joint': 'cos joint',
         'ncos': 'norm. cos', 'l2': 'L² 1-sided', 'l2_2s': 'L² 2-sided', 'l2_joint': 'L² joint',
         'mse': 'MSE + noise', 'gauss': 'Gauss NLL'}
ORDER = list(LABEL)
STEPS = (0, 5, 17, 35)            # horizon steps shown: 10 min, 1 h, 3 h, 6 h


def load(setup):
    """{(model, encoder): [results of each seed]} for one setup tag (e.g. 'A', 'A-rain')."""
    runs = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / f'{setup}_*' / 'results.json'))):
        tag = Path(f).parent.name
        rest = tag[len(setup) + 1:]
        if not rest[0].islower() or '_s' not in rest:
            continue
        model_enc, seed = rest.rsplit('_s', 1)
        model, enc = model_enc.rsplit('_', 1)
        r = json.load(open(f))
        r['seed'] = int(seed)
        runs[(model, enc)].append(r)
    return runs


def terciles(values, proxy):
    v, p = np.asarray(values), np.asarray(proxy)
    edges = np.quantile(p, [0, 1 / 3, 2 / 3, 1])
    return [v[(p >= edges[i]) & (p <= edges[i + 1])].mean() for i in range(3)]


_queries = {}


def nll_width_proxy(setup, seed):
    """Predicted width (mean per-step std) of the Gaussian-NLL MLP on the test
    queries, rebuilt exactly as experiment.run() drew them (seed 4321)."""
    key = (setup, seed)
    if key in _queries:
        return _queries[key]
    path = ROOT / f'{setup}_gauss_mlp_s{seed}'
    if not (path / 'model.pt').exists():
        _queries[key] = None
        return None
    from experiment import build_modality, build_model
    cfg = json.load(open(path / 'results.json'))['config']
    mod = build_modality(cfg['modality'], None)
    model = build_model(cfg['model'], mod)
    model.load_state_dict(torch.load(path / 'model.pt'))
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(4321)
        e = mod.split('test').eval_set(cfg['blackbox']['n_test_final'])
    with torch.no_grad():
        _, sd = model.predict(e['u'])
    _queries[key] = sd.double().mean(1).numpy()
    return _queries[key]


def scale_of(r):
    """Std of the first target column (standardised -> physical units)."""
    from modalities.weather import WeatherModality
    cfg = r['config']['modality']
    key = tuple(cfg['v_groups'])
    if key not in scale_of.cache:
        mod = WeatherModality(cfg['u_groups'], cfg['v_groups'], cfg['past_len'], cfg['future_len'],
                              cfg['past_stride'], tuple(cfg['fractions']), cfg['bank'],
                              cfg['bank_block'], cfg['bank_every'])
        scale_of.cache[key] = (mod.v_scale.numpy(), mod.v_cols[0])
    return scale_of.cache[key]


scale_of.cache = {}


def row_stats(rs, setup):
    """Seed-averaged numbers for one (model, encoder)."""
    out = defaultdict(list)
    for r in rs:
        d = r['blackbox_detail']
        F = d['forecasters']
        m = F['model']
        s = m['scores']
        scale, _ = scale_of(r)
        out['skill'].append(s['energy_skill'])
        out['skill_p'].append(1 - s['energy'] / F['persistence']['scores']['energy'])
        out['crps'].append(np.asarray(m['crps_step'])[list(STEPS)] * scale[list(STEPS)])
        out['crps_mean'].append(np.mean(np.asarray(m['crps_step']) * scale))
        out['cov50'].append(s['cov50'])
        out['cov90'].append(s['cov90'])
        out['width90'].append(s['width90'] * scale[0])
        out['bias'].append(np.mean(m['bias_step'][:r['config']['modality']['future_len']]) * scale[0])
        out['cv_vol'].append(terciles(m['cov90_query'], d['proxy']['past_vol']))
        # the test queries do not depend on the u groups, so ablations reuse setup A's NLL model
        w = nll_width_proxy(setup.split('-')[0], r['seed'])
        out['cv_nll'].append(terciles(m['cov90_query'], w) if w is not None else [math.nan] * 3)
        out['cov_step'].append(np.asarray(m['cov90_step'])[list(STEPS)])
        h = np.asarray(m['pit_hist'])
        out['pit_tails'].append((h[0] + h[-1]) / h.sum())
        out['pit_low'].append(h[:5].sum() / h.sum())
        if 'ess' in d:
            out['ess_med'].append(np.median(d['ess']))
            out['ess_min'].append(np.min(d['ess']))
        if 'below_floor' in m:
            out['below_floor'].append(m['below_floor'])
            out['brier'].append(m['brier_any'])
            out['event_rate'].append(m['event_rate'])
        out['stop'].append(r.get('best_step'))
    return {k: np.mean(np.asarray(v, dtype=float), axis=0) for k, v in out.items()}, len(rs)


def refs_stats(rs):
    """Reference forecasters (climatology, persistence, seasonal naive) from any run."""
    r = rs[0]
    F = r['blackbox_detail']['forecasters']
    scale, _ = scale_of(r)
    out = {}
    for k in ('climatology', 'persistence', 'seasonal_naive'):
        s = F[k]['scores']
        out[k] = {'skill': s['energy_skill'],
                  'skill_p': 1 - s['energy'] / F['persistence']['scores']['energy'],
                  'crps': np.asarray(F[k]['crps_step'])[list(STEPS)] * scale[list(STEPS)],
                  'crps_mean': np.mean(np.asarray(F[k]['crps_step']) * scale),
                  'cov90': s.get('cov90', math.nan), 'brier': F[k].get('brier_any', math.nan),
                  'below_floor': F[k].get('below_floor', math.nan)}
    return out


f2 = lambda x: '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.2f}'
f3 = lambda x: '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'
# CRPS / width: 3 significant digits (rain amounts are small numbers)
g3 = lambda x: '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3g}'


def summary(setup):
    runs = load(setup)
    if not runs:
        raise SystemExit(f'no results for setup {setup}')
    any_r = next(iter(runs.values()))[0]
    _, col = scale_of(any_r)
    cfg = any_r['config']['modality']
    keys = sorted(runs, key=lambda k: (k[1] != 'mlp', ORDER.index(k[0]) if k[0] in ORDER else 99))
    stats = {k: row_stats(runs[k], setup) for k in keys}
    L = [f"# Weather sweep, setup {setup}: {' + '.join(cfg['u_groups'])} → future {' + '.join(cfg['v_groups'])}",
         '',
         f"Test period {any_r['data']['test']['from']} – {any_r['data']['test']['to']}, "
         f"{any_r['config']['blackbox']['n_test_final']} test queries, bank = "
         f"{any_r['data']['bank']['windows']} held-out futures ({cfg['bank']}). Mean over seeds "
         f"(n = number of seeds). CRPS and width in the units of `{col}`.",
         '',
         '## Main table',
         '',
         '| model | enc | n | skill vs clim | skill vs persist | CRPS 10m / 1h / 3h / 6h | cov50 | cov90 | width90 | bias | ESS med / min | stop |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in keys:
        s, n = stats[k]
        ess = f"{s['ess_med']:.0f} / {s['ess_min']:.1f}" if 'ess_med' in s else '—'
        L.append(f"| {LABEL.get(k[0], k[0])} | {k[1]} | {n} | {f3(s['skill'])} | {f3(s['skill_p'])} | "
                 + ' / '.join(g3(x) for x in s['crps']) + f" | {f2(s['cov50'])} | {f2(s['cov90'])} | "
                 f"{g3(s['width90'])} | {s['bias']:+.3g} | {ess} | {s['stop']:.0f} |")
    for name, r in refs_stats(next(iter(runs.values()))).items():
        L.append(f"| *{name.replace('_', '-')}* | — | — | {f3(r['skill'])} | {f3(r['skill_p'])} | "
                 + ' / '.join(g3(x) for x in r['crps']) + f" | — | {f2(r['cov90'])} | — | — | — | — |")
    L += ['', '## Calibration detail', '',
          'cov90 per query, by tercile of a difficulty proxy (calm / mid / hard), coverage by horizon, '
          'and the PIT histogram (ideal: 0.20 of values in the two outer bins, 0.50 in the lower half).',
          '',
          '| model | enc | by past-vol | by NLL width | cov90 at 10m / 1h / 3h / 6h | PIT outer bins | PIT lower half |',
          '|---|---|---|---|---|---|---|']
    for k in keys:
        s, _ = stats[k]
        L.append(f"| {LABEL.get(k[0], k[0])} | {k[1]} | " + ' / '.join(f2(x) for x in s['cv_vol'])
                 + ' | ' + ' / '.join(f2(x) for x in s['cv_nll']) + ' | '
                 + ' / '.join(f2(x) for x in s['cov_step'])
                 + f" | {f2(s['pit_tails'])} | {f2(s['pit_low'])} |")
    if any('brier' in stats[k][0] for k in keys):
        refs = refs_stats(next(iter(runs.values())))
        L += ['', '## Rain-specific', '',
              '"below floor": fraction of forecast samples below 0 (impossible rain). '
              '"Brier (any rain)": squared error of the forecast probability that it rains at all in '
              f"the next {cfg['future_len'] // 6} h (lower is better).",
              '',
              '| model | enc | below floor | Brier (any rain) |', '|---|---|---|---|']
        for k in keys:
            s, _ = stats[k]
            L.append(f"| {LABEL.get(k[0], k[0])} | {k[1]} | {f3(s['below_floor'])} | {f3(s['brier'])} |")
        for name in ('climatology', 'persistence', 'seasonal_naive'):
            L.append(f"| *{name.replace('_', '-')}* | — | {f3(refs[name]['below_floor'])} | {f3(refs[name]['brier'])} |")
        L.append(f"\nObserved rate of 'any rain in the window' on the test queries: "
                 f"{stats[keys[0]][0]['event_rate']:.3f}.")
    return '\n'.join(L) + '\n'


def ablation(setup, model_enc):
    """Energy skill of `model_enc` (e.g. l2_joint_mlp) with each group dropped from u."""
    model, enc = model_enc.rsplit('_', 1)
    full = load(setup).get((model, enc))
    rows = []
    groups = json.load(open(glob.glob(str(ROOT / f'{setup}_{model_enc}_s*' / 'results.json'))[0]))[
        'config']['modality']['u_groups']
    base, _ = row_stats(full, setup)
    L = [f'# Modality ablation, setup {setup}, {LABEL.get(model, model)} ({enc})', '',
         'Each row retrains with one group dropped from the past. Δ = change against the full model '
         '(negative skill change = the group helped). Test period, mean over seeds.', '',
         '| dropped | n | skill vs clim | Δ skill | CRPS 1h / 6h | Δ CRPS 6h | cov90 | width90 | cov90 by past-vol |',
         '|---|---|---|---|---|---|---|---|---|',
         f"| (none) | {len(full)} | {f3(base['skill'])} | — | {f2(base['crps'][1])} / {f2(base['crps'][3])} | — | "
         f"{f2(base['cov90'])} | {f2(base['width90'])} | " + ' / '.join(f2(x) for x in base['cv_vol']) + ' |']
    for g in groups:
        rs = load(f'{setup}-{g}').get((model, enc))
        if not rs:
            continue
        s, n = row_stats(rs, f'{setup}-{g}')
        L.append(f"| {g} | {n} | {f3(s['skill'])} | {s['skill'] - base['skill']:+.3f} | "
                 f"{f2(s['crps'][1])} / {f2(s['crps'][3])} | {s['crps'][3] - base['crps'][3]:+.3f} | "
                 f"{f2(s['cov90'])} | {f2(s['width90'])} | " + ' / '.join(f2(x) for x in s['cv_vol']) + ' |')
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--setup', default='A')
    p.add_argument('--ablation', default=None, help='model_encoder, e.g. l2_joint_mlp')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.ablation:
        text, out = ablation(a.setup, a.ablation), ROOT / f'ablation_{a.setup}_{a.ablation}.md'
    else:
        text, out = summary(a.setup), ROOT / f'summary_{a.setup}.md'
    out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
