"""Goal 3 sweep on the Jena Weather data: write one experiment.py config per run.

A run = (setup, model, encoder, seed[, dropped group]).  Every run goes through
the user's entry point,  python experiment.py --config <file>,  with early
stopping on the validation energy score (same rule for every model).  The
runner is analysis/run_weather_sweep.sh; the tables come from
analysis/summarize_weather_sweep.py.

Setups (u groups -> v groups, see modalities/weather.py):
  A    all six groups              -> future temperature
  B    pressure + wind + moisture  -> future rain      (cross-modal, spiky target)
  B2   all groups but temperature  -> future temperature (cross-modal)
Ablation: setup A with one group dropped from u  (--drop <group>, tag A-<group>).
Bank check: --bank train resamples from the training futures (tag A+trainbank).

Models: the contrastive grid {cosine, L2} x {one-sided, two-sided, joint},
normalised cosine, and the two regression baselines (MSE + residual noise,
Gaussian NLL), each with linear or MLP networks.  MLP runs use
config_weather.yaml (lr 3e-4, 10k steps); linear runs lr 1e-3, 15k steps.
Both settings were picked on the validation period only.

Usage (from test_bed/):
    python -m analysis.weather_sweep --setup A --model l2_joint --encoder mlp --seed 0
prints the path of the written config.
"""
import argparse
import copy
from pathlib import Path

import yaml

from analysis.domain_sweep import LOSSES
from modalities.weather import GROUPS

MODELS = ['cosine_1s', 'cosine', 'cosine_joint', 'ncos', 'l2', 'l2_2s', 'l2_joint', 'mse', 'gauss']
SETUPS = {
    'A': (list(GROUPS), ['temperature']),
    'B': (['pressure', 'wind', 'moisture'], ['rain']),
    'B2': ([g for g in GROUPS if g != 'temperature'], ['temperature']),
}


def make_config(setup, model, encoder, seed, drop=None, steps=None, bank=None, out_root='results/weather'):
    cfg = yaml.safe_load(open('config_weather.yaml'))
    u_groups, v_groups = SETUPS[setup]
    if drop:
        assert drop in u_groups, drop
        u_groups = [g for g in u_groups if g != drop]
    # setup tag: A, A-<dropped group>, A+trainbank (bank = the training futures)
    tag = f"{setup}{'-' + drop if drop else ''}{'+' + bank + 'bank' if bank else ''}_{model}_{encoder}_s{seed}"
    cfg['experiment'].update(name=f'weather_{tag}', output_dir=f'{out_root}/{tag}', seed=seed)
    cfg['modality'].update(u_groups=u_groups, v_groups=v_groups)
    if bank:
        cfg['modality']['bank'] = bank
    if encoder == 'linear':
        # linear models learn slowly and do not over-fit (validation curves)
        cfg['training'].update(lr=1e-3, num_steps=15000)
    if steps:
        cfg['training']['num_steps'] = steps
    mlp = cfg['model']                       # hidden / depth / skip as in config_weather.yaml
    if model in ('mse', 'gauss') and not (model == 'mse' and encoder == 'linear'):
        # regression baselines (models/mlp_regressor.py); depth 0 = linear
        cfg['model'] = {'type': 'mean_regressor' if model == 'mse' else 'gaussian_regressor',
                        'hidden': mlp['hidden'], 'depth': mlp['depth'] if encoder == 'mlp' else 0,
                        'skip': True}
        cfg['loss'] = {'type': 'regression'}
    else:
        if encoder == 'linear':
            cfg['model'] = {'type': 'linear_clip', 'embed_dim': mlp['embed_dim'], 'bias': False,
                            'init_logit_scale': mlp['init_logit_scale']}
        cfg['loss'] = copy.deepcopy(LOSSES[model])
    return tag, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--setup', choices=SETUPS, required=True)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--encoder', choices=['linear', 'mlp'], default='mlp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--drop', choices=list(GROUPS), default=None)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--bank', choices=['train'], default=None,
                   help="resample from the training futures instead of the held-out weeks")
    a = p.parse_args()
    tag, cfg = make_config(a.setup, a.model, a.encoder, a.seed, a.drop, a.steps, a.bank)
    path = Path('results/weather/configs') / f'{tag}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
    print(path)


if __name__ == '__main__':
    main()
