"""Run all experiments defined in a sweep yaml file.

Usage:
    python run_experiments.py --sweep hyperparam_search.yaml
    python run_experiments.py --sweep experiments.yaml

Outputs (model.pt / history.pt / eval.json) are written under OUTPUT_ROOT,
which defaults to /kaggle/working on Kaggle (the only writable place there;
/kaggle/input is read-only) and to the current directory otherwise.  Override
with the OUTPUT_ROOT env var.
"""
import argparse
import copy
import json
import os
import traceback
from pathlib import Path

import torch
import yaml
from experiment import run


def output_root():
    env = os.environ.get('OUTPUT_ROOT')
    if env:
        return Path(env)
    if os.path.isdir('/kaggle/working'):
        return Path('/kaggle/working')
    return Path('.')


def deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True)
    args = parser.parse_args()

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)

    with open(sweep['base_config']) as f:
        base_cfg = yaml.safe_load(f)

    root = output_root()
    runs = sweep['runs']
    print(f"Starting sweep: {len(runs)} runs | output root: {root.resolve()}\n")

    for i, run_spec in enumerate(runs):
        name = run_spec['name']
        overrides = run_spec.get('overrides', {})
        cfg = deep_merge(base_cfg, overrides)
        out_dir = root / 'results' / name
        cfg['experiment']['name'] = name
        cfg['experiment']['output_dir'] = str(out_dir)

        print(f"[{i+1}/{len(runs)}] {name}")
        try:
            model, history = run(cfg)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / 'model.pt')
            torch.save(history, out_dir / 'history.pt')
            with open(out_dir / 'config.yaml', 'w') as f:
                yaml.dump(cfg, f)
            with open(out_dir / 'eval.json', 'w') as f:
                json.dump(history.get('eval', {}), f, indent=2)
            print(f"  saved -> {out_dir}")
        except Exception:
            print(f"  FAILED — continuing to next run")
            traceback.print_exc()

    print("\nSweep complete.")


if __name__ == '__main__':
    main()
