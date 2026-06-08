"""Run all experiments defined in a sweep yaml file.

Usage:
    python run_experiments.py --sweep hyperparam_search.yaml
    python run_experiments.py --sweep experiments.yaml
"""
import argparse
import copy
import traceback
import yaml
from experiment import run


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

    runs = sweep['runs']
    print(f"Starting sweep: {len(runs)} runs\n")

    for i, run_spec in enumerate(runs):
        name = run_spec['name']
        overrides = run_spec.get('overrides', {})
        cfg = deep_merge(base_cfg, overrides)
        cfg['experiment']['name'] = name
        cfg['experiment']['output_dir'] = f"results/{name}"

        print(f"[{i+1}/{len(runs)}] {name}")
        try:
            run(cfg)
        except Exception:
            print(f"  FAILED — continuing to next run")
            traceback.print_exc()

    print("\nSweep complete.")


if __name__ == '__main__':
    main()
