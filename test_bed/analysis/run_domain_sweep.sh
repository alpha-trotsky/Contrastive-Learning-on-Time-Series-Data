#!/usr/bin/env bash
# Goal-1 sweep, 4 jobs in parallel.  From test_bed/:  bash analysis/run_domain_sweep.sh
set -u
export WANDB_MODE=disabled
mkdir -p logs/domain_sweep
run() { python -m analysis.domain_sweep --domain "$1" --dial "$2" --model "$3" --seed "$4" \
          > "logs/domain_sweep/${1}_${2}_${3}_s${4}.log" 2>&1; }
export -f run
for seed in 0 1; do
  for spec in "t inf" "t 10" "t 3" "warp 0" "warp 0.5" "warp 1" "dwell 0" "dwell 1" "dwell 1.5"; do
    for model in ${MODELS:-cosine l2 mse}; do
      echo "$spec $model $seed"
    done
  done
done | xargs -P 4 -L 1 bash -c 'run "$@"' _
