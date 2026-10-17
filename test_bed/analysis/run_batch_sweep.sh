#!/usr/bin/env bash
# K (batch-size) sweep, 4 jobs in parallel.  From test_bed/:  bash analysis/run_batch_sweep.sh
set -u
export WANDB_MODE=disabled
mkdir -p logs
run() { python -m analysis.batch_sweep --loss "$1" --batch "$2" --steps "$3" --every "$4" > "logs/${1}_K${2}.log" 2>&1; }
export -f run
printf '%s\n' \
  "cosine 4096 8000 500" \
  "cosine 1024 20000 1000" \
  "cosine 256 30000 1000" \
  "cosine 64 40000 1000" \
  "l2_vgu 1024 15000 1000" \
  "l2_vgu 64 40000 1000" \
  | xargs -P 4 -L 1 bash -c 'run "$@"' _
