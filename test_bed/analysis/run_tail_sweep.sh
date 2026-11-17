#!/bin/bash
# Runs every line of results/tail_sweep/jobs.txt as  python -m analysis.tail_sweep <line>,
# 4 at a time (each job is single-threaded).  Usage (from test_bed/):  bash analysis/run_tail_sweep.sh
export WANDB_MODE=disabled
mkdir -p results/tail_sweep/logs
cat "${1:-results/tail_sweep/jobs.txt}" | xargs -P "${JOBS:-4}" -I{} bash -c \
  'log=results/tail_sweep/logs/$(echo "{}" | tr -s " -" "_").log; python -m analysis.tail_sweep {} > "$log" 2>&1'
