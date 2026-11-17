#!/bin/bash
# Embedding-dimension and correlation sweep (from test_bed/):  bash analysis/run_dim_sweep.sh
export WANDB_MODE=disabled
xargs -P 4 -I{} bash -c 'log=results/dim_sweep/logs/$(echo "{}" | tr -s " -," "_").log; python -m analysis.tail_sweep {} --out results/dim_sweep > "$log" 2>&1' < results/dim_sweep/jobs.txt
