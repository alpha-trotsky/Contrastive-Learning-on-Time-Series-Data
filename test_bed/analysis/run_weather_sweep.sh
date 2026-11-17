#!/usr/bin/env bash
# Goal 3 weather sweep, 4 jobs in parallel, every run through experiment.py.  From test_bed/:
#   bash analysis/run_weather_sweep.sh                                   # setup A, full grid
#   SETUPS="B" bash analysis/run_weather_sweep.sh                        # setup B
#   SETUPS="A" MODELS="l2_joint" ENCODERS=mlp DROPS="rain wind" bash analysis/run_weather_sweep.sh
#   BANK=train SETUPS=A MODELS=l2_joint bash analysis/run_weather_sweep.sh   # bank = training futures
# Runs whose results.json already exists are skipped.
set -u
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/weather
run() {  # setup model encoder seed [drop]
  local drop=${5:-}
  local cfg
  cfg=$(python -m analysis.weather_sweep --setup "$1" --model "$2" --encoder "$3" --seed "$4" \
          ${drop:+--drop "$drop"} ${BANK:+--bank "$BANK"}) || return
  local tag
  tag=$(basename "$cfg" .yaml)
  [ -f "results/weather/$tag/results.json" ] && return
  python experiment.py --config "$cfg" > "logs/weather/$tag.log" 2>&1
}
export -f run
for seed in ${SEEDS:-0 1}; do
  for setup in ${SETUPS:-A}; do
    for enc in ${ENCODERS:-mlp linear}; do
      for model in ${MODELS:-l2_joint ncos gauss mse cosine cosine_1s cosine_joint l2 l2_2s}; do
        for drop in ${DROPS:-none}; do
          # (no trailing blank: xargs -L would join the next line)
          if [ "$drop" = none ]; then echo "$setup $model $enc $seed"; else echo "$setup $model $enc $seed $drop"; fi
        done
      done
    done
  done
done | xargs -P "${JOBS:-4}" -L 1 bash -c 'run "$@"' _
