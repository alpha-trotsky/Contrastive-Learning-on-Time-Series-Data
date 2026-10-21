#!/usr/bin/env bash
# goal 3: everything after setup A, in order
cd "$(dirname "$0")/.."
SETUPS=B bash analysis/run_weather_sweep.sh
SETUPS=A ENCODERS=linear MODELS="l2_joint gauss" DROPS="temperature moisture pressure wind radiation rain" bash analysis/run_weather_sweep.sh
SETUPS=Arel MODELS="l2_joint ncos cosine mse gauss" bash analysis/run_weather_sweep.sh
BANK=train SETUPS=A MODELS="l2_joint" bash analysis/run_weather_sweep.sh
echo CHAIN DONE
