# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing **supervised learning** vs **contrastive learning** for time series forecasting Uses synthetic sinusoidal data as a proof of concept; real S&P 500 data (`sp500_data.csv`) is available for future experiments.

## Running Experiments

```bash
cd Synthetic_Proof_of_Concept

# Original version (overlapping windows for contrastive pairs)
python synthetic_experiment.py

# Updated version (independent A/B signal pairs to eliminate false negatives)
python synthetic_exp_updated_CL.py
```

Requires a wandb login before running: `wandb login`

**Minimal dependencies:** `torch`, `matplotlib`, `pyyaml`, `wandb`

## Configuration

All hyperparameters live in `Synthetic_Proof_of_Concept/config.yaml`:
- `sequence_length` — sliding window size L
- `temperature` — InfoNCE loss temperature scaling
- `num_epochs` / `num_probe_epochs` — training epochs for encoder and linear probe respectively
- `future_steps` — autoregressive rollout length for evaluation

## Architecture

Both scripts follow the same 4-phase workflow:

1. **Joint training** — `SupervisedMLP` (MSE loss) and `TimeSeriesEncoder` (InfoNCE loss) train simultaneously on synthetic sinusoidal data
2. **Probe training** — A `LinearProbe` trains on frozen encoder embeddings to predict the next scalar value
3. **Forecasting** — Autoregressive n-step rollout comparing both approaches
4. **Visualization** — Saves `forecast_plot.png` and logs everything to wandb

**Models:**
- `SupervisedMLP`: 2-layer MLP mapping window x_t → predicted next window x̂_{t+1}, MSE loss
- `TimeSeriesEncoder`: 2-layer MLP producing L2-normalized embeddings, symmetric InfoNCE (CLIP-style) loss
- `LinearProbe`: Trained on frozen encoder; `synthetic_experiment.py` uses a single linear layer, `synthetic_exp_updated_CL.py` uses a 2-layer MLP

**Key design difference between scripts:**
- `synthetic_experiment.py`: Contrastive pairs are adjacent windows from a single long series (risk of false negatives from overlap)
- `synthetic_exp_updated_CL.py`: Uses `ContrastiveABDataset` — each sample draws two fully independent short signals (non-overlapping A/B windows), eliminating false negatives

## Experiment Tracking

Results are logged to Weights & Biases (project: `synthetic_ts_contrastive`). wandb run artifacts are stored under `Synthetic_Proof_of_Concept/wandb/`.
