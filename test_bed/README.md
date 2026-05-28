# Test Bed: Theoretical CLIP on Gaussian Processes

A controlled experimental framework for studying **contrastive learning (CLIP-style)** on synthetic Gaussian process data. The primary goal is to validate that learned encoder weights converge to theoretically predicted optima.

## What It Does

1. **Generates synthetic data** — Gaussian process realizations observed at sparse spatial locations (modality u) paired with the underlying KL coefficients (modality v)
2. **Trains a linear CLIP model** — two linear encoders with a learnable logit scale, optimized with a contrastive loss
3. **Compares against theory** — tracks whether the learned cross-term `W_u^T W_v` converges to the analytically derived optimal `A*`
4. **Evaluates downstream quality** — measures forecast MSE using the learned encoder as a predictor

## Directory Structure

```
test_bed/
├── config.yaml               # Experiment hyperparameters
├── experiment.py             # Main training loop and entry point
│
├── signals/                  # Signal generators (data sources)
│   ├── base.py               # Abstract SignalGenerator interface
│   ├── gaussian_process.py   # 1D GP via KL expansion (Matérn-like spectrum)
│   └── sines.py              # (stub)
│
├── modalities/               # Paired observation models (u, v samplers)
│   ├── base.py               # Abstract PairSampler interface
│   ├── field_coeff.py        # u = noisy field values; v = truncated KL coefficients
│   └── spectral.py           # (stub)
│
├── models/
│   ├── linear_clip.py        # LinearCLIP: two linear encoders + learnable logit scale
│   └── encoders.py           # (duplicate of linear_clip.py)
│
├── losses/
│   ├── clip_losses.py        # CLIPLoss, CLIPConditionalLoss, CLIPJointLoss
│   ├── standard_CLIP.py      # Alternate implementation (same classes)
│   └── one_way_conditional.py # (stub)
│
├── evaluation/
│   ├── theory_match.py       # Relative Frobenius error vs theoretical A*
│   └── forecast.py           # MSE of u predictions from v via learned cross-term
│
└── theory/
    └── gaussian_predictions.py  # Analytical optimal encoders (Theorems 5.1 & 5.6)
```

## Data Pipeline

### Signal: `GaussianProcess1D`

Generates 1D GP samples via KL expansion on a grid of `dim_true` points over [0, 1]:

```
field(t) = Σ_j  sqrt(λ_j) · ξ_j · φ_j(t)
```

- **Eigenvalues**: `λ_j = (π²j² + τ²)^(-α)` — controls smoothness (`α`) and length scale (`τ`)
- **Basis**: orthonormal cosine functions (IDCT-2, Neumann BCs)
- **Coefficients** `ξ`: i.i.d. standard normal

### Modality: `FieldCoeffModality`

Produces paired observations `(u, v)` from each GP sample:

| Variable | Definition |
|----------|-----------|
| `u` | `field[u_indices] + N(0, σ²I)` — noisy pointwise field observations |
| `v` | `ξ[:dim_coeff]` — noiseless leading KL coefficients |

Analytical covariances `C_uu`, `C_vv`, `C_uv` are available for theory comparison.

## Model: `LinearCLIP`

Two linear encoders (no bias by default) mapping each modality to a shared `embed_dim`-dimensional space:

```
f_u(u) = W_u u,    f_v(v) = W_v v
```

The key quantity for downstream tasks and theory comparison is the **cross-term**:

```
A^ = logit_scale · W_u^T W_v    (shape: dim_u × dim_v)
```

## Losses

| Loss | Description |
|------|-------------|
| `CLIPConditionalLoss` | Symmetric cross-entropy on both modalities; weights controlled by `lambda_u`, `lambda_v` |
| `CLIPJointLoss` | InfoNCE-style: maximizes diagonal log-probability minus logsumexp over negatives |
| `CLIPLoss` | Base class; supports `inner_product` and `l2` similarity modes |

One-sided variants are available by setting one lambda to zero.

## Theory

`theory/gaussian_predictions.py` derives the analytically optimal cross-term `A*` for each loss type:

**Conditional loss (Theorem 5.1):**
```
A* = C_uu^(-1) C_uv C_vv^(-1)
```

**Joint/InfoNCE loss (Theorem 5.6):**
```
A* = C_uu^(-1/2) · U · diag(h(σ)) · V^T · C_vv^(-1/2)

where  M = C_uu^(-1/2) C_uv C_vv^(-1/2) = U S V^T  (SVD)
and    h(σ) = (1/σ)(0.5√(1 + 4σ²) − 0.5)
```

## Evaluation Metrics

- **Theory match error** — `‖A^ − A*‖_F / ‖A*‖_F`: how close the learned cross-term is to the theoretical optimum
- **Forecast MSE** — mean squared error of `u_pred = v @ A^T` versus the MMSE baseline `u_cond = v @ (C_vv^(-1) C_uv^T)^T`

## Running an Experiment

```bash
cd test_bed
python experiment.py
```

Outputs are saved to a timestamped directory (configured in `config.yaml`), including the saved model weights, training history, and a 3-panel plot of loss / theory error / forecast MSE over training.

## Configuration (`config.yaml`)

```yaml
experiment:
  name: phase1_gp_conditional_baseline
  seed: 42
  num_steps: 20000
  log_every: 100
  eval_every: 500

signal:
  type: gp1d
  alpha: 2          # smoothness exponent
  tau: 3.0          # length-scale regularizer
  dim_true: 1000    # spatial grid resolution
  zero_mean: true

modality:
  type: field_coeff
  u_indices: [evenly spaced, 12 points]
  dim_coeff: 5
  noise_std: 0.05

model:
  type: linear_clip
  embed_dim: 5

loss:
  type: conditional
  lambda_u: 0.5
  lambda_v: 0.5
  similarity: inner_product

optimizer:
  type: adam
  lr: 1.0e-4
  batch_size: 1024
```

## Stubs (Not Yet Implemented)

- `signals/sines.py` — sinusoidal signal generator (cf. `Synthetic_Proof_of_Concept/`)
- `modalities/spectral.py` — spectral coefficient modality
- `losses/one_way_conditional.py` — asymmetric one-way conditional loss
