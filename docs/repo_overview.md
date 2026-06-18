# Repository Overview — Contrastive Learning on Time-Series Data

A controlled test bed for studying **CLIP-style contrastive learning** on
synthetic Gaussian-process (GP) data. The goal is to validate, in a setting
where every quantity has a closed form, that learned linear encoders converge
to the theoretical optima derived in Baptista, Stuart & Tran,
*A Mathematical Perspective on Contrastive Learning* (referred to below as
"the paper"; OCR text in `contrastive_learning.md`).

The data are framed as a **past → future forecasting** problem on a 1-D GP:
`u` is the field on an early window of the domain, `v` is the field on a later
window. Because the joint law of `(u, v)` is Gaussian and known analytically,
we can compare the encoders, conditional means, and conditional covariances the
model learns against the paper's predicted closed forms.

---

## Directory layout (`test_bed/`)

```
test_bed/
├── config.yaml              # single-run hyperparameters (the active baseline)
├── experiments.yaml         # loss-variant sweep (7 runs at 50×50)
├── dimensionsweep.yaml      # dimensionality sweep (currently all commented out)
├── hyperparam_search.yaml   # lr / batch-size search
├── experiment.py            # single experiment: build → train → evaluate → log
├── run_experiments.py       # runs every entry in a sweep yaml, saving per-run outputs
│
├── signals/
│   ├── base.py              # SignalGenerator abstract interface
│   └── gaussian_process.py  # GaussianProcess1D: 1-D GP via KL/IDCT expansion
│
├── modalities/
│   ├── base.py              # PairSampler interface (+ optional C_uu / C_vv / C_uv)
│   ├── past_future.py       # PastFutureModality: u = past window, v = future window
│   └── field_coeff.py       # FieldCoeffModality: u = noisy field points, v = KL coeffs
│
├── models/
│   └── linear_clip.py       # LinearCLIP: two linear encoders + learnable logit_scale
│
├── losses/
│   ├── clip_losses.py       # CLIPConditionalLoss, CLIPJointLoss, MSEloss
│   └── one_way_conditional.py
│
├── evaluation/
│   ├── theory_match.py      # ‖A_learned − A*‖_F / ‖A*‖_F  vs each analytical optimum
│   ├── forecast.py          # conditional means, conditional covariances, forecast MSE
│   ├── similarity.py        # in-run single-sample self vs random cosine similarity
│   └── retrieval.py         # post-training fixed-batch recall@k + averaged similarity
│
└── theory/
    └── gaussian_predictions.py  # closed-form A* for each loss (Thms 5.1, 5.3, 5.6)
```

---

## Data pipeline

### Signal — `GaussianProcess1D` (`signals/gaussian_process.py`)

Samples a zero-mean 1-D GP on a grid of `dim_true` points via a
Karhunen–Loève / IDCT-2 expansion:

```
field(t) = Σ_j sqrt(λ_j) · ξ_j · φ_j(t),   ξ_j ~ N(0,1)
λ_j      = (π² j² + τ²)^(−α)
φ_j      = orthonormal cosine basis (IDCT-2, Neumann BCs)
```

- `α` controls smoothness, `τ` the length scale.
- `zero_mean=True` zeroes the constant mode (`λ_0 = 0`).
- `coeff_to_field_map()` returns the matrix `Φ` with `field = ξ @ Φ.T`. This
  single matrix is what makes the modality covariances closed-form.

### Modality — `PastFutureModality` (`modalities/past_future.py`) — **active**

Splits one GP realization into two windows of grid indices:

| Variable | Definition |
|----------|------------|
| `u` | `field[past_index]  (+ N(0, σ_u²))` — past window |
| `v` | `field[future_index] (+ N(0, σ_v²))` — future window |

Because `field = ξ @ Φ.T` with `ξ ~ N(0, I)`, the exact covariances are just
sub-blocks of `Φ Φ^T`:

```
C_uu = Φ_p Φ_pᵀ (+ σ_u² I)
C_vv = Φ_f Φ_fᵀ (+ σ_v² I)
C_uv = Φ_p Φ_fᵀ
```

where `Φ_p = Φ[past_index]`, `Φ_f = Φ[future_index]`. These feed the theory and
evaluation layers.

### Modality — `FieldCoeffModality` (`modalities/field_coeff.py`) — alternate

`u` = noisy pointwise field observations, `v` = leading KL coefficients
`ξ[:dim_coeff]`. Present and wired into `build_modality`, but the active
`config.yaml` uses `PastFuture`.

---

## Model — `LinearCLIP` (`models/linear_clip.py`)

Two bias-free linear encoders mapping each modality into a shared `embed_dim`
space, plus a learnable `logit_scale` (initialized to the CLIP default
`log(1/0.07)`):

```
f_u(u) = W_u u,    f_v(v) = W_v v
```

The quantity compared against theory is the **cross-term**

```
A_hat = cross_term() = logit_scale.exp() · W_uᵀ W_v      # (dim_u × dim_v)
```

with the temperature folded in. `encode_u` / `encode_v` expose the raw
embeddings used by the retrieval metrics.

---

## Losses (`losses/clip_losses.py`)

All losses share the logit construction in `CLIPLoss.get_logits`, switched by
`inner_product`:

- `inner_product=True` → **cosine tilting**: `logits = s · g_u g_vᵀ`.
- `inner_product=False` → **L² tilting**: `logits = s · (g_u·g_v − ½|g_u|² − ½|g_v|²)`,
  i.e. `−½|g_u − g_v|²` up to constants.

| Class | Objective |
|-------|-----------|
| `CLIPConditionalLoss` | `λ_u · CE(logits_per_u) + λ_v · CE(logits_per_v)` — two-sided when both λ = 0.5, one-sided when one λ = 0 |
| `CLIPJointLoss` | InfoNCE-style: `mean(positive diagonal) − logsumexp(negatives)` |
| `MSEloss` | plain regression: `‖u W_uᵀ W_v − v‖²` (uses raw `v_encoder` weight; `logit_scale` never enters, so it stays frozen) |

`experiment.py::build_loss` maps a config `loss.type`
(`conditional` / `joint` / `one_sided_u` / `one_sided_v` / `mse`) onto these.

---

## Theory layer (`theory/gaussian_predictions.py`)

Closed-form optimal cross-term `A*` for each loss family:

| Function | Paper result | Formula |
|----------|--------------|---------|
| `predicted_A_conditional` | Thm 5.1 (cosine conditional) | `C_uu⁻¹ C_uv C_vv⁻¹` |
| `predicted_A_joint` | Thm 5.6 (cosine joint / InfoNCE) | `C_uu^(−½) U·diag(h(σ))·Vᵀ C_vv^(−½)`, `h(σ)=(1/σ)(½√(1+4σ²)−½)` |
| `predicted_A_quadratic_v` | Thm 5.3 (L² one-sided `v\|u`) | `C_uu⁻¹ C_uv C_{v\|u}⁻¹` |
| `predicted_A_quadratic_u` | Thm 5.3 (L² one-sided `u\|v`) | `C_{u\|v}⁻¹ C_uv C_vv⁻¹` |

Two-sided L² and L² joint have **no** closed form and are reported as `none`.

---

## Evaluation layer

### `theory_match.py`
`theory_match_error(model, modality, target)` →
`‖A_learned − A*‖_F / ‖A*‖_F`, dispatching to the right `A*` via `target`.
For `mse` it compares the **raw** map `W_uᵀ W_v` against `K = C_uu⁻¹ C_uv`
(no `logit_scale`); returns `None` for `target='none'`.

### `forecast.py` — conditional means, covariances, forecast MSE
The model's prediction head depends on the **tilting**, resolved by
`experiment.py::resolve_mean_family`:

| Family | Model `E_ν[v\|u]` (paper eq.) | Model `Cov_ν[v\|u]` |
|--------|-------------------------------|----------------------|
| `cosine` | `u A C_vv` (eq. 41b) | `C_vv` — prior, never shrinks (Cor. 5.2) |
| `quadratic` | `u A (C + C_vv⁻¹)⁻¹` (eq. 48b), `C = s W_vᵀ W_v` | `(C + C_vv⁻¹)⁻¹` — can match truth (Cor. 5.4) |
| `mse` | `u (W_uᵀ W_v)` (plain regression) | `None` — point predictor |

Metrics:
- `forecast_mse` — MSE of the model head vs the data MMSE mean `u K`, `K = C_uu⁻¹ C_uv`.
- `residual_cov_error` — sampling-based: covariance of residuals `v − E_ν[v|u]` vs
  the irreducible `C_{v|u} = C_vv − C_uvᵀ C_uu⁻¹ C_uv`. Works for every family
  (including `mse`).
- `model_cov_error` — purely analytic: the model's *claimed* `Cov_ν[v|u]` vs
  `C_{v|u}`; `None` for `mse`.

### `similarity.py`
`compute_similarities` — single-sample in-run diagnostic: cosine of
predicted-`v` (`= g_u W_v`) against its true `v`, vs the mean cosine against
`n_samples` random `v`. High ratio ⇒ good alignment. (Noisy because it uses one
query; the stable version lives in `retrieval.py`.)

### `retrieval.py`
`evaluate_encoders` — post-training, on a **fixed-seed** test batch so every
loss variant is scored on identical data:
- cross-modal **recall@k** (`u→v` and `v→u`) in normalized embedding space;
- averaged predicted-vs-true cosine `self_sim`, vs shuffled `rand_sim`, and the
  `margin` between them.

---

## Running

Single run (uses `config.yaml`):
```bash
cd test_bed
python experiment.py --config config.yaml
```

Sweep (one run per entry, continues past failures, saves `model.pt` /
`history.pt` / `config.yaml` / `eval.json` under `OUTPUT_ROOT/results/<name>`):
```bash
python run_experiments.py --sweep experiments.yaml
```

`OUTPUT_ROOT` defaults to `/kaggle/working` on Kaggle, else the current dir.
All metrics are streamed to **Weights & Biases** (project `contrastive-ts`);
the in-script matplotlib plotting is currently commented out in favour of W&B.

### Active configuration (`config.yaml`)
GP `α=2, τ=3, dim_true=1000`; `PastFuture` `50×50` windows, `σ_u=σ_v=0.05`;
`embed_dim=100`; cosine conditional loss `λ_u=λ_v=0.5`; Adam `lr=5e-4`,
`batch_size=256`, `50 000` steps.

---

## Status / known gaps

- `signals/sines.py`, `modalities/spectral.py`, `losses/one_way_conditional.py`,
  `models/encoders.py` are stubs or duplicates.
- No Monte-Carlo conditional-covariance estimator exists yet for the MSE model;
  the report's "estimated in a Monte-Carlo fashion" is currently covered only by
  `residual_cov_error` (residual-based), not a dedicated sampler.
- `theory_err` logs `NaN` on the L²-joint / two-sided-L² runs by design
  (`target='none'`), and the cosine baseline showed `theory_err ≈ NaN` at the
  start of the crashed 2026-06-14 run — worth confirming it is the
  expected-`None` path and not a numerical issue.
</content>
