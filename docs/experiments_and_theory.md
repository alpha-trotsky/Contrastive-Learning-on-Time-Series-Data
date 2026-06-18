# Experiments & Relation to the Paper

This document maps the test bed's experiments onto the theory in Baptista,
Stuart & Tran, *A Mathematical Perspective on Contrastive Learning*
(`contrastive_learning.md`). The paper analyzes contrastive learning in the
multivariate-Gaussian setting, where latent-space identification becomes a
low-rank matrix-approximation problem with closed-form optima. Our GP test bed
*is* that Gaussian setting, built from a `past → future` forecasting task so the
encoders, conditional means, and conditional covariances all have references to
check against.

---

## 1. Setup as an instance of the paper's framework

The paper couples two modalities `u, v` through a common latent `w` (eq. 1) and
studies the population losses `L_cond`, `L_joint`, and their tiltings. In our
test bed:

- `w` = the latent GP coefficients `ξ ~ N(0, I)`.
- `u = Φ_p ξ (+ noise)` (past window) and `v = Φ_f ξ (+ noise)` (future window),
  so `(u, v)` is jointly Gaussian with `C_uu, C_vv, C_uv` given in closed form
  (`modalities/past_future.py`).
- Encoders are **linear** (`LinearCLIP`), so the learned object reduces to the
  cross-term `A_hat = s · W_uᵀ W_v`, exactly the matrix the paper's Gaussian
  analysis predicts.

This is the cleanest possible realization of the paper's §5 analysis: the
experiments measure whether `A_hat` converges to the predicted `A*`, and whether
the implied conditional mean/covariance match the data statistics.

---

## 2. Engineering metrics and their theoretical references

Three families of metric are computed; each has a per-loss closed form.

### 2.1 Conditional mean — forecast MSE

The model's `E_ν[v|u]` depends on the **tilting** (`resolve_mean_family`):

| Loss / tilting | Model `E_ν[v\|u]` | Paper eq. |
|----------------|-------------------|-----------|
| cosine (`inner_product=True`) | `C_vv Aᵀ u`  → batch `u A C_vv` | (41b) |
| L² (`inner_product=False`) | `(C + C_vv⁻¹)⁻¹ Aᵀ u`, `C = s W_vᵀ W_v` | (48b) |
| MSE | `(W_uᵀ W_v)ᵀ u` (plain regression head) | — |

`forecast_mse` compares this head against the **data MMSE mean** `u K`,
`K = C_uu⁻¹ C_uv`. It → 0 exactly when the model recovers the true conditional
mean (e.g. cosine conditional at `A* = C_uu⁻¹ C_uv C_vv⁻¹`, where `u A* C_vv = u K`).

> Note on the report draft: the "$L^2$ quadratic" mean line should read
> `v_pred = u A (C + C_vv⁻¹)⁻¹` with `C = s W_vᵀ W_v` (per `forecast.py`), and
> the MSE head is `u (W_uᵀ W_v)` — i.e. `W_uᵀ W_v u` in column form, matching
> the draft's `W_uᵀ W_v u`.

### 2.2 Conditional variance — `cov_res_err` / `model_cov_error`

The contrast the paper draws (Cor. 5.2 vs 5.4) is the heart of this week's work:

| Loss / tilting | Model `Cov_ν[v\|u]` | Behaviour vs `C_{v\|u}` |
|----------------|----------------------|--------------------------|
| cosine | `C_vv` (Cor. 5.2) | stuck at the prior — always too large; never shrinks |
| L² | `(C + C_vv⁻¹)⁻¹` (Cor. 5.4) | can shrink to and match the truth |
| MSE | `None` | point predictor — no covariance to read off |

`C_{v|u} = C_vv − C_uvᵀ C_uu⁻¹ C_uv` is the irreducible data conditional
covariance. Two estimators:
- `model_cov_error` — analytic: model's *claimed* covariance vs `C_{v|u}`.
- `residual_cov_error` (`cov_res_err`) — **sampling-based**: covariance of the
  residuals `v − E_ν[v|u]` vs `C_{v|u}`. This is the sampling/Monte-Carlo route
  and is the only covariance metric defined for the MSE model.

### 2.3 Theory error

`theory_match_error` = `‖A_learned − A*‖_F / ‖A*‖_F`, with `A*` selected by
`resolve_theory_target`:

| Loss | `A*` reference | Paper |
|------|----------------|-------|
| cosine conditional / one-sided cosine | `C_uu⁻¹ C_uv C_vv⁻¹` | Thm 5.1 |
| cosine joint | SVD formula with `h(σ)` | Thm 5.6 |
| L² one-sided `v\|u` | `C_uu⁻¹ C_uv C_{v\|u}⁻¹` | Thm 5.3 |
| L² one-sided `u\|v` | `C_{u\|v}⁻¹ C_uv C_vv⁻¹` | Thm 5.3 |
| MSE | `K = C_uu⁻¹ C_uv` (raw `W_uᵀ W_v`) | — |
| two-sided L², L² joint | none (no closed form) | — |

> The draft's theory-error line for L² one-sided is `c_uu⁻¹ c_uv c_{v|u}⁻¹`
> (matches `predicted_A_quadratic_v`); the MSE target is `c_uu⁻¹ c_uv`. Both
> agree with the code.

---

## 3. The loss-variant sweep (`experiments.yaml`)

Seven runs, all at `past_len = future_len = 50` on the GP field defined in the
paper, `u = (0..49)`, `v = (50..99)`:

| Run name | `loss.type` | tilting | λ_u, λ_v | theory target | conditional cov behaviour |
|----------|-------------|---------|----------|---------------|---------------------------|
| `loss_conditional_dot` | conditional | cosine | 0.5, 0.5 | Thm 5.1 | inflated → `C_vv` |
| `loss_conditional_l2`  | conditional | L²     | 0.5, 0.5 | none | matched (claimed) |
| `loss_joint_dot`       | joint       | cosine | —        | Thm 5.6 | inflated → `C_vv` |
| `loss_joint_l2`        | joint       | L²     | —        | none | — |
| `loss_one_sided_v_dot` | one_sided_v | cosine | 0.0, 1.0 | Thm 5.1 | inflated → `C_vv` |
| `loss_one_sided_v_l2`  | one_sided_v | L²     | 0.0, 1.0 | Thm 5.3 (`v\|u`) | matched → `C_{v\|u}` |
| `loss_mse`             | mse         | —      | —        | `K` | residual-only |

Each run logs over training: **learning loss**, **theory error**,
**forecast MSE** (conditional mean), **residual/model cov error** (conditional
variance), and the **self/random similarity** ratio; plus a post-training
fixed-batch retrieval eval (`recall@k`, `margin`).

### Expected qualitative outcomes (the report's per-loss subsections)

- **CLIP two-way conditional (cosine):** matches conditional and marginal means;
  inflates conditional and marginal variances (claims `C_vv`, so
  `model_cov_error` sits at a nonzero floor while `forecast_mse → 0`).
- **One-sided cosine conditional:** same mean recovery as two-way (Thm 5.1 holds
  for any λ under cosine), same variance inflation.
- **One-sided L²:** matches both the conditional mean (Thm 5.3) **and** the
  conditional covariance `C_{v|u}` (Cor. 5.4) — the case where `model_cov_error`
  should drive toward 0.
- **Joint cosine:** Thm 5.6 optimum; mean is rescaled by `h(σ)`, variance still
  inflated.
- **Joint L²:** no closed form — reported numerically (theory error `None`).
- **MSE:** recovers the regression map `W_uᵀ W_v → K`; competitive forecast MSE
  but no probabilistic covariance (only `residual_cov_error`).

---

## 4. What actually ran vs what's missing

**Completed (Weights & Biases, project `contrastive-ts`):**
- The cosine conditional **baseline** (`config.yaml`, `phase1_gp_conditional_baseline`)
  ran to completion multiple times on 2026-06-05 (e.g. 50k and 100k step runs;
  final `theory_err ≈ 0.32–0.38`). These are the dimensionality-sweep /
  debugging runs referenced in the weekly summary.

**Crashed / incomplete:**
- The 2026-06-14 attempt to run the **loss-variant sweep** via
  `run_experiments.py` stopped at **step 700** of the first run
  (`output.log`: `KeyboardInterrupt` inside `sample_pair → torch.randn`). The
  local W&B summary for that run shows partial metrics only
  (`forecast_mse ≈ 1.2e-6`, `res_cov_error ≈ 0.22`, `model_cov_error ≈ 0.028`,
  `theory_err = NaN`).
- No `results/<run>/model.pt|history.pt|eval.json` artifacts exist for the seven
  sweep variants — only `results/phase1_gp_conditional_baseline/config.yaml`.

So the **engineering-metric definitions are implemented and correct**, but the
**numbers/plots that fill the per-loss subsections of the report have not been
generated** for six of the seven loss variants.
</content>
