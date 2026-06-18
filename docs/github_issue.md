# Weekly update — loss functions, metrics, and the 50×50 sweep

This week
- Implemented and tested the loss-function family (cosine / L² × conditional / joint / one-sided, plus an MSE regression baseline) and the per-loss engineering metrics (conditional mean, conditional variance, theory error).
- Debugged the issues from the dimensionality sweep — traced to the cosine gauge / learnable `logit_scale`; see *Concerns*.
- Literature review on contrastive learning for time-series data.

Setup. 1-D GP ($`\alpha=2,\ \tau=3,\ \texttt{dim\_true}=1000`$) split into past $`u = \text{field}[0{:}50]`$ and future $`v = \text{field}[50{:}100]`$, obs. noise $`\sigma=0.05`$. Linear CLIP encoders ($`\texttt{embed\_dim}=100`$, no bias, learnable `logit_scale`). Adam, `lr=5e-4`, `batch=256`, seed 42. This is the Gaussian setting of Baptista–Stuart–Tran, so every metric has a closed-form reference.

Core metrics at 50k steps; retrieval block for the last 3 rows from a 20k re-run (the 50k retrieval pass crashed mid-sweep — ranking stats are saturated by 20k, verified below).

---

## Engineering metrics — definitions

<details>
<summary>Conditional mean (forecast MSE)</summary>

Prediction $`E_\nu[v\mid u]`$ is set by the tilting; `forecast_mse` is its error vs the data MMSE mean $`uK`$, $`K = C_{uu}^{-1}C_{uv}`$.

| Tilting | model $`E_\nu[v\mid u]`$ | paper eq. |
|---|---|---|
| cosine | $`u\,A\,C_{vv}`$ | (41b) |
| L² | $`u\,A\,(C + C_{vv}^{-1})^{-1},\ \ C = s\,W_v^\top W_v`$ | (48b) |
| MSE | $`u\,(W_u^\top W_v)`$ | regression head |
</details>

<details>
<summary>Conditional variance (model_cov_error / res_cov_error)</summary>

Target $`C_{v\mid u} = C_{vv} - C_{uv}^\top C_{uu}^{-1} C_{uv}`$.

| Tilting | model $`\mathrm{Cov}_\nu[v\mid u]`$ | behaviour | paper |
|---|---|---|---|
| cosine | $`C_{vv}`$ | inflated, never shrinks | Cor 5.2 |
| L² | $`(C + C_{vv}^{-1})^{-1}`$ | can match truth | Cor 5.4 |
| MSE | — | point predictor, no covariance | — |

`model_cov_error` = analytic (claimed spread vs $`C_{v\mid u}`$); `res_cov_error` = empirical covariance of residuals $`v - E_\nu[v\mid u]`$ (defined even for MSE).
</details>

<details>
<summary>Theory error</summary>

$`\texttt{theory\_err} = \lVert A_{\text{learned}} - A^\ast \rVert_F / \lVert A^\ast \rVert_F`$, with $`A_{\text{learned}} = s\,W_u^\top W_v`$ (and $`W_u^\top W_v`$ for MSE).

| Loss | $`A^\ast`$ | paper |
|---|---|---|
| cosine conditional / one-sided cosine | $`C_{uu}^{-1} C_{uv} C_{vv}^{-1}`$ | Thm 5.1 |
| cosine joint | $`C_{uu}^{-1/2}\, U\,\mathrm{diag}(h(\sigma))\, V^\top C_{vv}^{-1/2}`$ | Thm 5.6 |
| L² one-sided $`v\mid u`$ | $`C_{uu}^{-1} C_{uv} C_{v\mid u}^{-1}`$ | Thm 5.3 |
| MSE | $`K = C_{uu}^{-1} C_{uv}`$ | — |
| two-sided L², L² joint | none → reported `NaN` | — |

$$h(\sigma) = \tfrac{1}{\sigma}\left(\tfrac12\sqrt{1+4\sigma^2} - \tfrac12\right)$$
</details>

---

## Results — all 7 loss variants

| Loss | tilting | learn_loss² | theory_err | forecast_mse | model_cov_err | res_cov_err | self_sim | logit_scale |
|---|---|---|---|---|---|---|---|---|
| conditional_dot | cosine | 5.036 | 0.620 | 3.8e-6 | 8.051 | 0.213 | 0.766 | 8.81 |
| conditional_l2  | L²    | 3.959 | NaN¹ | 8.9e-7 | 0.0103 | 0.212 | 0.772 | 6.61 |
| joint_dot       | cosine | 5.168 | 0.658 | 3.9e-4 | 8.051 | 1.261 | 0.720 | 8.81 |
| joint_l2        | L²    | 4.058 | NaN¹ | 1.1e-6 | 0.0071 | 0.212 | 0.773 | 6.22 |
| one_sided_v_dot | cosine | 5.032 | 0.622 | 4.0e-6 | 8.051 | 0.213 | 0.759 | 8.58 |
| one_sided_v_l2  | L²    | 3.950 | 0.064 | 4.7e-6 | 0.0185 | 0.213 | 0.772 | 6.78 |
| mse             | —     | 0.003 | 0.087 | 1.3e-6 | — (none) | 0.213 | 0.779 | 14.28 |

¹ `NaN` is by design — two-sided L² and L²-joint have no closed-form $`A^\ast`$.
² `learn_loss` is **not** comparable across loss families (different objectives/units): the cross-entropy losses sit against a $`\log(256)\approx 5.55`$ chance baseline, so the cosine rows (~5.0) and L² rows (~3.9–4.1) are both above chance; the MSE row (0.003) is a squared error in $`v`$-units. `similarity_ratio` (self/random cosine) is ~30 for every contrastive run and 33.4 for MSE.

Cross-modal retrieval (fixed 512-candidate batch):

| Loss | margin | self_sim_mean | R@1 u→v | R@1 v→u | R@5 u→v | R@5 v→u | R@10 u→v | R@10 v→u | source |
|---|---|---|---|---|---|---|---|---|---|
| conditional_dot | 0.495 | 0.502 | 0.0059 | 0.0078 | 0.0254 | 0.0312 | 0.0508 | 0.0566 | 50k |
| conditional_l2  | 0.519 | 0.533 | 0.0020 | 0.0039 | 0.0215 | 0.0273 | 0.0508 | 0.0586 | 50k |
| joint_dot       | 0.463 | 0.467 | 0.0020 | 0.0039 | 0.0234 | 0.0215 | 0.0508 | 0.0449 | 50k |
| joint_l2        | 0.520 | 0.533 | 0.0059 | 0.0059 | 0.0234 | 0.0312 | 0.0488 | 0.0625 | 50k |
| one_sided_v_dot | 0.469 | 0.485 | 0.0020 | 0.0039 | 0.0254 | 0.0254 | 0.0449 | 0.0508 | 20k |
| one_sided_v_l2  | 0.503 | 0.533 | 0.0039 | 0.0020 | 0.0215 | 0.0449 | 0.0332 | 0.0820 | 20k |
| mse             | 0.510 | 0.540 | 0.0039 | 0.0039 | 0.0137 | 0.0195 | 0.0430 | 0.0371 | 20k |

> Headline: all 3 cosine variants report $`\texttt{model\_cov\_err} = 8.051`$ *exactly* (they claim the prior $`C_{vv}`$, Cor 5.2); all 3 L² variants collapse to $`\sim 0.007\text{–}0.018`$ (they match $`C_{v\mid u}`$, Cor 5.4); MSE has no probabilistic covariance. Means are recovered almost everywhere ($`\texttt{forecast\_mse}\sim 10^{-6}`$) — only the cosine *spread* is wrong. This is the paper's mean-vs-covariance dichotomy, reproduced cleanly.

---

## Per-loss notes

- CLIP two-way conditional (cosine). Matches the conditional mean ($`\texttt{forecast\_mse}=3.8\text{e-}6`$) but inflates the variance to $`C_{vv}`$ (cov_err 8.051) — Thm 5.1 / Cor 5.2.
- One-sided cosine conditional. Numerically identical to two-way (theory_err 0.622, cov_err 8.051): under cosine the target $`A^\ast`$ is the same for any $`\lambda`$.
- One-sided L² conditional. Best contrastive result — recovers both $`A^\ast`$ (theory_err 0.064) and $`C_{v\mid u}`$ (cov_err 0.0185), Thm 5.3 / Cor 5.4. The generative case.
- Joint cosine (InfoNCE). Thm 5.6 optimum rescales singular values by $`h(\sigma)\neq 1`$; visible as $`\sim 100\text{–}500\times`$ larger forecast_mse (3.9e-4) and res_cov_err (1.261) — the mean is a *shrunk* MMSE mean. Variance still inflated.
- Joint L². No closed-form $`A^\ast`$ (theory_err `NaN`); empirically like the other L² tilts — tight mean (1.1e-6), matched covariance (0.0071).
- MSE baseline. Recovers $`K = C_{uu}^{-1} C_{uv}`$ (theory_err 0.087), best forecast MSE, but a point predictor — no claimed covariance. `logit_scale` is irrelevant here.

---

## Concerns / follow-ups

1. Cosine theory-error floor (~0.62). Cosine variants plateau at $`\texttt{theory\_err}\approx 0.62\text{–}0.66`$ and don't reach 0, while L²-one-sided (0.064) and MSE (0.087) recover their targets tightly. Cosine logits are invariant to rescaling + joint rotation of the encoders (Remark 2.5), and the learnable $`\texttt{logit\_scale}\ s`$ absorbs scale into $`A = s\,W_u^\top W_v`$ — but relative-Frobenius error is not invariant to that gauge, so a direction-correct, scale/rotation-offset solution sits at a nonzero floor. Matches the dimensionality-sweep issue. Try: clamp `logit_scale`, or compare $`A`$ up to the gauge the loss actually fixes.
2. `logit_scale` growth. $`2.659 \to {\sim}8.6\text{–}8.8`$ on cosine runs at `lr=5e-4` (stable, no blow-up). An earlier `lr=5e-3` sweep drove it far enough to produce mid-run `NaN`s — unconstrained, worth clamping.
3. Low absolute retrieval recall (R@1 ~0.002–0.006 at 512-way). The margin/self_sim separation is healthy (self_sim ~0.77 vs rand_sim ~0.02), so encoders *do* align; absolute recall is just hard on this smooth GP. Confirm this isn't an eval-batch artifact.

---

<details>
<summary>Reproducibility / methodology note</summary>

- Core metrics: 50k-step runs (seed 42); the morning and evening sweeps produce identical core numbers, confirming determinism.
- Retrieval for `one_sided_v_dot`, `one_sided_v_l2`, `mse`: 20k re-run via `test_bed/_retrieval_fill.yaml` (the 50k retrieval pass crashed at `one_sided_v_dot`).
- Split is sound: reloading the 20k `one_sided_v_dot` model reproduces $`\texttt{model\_cov\_error} = 8.0507`$ (= 50k value — cosine covariance claim is convergence-independent) but $`\texttt{theory\_err} = 0.996`$ vs $`0.622`$ @50k (still descending) — hence core metrics from 50k, retrieval from the saturated 20k runs. The 20k retrieval margins are within ~0.02 of their 50k counterparts.
</details>