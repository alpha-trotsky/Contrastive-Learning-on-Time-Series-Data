# Goal 1 — Non-linearity in the domain: strategy, metric, and issue #3 follow-ups

Roadmap: (1) non-linear **domain**, linear encoders → (2) non-linear **encoders** →
(3) a **real** dataset. This doc covers the close-out of issue #3, the new
evaluation metric (which all three goals use), and the plan for goal 1.

**Constraint:** the linear test bed's field setup (`config.yaml`: GP α=2, τ=3,
`dim_true`=1000, contiguous past `[0:50]` / future `[50:100]`, σ_u=σ_v=0.05) is
held fixed. Every non-Gaussian domain below has a strength knob whose zero point
is that exact setup (the double-well tier's zero is an OU process instead).

---

## 0. Close-out of issue #3 ("order of attack" items 1–3)

Scripts: `test_bed/analysis/issue3_followups.py`, `analysis/batch_sweep.py`
(+ `run_batch_sweep.sh`), logs in `test_bed/logs/`.

### (2) Canonical spectrum: the 50×50 task is effectively rank-1

| | value |
|---|---|
| canonical correlations σ | **0.980**, 0.029, then 0 to 3 d.p. |
| I(u;v) | **1.63 nats** (log 256 = 5.55, log 512 = 6.24) |
| joint shrink h(σ₁) | 0.613 (≈ 0.63·σ₁) |

The GP's pointwise std is **0.039**, *below* the observation noise σ = 0.05, and
both windows are short compared with the GP length scale. So the past carries
essentially one number about the future (the local level), and everything else
is noise. This confirms the "modest σ" reading of §1 in issue #3, which was really
"one strong σ plus noise". It also matters for interpreting every metric below.
(Not changed, by design; just noted.)

### (1) E_dir: the cosine "theory-error floor" is not a floor

| run | raw ‖A−A*‖/‖A*‖ | E_dir | scale | **whitened error** |
|---|---|---|---|---|
| saved one_sided_v_dot (cosine, 20k) | 0.996 | 0.706 | 1.41 | — |
| saved phase-1 baseline (cosine, embed 20, 100k) | 0.340 | 0.326 | 1.05 | — |
| cosine cond K=64, 40k | 1.050 | 0.77 | 1.35 | 0.067 |
| cosine cond K=256, 30k | 0.705 | 0.62 | 1.12 | 0.051 |
| cosine cond K=1024, 20k | 0.506 | 0.48 | 1.05 | 0.050 |
| L² v\|u K=64, 40k | 0.106 | 0.10 | 1.02 | 0.018 |
| L² v\|u K=1024, 15k | 0.064 | 0.06 | 1.01 | 0.011 |

* The raw error is a **direction** error (E_dir ≈ raw, scale ≈ 1), so it is not the "scale lags, fix s" story.
* New metric `theory_match_whitened_error` = ‖C_uu^½(A−A*)C_vv^½‖/‖C_uu^½A*C_vv^½‖
  measures A in the units the model distribution depends on. In those units
  cosine is within **~5%** of Thm 5.1 after ~10k steps and then flat. The raw
  error keeps sliding (K=256: 1.46 → 1.20 → 0.82 → 0.70 at 5k/10k/20k/30k). The
  raw Frobenius norm is dominated by the noise-floor directions of C_uu, C_vv
  (eigenvalues ≈ σ² = 0.0025), where the loss is nearly flat and Adam drifts
  slowly. That is the "0.62 floor".
* My hypothesis was that the bilinear critic cannot represent the Gaussian PMI's
  v-only quadratic term, so its finite-K optimum should depend on batch size.
  This is **not supported** at this resolution: whitened error is 0.067 / 0.051 / 0.050 for
  K = 64 / 256 / 1024, with no clear trend. Cosine's whitened error stays ~4× L²'s,
  which looks like an SGD-noise floor and not a bias. Worth revisiting
  with lr decay if it matters.

**Recommendation:** report the whitened error as the theory metric from now on.
Keep raw/E_dir as diagnostics only.

### (3) Retrieval: the old eval mis-measured it; the Bayes ceiling is low

512-way u→v, same fixed batch:

| scorer | R@1 | R@5 | R@10 |
|---|---|---|---|
| chance | 0.002 | 0.010 | 0.020 |
| **Bayes-optimal (true Gaussian PMI)** | **0.018** | **0.102** | **0.172** |
| L² model, its own logit | 0.025 | 0.090 | 0.164 |
| L² model, embedding cosine (old eval) | 0.004 | 0.021 | 0.033 |
| MSE, −‖v̂−v_j‖² | 0.016 | 0.074 | 0.150 |
| cosine model, own logit / embedding cosine | 0.004 / 0.002 | 0.018 / 0.025 | 0.039 / 0.045 |

* The issue's guess was right: with I(u;v) = 1.63 nats, near-perfect retrieval is impossible.
* The old eval L2-normalises the embeddings. That is not the score any of these
  losses train, and it under-reports L² by ~5×. Scored with its own logit, the
  L² model **sits on the Bayes ceiling**.
* The cosine model stays near chance even with its own logit. A bilinear score
  uᵀAv ranks candidates without the −½vᵀ(C_{v|u}⁻¹ − C_vv⁻¹)v term of the true
  PMI, so large-norm futures win. This is the retrieval face of Cor 5.2:
  right mean, wrong spread.

---

## 1. Replacing the cosine-similarity metric

### Why `self_sim` doesn't work

1. **Scale-blind.** cos(v̂, v) ignores amplitude, so a forecast 10× too large scores the same.
2. **Wrong object.** `u_features @ W_v` is the score direction, not the model's
   prediction (it is missing the C_vv / (C+C_vv⁻¹)⁻¹ head).
3. **No reference points.** ~0.5 means nothing without a ceiling (oracle) and a floor
   (climatology). L² vs MSE scored 0.548 vs 0.556 even though they are very
   different forecasters.
4. **Point-only.** It is blind to uncertainty, and uncertainty is the paper's main
   object (mean vs covariance).
5. **Not black-box.** It needs encoder internals.

### Replacement: proper scoring rules on predictive samples (`evaluation/scoring.py`)

Every forecaster is reduced to **M samples of v per u** (shape N×M×d) and compared
with the single realised v. Any engine fits this interface:

| engine | samples via (`evaluation/predictors.py`) |
|---|---|
| contrastive model (any encoder) | **SIR**: resample a bank of training futures with weights softmax_b s(u, v_b). This *is* the paper's ν(v\|u) ∝ e^{s} μ_v, so it needs only score calls (no closed form). ESS is reported. |
| point forecaster / true black box | M = 1 (or + residual noise) |
| closed-form Gaussian, oracle, climatology | direct sampling |

Headline and secondary scores:

* **Energy score** (headline). Strictly proper and multivariate. For a point forecast it
  reduces to the Euclidean error, so point and probabilistic engines share **one
  scale**. With an oracle, `ES − ES_oracle` = ½ the energy distance to the true
  conditional; without one (real data) the ranking is the same in expectation.
  Reported as skill 1 − ES/ES_climatology.
* **CRPS** per coordinate, giving per-horizon curves.
* **Variogram score**, for dependence across horizons (where ES is weak).
* **Calibration**: central-interval coverage at 50/80/90/95% plus width
  (sharpness), PIT histograms, and *conditional* coverage binned by the oracle's
  spread. Conditional coverage is needed for heteroscedastic domains, because
  climatology is marginally calibrated by construction.
* **Mean MSE** against the realised v (the old `forecast_mse`, made black-box).
* Retrieval scored with the model's **own** logit vs the Bayes ceiling, as a critic diagnostic.

### Validation on the Gaussian test bed (`analysis/validate_scoring.py`)

N=1000 test pairs, M=500, bank 50k:

| forecaster | energy | excess over oracle | mean MSE | cov50 | cov90 | width90 | old self-sim |
|---|---|---|---|---|---|---|---|
| oracle N(uK, C_{v\|u}) | 0.2524 | 0 | 0.0026 | 0.501 | 0.899 | 0.166 | — |
| L² v\|u, closed form | 0.2525 | 0.0001 | 0.0026 | 0.499 | 0.897 | 0.166 | 0.547 |
| **L² v\|u, black-box SIR** | **0.2526** | 0.0002 | 0.0026 | 0.501 | 0.897 | 0.166 | |
| cosine cond, closed form | 0.2664 | 0.0140 | 0.0026 | **0.676** | **0.982** | **0.244** | 0.514 |
| **cosine cond, black-box SIR** | **0.2660** | 0.0135 | 0.0026 | 0.673 | 0.982 | 0.243 | |
| MSE + Gaussian residuals | 0.2527 | 0.0003 | 0.0026 | 0.501 | 0.898 | 0.166 | 0.556 |
| MSE point forecast | 0.3573 | 0.1049 | 0.0026 | — | — | — | |
| climatology | 0.3595 | 0.1071 | 0.0058 | 0.484 | 0.889 | 0.244 | — |

* SIR reproduces each closed form to within Monte-Carlo error (median ESS
  10.7k–29.8k of a 50k bank), so the black-box adapter is exact here.
* **Cor 5.2 is directly visible:** cosine has the oracle's mean error but
  climatology's interval width, so it over-covers (98% at nominal 90%) and pays
  an energy excess. L² is indistinguishable from the oracle. The old self-sim
  ranks these models 0.514 / 0.547 and says none of this.
* A point forecast is penalised for having no spread: here it scores almost as
  badly as climatology.

---

## 2. Goal 1 plan: three non-Gaussian domains, each with an exact oracle

Design principle: **each knob's zero is a Gaussian process**. Every result is then
a curve "metric vs non-Gaussianity", anchored at the validated Gaussian theory.
Each domain also exposes `oracle_samples(u, M)`, exact draws from the true
p(v|u). That gives excess-over-oracle and conditional calibration
(`modalities/nonlinear.py`).

| tier | domain | knob (0 = Gaussian) | what it isolates | oracle |
|---|---|---|---|---|
| T1 | **Student-t process**: (u,v) = c·(u_g,v_g)/√w | df ∈ {∞,30,10,5,3} | **same covariances, same linear mean** as the GP; heavy tails + heteroscedastic spread. Pure test of the covariance/uncertainty claims. | closed-form conditional t |
| T2 | **Warped GP** (Gaussian copula): u=g(x_u), v=g(x_v), sinh-arcsinh g | strength k ∈ [0,1] | **nonlinear conditional mean**, skew. The PMI equals the latent GP's, so goal 2's encoders have a known target, g⁻¹. | invert g, Gaussian conditional, re-warp |
| T3 | **Double-well SDE** dx = [−x + λκ(x−x³)]dt + s dW | λ (0 = OU) | genuinely nonlinear dynamics: regime switching, **bimodal** futures, i.e. what real series look like | particle filter + forward simulation |

### The theory these domains test

With linear encoders the cosine model is an **exponential tilt** of the marginal,
ν(v|u) ∝ exp(θᵀv) μ_v(v) with θ = Aᵀu. Its conditional mean is ∇κ(θ) and its
conditional covariance is ∇²κ(θ), where κ is the cumulant generating function of μ_v.
"Cosine is stuck at the prior covariance" (Cor 5.2) is the special case where
∇²κ is constant, i.e. Gaussian μ_v. Off-Gaussian this gives sharp predictions:

* **P1 (T1, heavy tails):** E[exp(θᵀv)] = ∞ for a Student-t marginal for every θ ≠ 0.
  The population cosine/inner-product loss is ill-posed, and only the finite batch
  keeps it finite. Predictions: ‖A‖ shrinks as K grows; the SIR predictive
  degenerates onto a few extreme bank samples (ESS collapses as the bank grows);
  calibration gets worse as df ↓. The L² tilt exp(−½‖Gu−Hv‖²) is always
  normalisable and should stay well-posed. Real CLIP's *normalised* cosine is bounded,
  hence also well-posed, so it gets added as a third critic. If confirmed, this is
  a clean, citable reason why embedding normalisation matters, derived from the
  probabilistic view.
* **P2 (T1, heteroscedasticity):** linear-encoder L² has a u-independent
  conditional covariance. It can match only the *average* C_{v|u}: marginally
  calibrated, conditionally mis-calibrated (over-covers small-|u|, under-covers
  large-|u|). The means stay right for every critic, since the t-process has the
  GP's linear conditional mean. The Gaussian-theory A* is unchanged by
  construction, so "does the learned A still match Thm 5.1/5.3?" is a direct
  null-hypothesis test of the second-moment theory.
* **P3 (T2, nonlinear mean):** linear encoders recover at best the best *linear*
  predictor. The excess energy over the oracle should track the measured Gaussian
  gap below (≈ 0 at k=0, growing with k). Which critic loses least is open.
  This becomes goal 2's baseline, where a nonlinear encoder should close the gap
  because the optimal encoder is known (g⁻¹ then the Gaussian A*).
* **P4 (T3, bimodality):** any Gaussian-family or point predictor puts mass between
  the wells. The SIR predictive of even a *linear* contrastive model is a
  reweighting of real futures, so it can in principle be bimodal: it inherits the
  shape of μ_v. Prediction: contrastive-SIR beats MSE(+Gaussian noise) on energy
  and calibration once λ is large, even with linear encoders. That would be the
  strongest argument for the contrastive approach as a probabilistic forecaster
  on real data.

### Domain sanity check (`analysis/domain_check.py`, original field setup)

Oracle calibration and the **Gaussian gap**: ES(best linear-Gaussian fit on
empirical moments) − ES(oracle), i.e. how much non-Gaussian structure there is to
capture.

| domain / knob | oracle ES | lin-Gauss ES | Gaussian gap (rel.) | oracle cov90 | lin-Gauss cov90 |
|---|---|---|---|---|---|
| t df=inf | 0.2516 | 0.2518 | +0.0001 (+0.1%) | 0.897 | 0.897 |
| t df=30 | 0.2503 | 0.2510 | +0.0007 (+0.3%) | 0.899 | 0.897 |
| t df=10 | 0.2418 | 0.2451 | +0.0033 (+1.4%) | 0.894 | 0.903 |
| t df=5 | 0.2318 | 0.2395 | +0.0077 (+3.3%) | 0.897 | 0.911 |
| t df=3 | 0.1937 | 0.2095 | +0.0158 (+8.2%) | 0.900 | 0.933 |
| warp k=0.0 | 0.2504 | 0.2507 | +0.0003 (+0.1%) | 0.897 | 0.897 |
| warp k=0.25 | 0.2534 | 0.2540 | +0.0006 (+0.3%) | 0.892 | 0.893 |
| warp k=0.5 | 0.2459 | 0.2500 | +0.0041 (+1.7%) | 0.890 | 0.906 |
| warp k=0.75 | 0.2351 | 0.2482 | +0.0131 (+5.6%) | 0.895 | 0.917 |
| warp k=1.0 | 0.2233 | 0.2490 | +0.0257 (+11.5%) | 0.892 | 0.923 |
| dwell lam=0.0 | 2.1352 | 2.1267 | -0.0085 (-0.4%) | 0.887 | 0.888 |
| dwell lam=0.5 | 2.4230 | 2.4836 | +0.0606 (+2.5%) | 0.893 | 0.897 |
| dwell lam=1.0 | 1.6847 | 1.8854 | +0.2007 (+11.9%) | 0.917 | 0.916 |
| dwell lam=1.5 | 1.1201 | 1.2850 | +0.1650 (+14.7%) | 0.896 | 0.945 |

* The oracles are calibrated everywhere (cov90 0.887–0.917, mostly within MC error of
  0.90). The gap is ≈ 0 at every Gaussian end. The double-well λ=0 value of −0.4% is
  Monte-Carlo noise at N=200.
* All three knobs open a gap monotonically (double-well peaks at λ=1). The best
  linear-Gaussian predictor also becomes mis-calibrated as the gap opens. That is
  the room goal 1 measures and goal 2 must close.
* **The fixed field setup mutes the warp tier.** At full strength the gap is
  11%, vs 128% in an earlier test on a strided, low-noise layout. The observation
  noise (σ=0.05 > field std 0.039) lives in the latent and dominates what the warp
  acts on. The double-well tier (+12–15%) is not tied to the GP field, so it is
  the strongest nonlinear signal available under the fixed setup.

### Runs (per tier × knob value)

Critics: inner-product (paper's "cosine"), normalised cosine (CLIP), L² v|u
one-sided, joint inner-product. Baselines: MSE (+Gaussian residuals), oracle,
climatology, best linear-Gaussian. Linear encoders, `embed_dim`=100,
`config.yaml` training settings, 3 seeds.

Logged per run: energy (+skill, +excess), CRPS by horizon, variogram, mean MSE,
coverage 50–95%, conditional coverage by oracle-spread decile, SIR ESS vs bank
size, ‖A‖ and whitened Gaussian-theory error (T1), and retrieval at own logit vs
the oracle-PMI Bayes ceiling.

Roughly 3 tiers × 4–5 knob values × 4 critics × 3 seeds ≈ 150 linear runs.
That fits on CPU at 20k steps if the sampler is vectorised.

### How the results feed goals 2 and 3

* T1 decides **which critic** to carry forward (expected: L² or normalised
  cosine; inner-product only if P1 is refuted).
* T2 gives goal 2 a domain with a **known optimal nonlinear encoder**. The gap
  measured here is exactly what goal 2 must close, and we can check whether
  learned encoders approximate g⁻¹.
* T3 + the black-box energy score are the bridge to goal 3. On real data (e.g.
  ETT / Weather, as planned in issue #1) there is no oracle, but energy, CRPS,
  variogram, coverage and PIT need only the realised future. We already know how
  they behave against an oracle.
