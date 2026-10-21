# Handoff: Goal 2 — nonlinear encoders

Branch `alpha-trotsky/trusting-dirac-bw7om8`, draft PR #4. Read first:
`docs/goal1_nonlinear_domain_strategy.md` (metric + domains) and
`docs/goal1_results.md` (what linear encoders achieved).

## Ground rules from the user
* **Do not change the field setup** in `test_bed/config.yaml` (GP α=2, τ=3,
  dim_true=1000, contiguous past [0:50] / future [50:100], σ=0.05). It is held
  fixed for the linear case. Do not edit `modalities/past_future.py`.
* **No attribution lines** in commits or PR text (no Co-Authored-By, no session link).
* Explain results plainly; avoid notation-heavy write-ups.
* Run the full grid of losses when comparing: {cosine, L²} × {one-sided, two-sided, joint} + MSE.

## What exists (reuse, don't rebuild)
* `modalities/nonlinear.py`: three domains with exact oracles (`oracle_samples`,
  `oracle_spread`): Student-t (`df`), warped GP (`strength`), double-well (`lam`).
* `evaluation/scoring.py`: energy score, CRPS, variogram, band coverage/width.
  `evaluation/predictors.py`: `sir_samples` turns ANY score function into
  forecasts by reweighting a bank of training futures. It works unchanged for
  nonlinear encoders: it only calls `model.encode_u/encode_v` via `clip_score_fn`.
* `analysis/domain_sweep.py` (one run = domain, dial, loss, seed), `run_domain_sweep.sh`
  (xargs -P 4; set `MODELS=...`), `summarize_domain_sweep.py` (markdown tables).
  Results JSON in `test_bed/results/domain_sweep/`.

## What to build
1. **`models/mlp_clip.py`**: same interface as `LinearCLIP` (`encode_u`, `encode_v`,
   `forward -> (gu, gv, logit_scale.exp())`, `u_encoder`/`v_encoder` attrs if
   possible). Small MLPs (e.g. 2 hidden layers, width 128–256, GELU), with an
   optional linear skip connection so the model contains the linear one.
   Register as `type: mlp_clip` in `experiment.build_model`.
2. **`--encoder {linear,mlp}` flag** in `domain_sweep.py`; include it in the output tag.
3. **MSE baseline with an MLP.** The current MSE path uses `W_u^T W_v`, which is
   linear-only. Replace it with an MLP regressor for the mean, plus a fair
   nonlinear probabilistic baseline: an MLP that predicts mean AND per-step log-variance
   (Gaussian NLL). That is the "best nonlinear Gaussian model" that L² must beat.
4. **Normalised cosine critic** (real CLIP: L2-normalise embeddings before the dot
   product). Add it as a loss option (`normalize: true`) and to `clip_score_fn`.
   It is expected to avoid the heavy-tail collapse of the unnormalised version.

## Experiments
Same 3 domains × 3 dials (`t: inf,10,3`; `warp: 0,0.5,1`; `dwell: 0,1,1.5`),
2 seeds, MLP encoders, losses: full grid + normalised cosine + MSE-MLP +
heteroscedastic-Gaussian-MLP. Compare against the linear numbers already saved.

## What success looks like (hypotheses to test)
* **Heavy tails:** the hard-case 90% band coverage (linear L² ≈ 0.76 at df=3,
  should be 0.90) moves toward 0.90. The band width can now depend on the past.
* **Bent GP:** the encoder can learn to "un-bend" the data (the optimal encoder is
  the inverse warp). Check by regressing the true latent GP values on the learned
  u-features: near-perfect linear fit = recovered.
* **L² stays ≤ the heteroscedastic Gaussian MLP** on the double-well (bimodal
  futures; a Gaussian cannot represent them).
* **Cosine (unnormalised):** does a nonlinear encoder fix or worsen the over-wide
  bands and the heavy-tail collapse? Normalised cosine is the practical comparison.
* Report energy vs oracle, overall + easy/mid/hard band coverage, mean error, and
  SIR effective sample size (watch for collapse; ESS min → 1).

## Practical notes
* CPU only, 4 cores: `torch.set_num_threads(1)` per job, 4 jobs in parallel.
  A linear run takes ~1–3 min. MLPs will be slower, so budget steps (15k at batch
  256 was enough for linear; check MLP convergence on one config first).
* Double-well oracle (particle filter) is the slowest part of evaluation.
* Don't `pkill -f <pattern>` with a pattern that appears in your own command line;
  it kills the calling shell.
* `wandb`: set `WANDB_MODE=disabled`.
* Install deps: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  and `numpy scipy pyyaml matplotlib wandb`.

## After goal 2 (goal 3)
Real dataset (issue #1 suggested ETT / Weather). No oracle there: use energy
score, CRPS by horizon, and band coverage against realised futures. SIR banks =
training-split futures.
