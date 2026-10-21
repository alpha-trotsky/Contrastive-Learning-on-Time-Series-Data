# Handoff: Goal 3 — a real dataset

Goal 2 is on branch `alpha-trotsky/relaxed-sagan-tkz0me`, draft PR #5 (stacked on
#4). Read first, in order:
1. `docs/goal2_results.md`: what nonlinear encoders achieved and which scores to carry forward.
2. `docs/goal1_results.md`: the linear baseline and why L² beats Gaussian models.
3. `docs/goal1_nonlinear_domain_strategy.md` §1: the scoring metric (energy score etc.) and why it works without an oracle.

## Ground rules from the user
* **Do not change the synthetic field setup** in `test_bed/config.yaml` and do not
  edit `modalities/past_future.py`. Real data gets its own modality class.
* **No attribution lines** in commits or PR text (no Co-Authored-By, no session link,
  no "Generated with" footer).
* Explain results plainly; avoid notation-heavy write-ups.
* Run the full grid of losses when comparing: {cosine, L²} × {one-sided, two-sided,
  joint} + MSE. Goal 2 added normalised cosine and a Gaussian-NLL MLP; keep both.

## Where things stand (one paragraph)
A trained contrastive model is a forecaster: resample real futures with weights
∝ exp(score). With **MLP encoders**, **L² joint** and **normalised cosine** are within
~1% of the perfect forecaster on heavy-tailed, bent and bimodal synthetic series, with
calibrated bands. Unnormalised cosine still breaks on heavy tails. A Gaussian-NLL MLP
(mean + past-dependent width) is strong on single-peaked data but loses on bimodal
futures. The question for goal 3: **does this hold on a real series, where there
is no oracle?**

## Work inside the user's framework: `test_bed/experiment.py`
`python experiment.py --config <config>` is the user's main entry point, and goal 3 should
run through it. As of goal 2 it works for any encoder:
* `model.type: linear_clip` or `mlp_clip` (see `config_mlp.yaml`, which has the same
  field setup as `config.yaml` but MLP encoders and L² joint loss).
* The user's linear-only metrics (theory match, closed-form forecast, similarity,
  retrieval) still run for linear models, unchanged. **Black-box forecast metrics**
  (`evaluation/blackbox.py`) run for every model:
  * energy score and skill vs climatology;
  * CRPS, coverage and width;
  * ESS;
  * excess over the oracle when the data has one.

  They are logged to wandb as `bb/*` every `logging.blackbox_every` steps, and once
  more at the end.
* **Hook for real data:** give the modality a `split(name)` method returning a
  `PairSampler` for `'val'`, `'test'` and `'bank'`. `run()` then evaluates on the
  validation period during training, on the test period at the end, and builds
  forecasts by resampling futures from the bank. Without `split()`, everything comes
  from the modality itself, which is right for synthetic data. This was tested
  with a dummy split modality.
* So the real-data work is mostly a new modality class plus a config. `run()` samples a
  fresh training batch with `modality.sample_pair(batch_size)` every step; for real data
  that means "random windows from the training period", which is cheap.
* **Not in `run()` yet** (add if needed, keep linear behaviour unchanged):
  * **early stopping:** keep the checkpoint with the best validation `bb/energy`;
  * **the MLP regression baselines** (`models/mlp_regressor.py`). They have their own
    `.loss` / `.predict`, see the regressor branch in `analysis/domain_sweep.py`.
    Loss `mse` in `run()` is the user's linear head only, and gives a clear error for
    MLPs.

## Other pieces to reuse (don't rebuild)
* `models/mlp_clip.py` (`MLPCLIP`) and `models/linear_clip.py`. MLP encoders
  standardise inputs from fixed stats, which `run()` sets automatically from a modality sample.
* `models/mlp_regressor.py`: `MeanRegressor` (MSE) and `GaussianRegressor`
  (Gaussian NLL). Each has `.loss(u, v)` and `.predict(u)`. `depth=0` gives the linear version.
* `losses/clip_losses.py`: all losses; `normalize: true` gives normalised cosine.
* `evaluation/predictors.py` (`sir_samples`, `clip_score_fn`, `climatology_samples`,
  `gaussian_samples`) and `evaluation/scoring.py` (`score_forecast`, `add_skills`).
  None of the scoring needs an oracle.
* `analysis/domain_sweep.py`: the goal 1–2 sweep script, with a pre-generated training
  set and the regressor baselines. Use it as the reference for the Gaussian-NLL
  sampling (per-step widths + residual correlation) and `cond_coverage`.
  `summarize_domain_sweep.py` / `run_domain_sweep.sh` show the summary-table and
  4-way-parallel patterns.

## What to build
1. **Data: the Weather dataset, with several modalities.** The user wants
   multiple modalities. Write `modalities/weather.py` to produce (u, v) windows built
   from groups of variables.
   * **Source:** the standard long-term-forecasting "Weather" benchmark (Jena, Max
     Planck roof station, 2020, every 10 min, 52,696 rows, 21 variables). It downloads
     from here (checked, HTTP 200, 7.2 MB):
     `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv`.
     Read it with `encoding='latin-1'`, because the units in the header have non-ASCII
     characters. Fallback: Keras' `jena_climate_2009_2016.csv.zip` on
     storage.googleapis.com (2009–2016, 14 variables, also reachable).
   * **Data cleaning:** `wv`, `max. PAR` and `OT` contain `-9999` sentinels (81 cells).
     Replace them with NaN and interpolate. `wd` (wind direction, degrees) is circular:
     encode it as sin/cos.
   * **Suggested modality groups.** Several columns are near-copies of each other:
     Tpot, Tlog, Tdew, VPmax, VPact, sh and H2OC are close functions of T, humidity and
     pressure. Putting such copies on both sides makes the task trivially easy, so use
     one representative per physical quantity:

     | modality | columns |
     |---|---|
     | temperature | `T (degC)` |
     | moisture | `rh (%)`, `VPdef (mbar)` |
     | pressure | `p (mbar)` |
     | wind | `wv`, `max. wv`, sin/cos(`wd`) |
     | radiation | `SWDR`, `PAR` |
     | rain | `rain (mm)`, `raining (s)` (zero 96% of the time, spiky: a real heavy-tail case) |

   * **Two setups. Do A first, then B if time allows:**
     * **A. Many modalities → one future.** u = past windows of all groups
       (standardised, concatenated). v = future temperature. This is the direct
       real-data version of goals 1–2, and every existing model works on it
       unchanged (just a bigger `dim_u`).
     * **B. Cross-modal, CLIP-style.** u and v are *different* modalities, e.g.
       u = past pressure + wind + moisture, v = future rain, or future temperature
       from everything except temperature. This is closest to what CLIP actually
       does (two different kinds of data, one shared space). The rain target also
       tests goal 2's main claim on real data: unbounded scores should struggle on
       spiky futures, and normalised cosine / L² should not.
   * **Modality ablation (cheap, informative):** retrain with one group dropped from u
     and report the loss in energy skill. This says what each modality contributes to
     the forecast, including its uncertainty, not just its average.
   * **Windows:** 10-min steps, so past 144 (1 day) / future 36 (6 h) is a natural
     start. All groups over 1 day give `dim_u` ≈ 144 × 10 = 1440: fine for the MLP,
     large for the linear model (consider striding the past by 3 to keep it manageable).
   * **Chronological split** (e.g. 70/10/20 train/val/test over the year), with a gap
     of at least past+future steps between splits so windows don't overlap across
     them. Standardise each column with **train statistics only**. The year has
     seasons, so test (Oct–Dec) is a harder, shifted regime: report that, don't hide it.
   * Optional second dataset: ETTh1 (electricity transformer, 7 columns), at
     `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`.
2. **Bank:** forecasts are built by resampling futures, so the bank must contain only
   futures the model could know. Use the training-split futures (all of them, or a
   strided subset). Goal 2 found MLPs can become over-confident, and a bank that
   overlaps training pairs could hide that. Option: hold the last chunk of
   the training period out of training and use it as the bank. Report which
   choice you made.
3. **No oracle, so different reference points:**
   * climatology (random training futures, ignores the past);
   * a seasonal-naive / persistence point forecast;
   * Gaussian-NLL MLP and MSE MLP (+ residual noise).

   Headline = energy **skill vs climatology** plus band coverage (`bb/energy_skill`,
   `bb/cov90` from `run()`; with no oracle the `vs oracle` entries are simply absent). For conditional coverage
   without an oracle, bin by a proxy such as the Gaussian-NLL model's predicted width,
   or by the past window's volatility.
4. **Early stopping on a validation energy score.** Goal 2's main open issue: MLPs
   got over-confident when trained too long (30k steps worse than 15k). `run()`
   already logs validation `bb/energy` every `blackbox_every` steps. Add "keep the
   best checkpoint" on top (a small validation set is enough, e.g.
   `blackbox: {n_test: 200, bank: 5000}`). Apply the same rule to every model.

## Experiments
* Models: full grid {cosine, L²} × {one-sided, two-sided, joint} + normalised cosine
  + MSE + Gaussian NLL, for **both** `linear` and `mlp` encoders, ≥ 2 seeds.
* Report per model:
  * energy skill vs climatology;
  * CRPS by horizon step (a curve);
  * 50/90% coverage and width;
  * conditional coverage (proxy bins);
  * mean error;
  * ESS median / min (collapse check);
  * the step chosen by early stopping.
* Setup A first (all groups → future temperature), then setup B (cross-modal, e.g.
  → future rain), then the modality ablation on the best model.

## Hypotheses to test (from goals 1–2)
* L² joint and normalised cosine (MLP) beat MSE + noise and match or beat the
  Gaussian-NLL MLP on energy, with calibrated bands.
* Unnormalised cosine has the widest or least reliable forecasts. Watch ESS min → 1
  if the series has spikes.
* MLP > linear for the contrastive scores. Linear normalised cosine should be
  poor (it discards the level of the past).
* Contrastive forecasts keep real-data shape (skew, daily cycles) that Gaussian
  baselines miss. Check PIT / coverage by horizon.
* Rain as the future (setup B) is the real-data heavy-tail test. Expect unnormalised
  cosine to collapse (ESS → 1) and the Gaussian models to put mass on negative rain.
  L² / normalised cosine should keep the "mostly zero, sometimes a burst" shape.
* Extra modalities should help most with the *uncertainty* (band width that
  tracks fronts and showers), not only the average. The ablation shows this.

## Practical notes
* CPU only, 4 cores: `torch.set_num_threads(1)` per job, 4 jobs in parallel. A
  synthetic MLP run at 15k steps took ~3–4 min.
* `export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1`. The repo tracks some
  `__pycache__` files, and running Python without the second variable modifies them.
* Long sweeps: launch detached (`setsid nohup bash script.sh &`). Background tool
  calls are capped at 2 hours.
* Don't `pkill -f <pattern>` with a pattern that appears in your own command line.
* Install: `pip install torch --index-url https://download.pytorch.org/whl/cpu` and
  `numpy scipy pyyaml matplotlib wandb pandas`.
* Keep downloaded CSVs out of git (`data/` + `.gitignore`) and download them in the
  loader. Weather is 7.2 MB.
