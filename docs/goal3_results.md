# Goal 3 results — a real dataset (Jena Weather 2020)

Goals 1–2 asked whether a contrastive model, used as a forecaster, can match the
perfect forecaster on synthetic series. Goal 3 asks the same on a real series,
where there is no perfect forecaster to compare against. Everything below follows
`docs/goal3_handoff.md`. Every run went through `python experiment.py --config …`.

Full tables:
* `test_bed/results/weather/summary_A.md`, `summary_B.md`, `summary_Arel.md`, `summary_A+trainbank.md`;
* `ablation_A_l2_joint_linear.md`, `ablation_A_gauss_linear.md`.

The per-run JSON is in `test_bed/results/weather/<run>/results.json`.

## Setup

* **Data:** Jena roof station, 2020, one row every 10 minutes (52,696 rows). It is
  downloaded on first use into `data/` (not tracked). The 81 `-9999` cells are
  interpolated, and wind direction is encoded as sin/cos.
* **Modalities:** six groups, one representative per physical quantity:

  | group | columns |
  |---|---|
  | temperature | T |
  | moisture | rh, VPdef |
  | pressure | p |
  | wind | wv, max. wv, sin/cos wd |
  | radiation | SWDR, PAR |
  | rain | rain, raining |

  The near-copies (Tpot, Tdew, VPmax, …) are left out.
* **Windows:** past = the last day at 30-minute resolution (48 values per
  column); future = the next 6 h at 10-minute resolution (36 values).
* **Split:** chronological.

  | period | dates | windows |
  |---|---|---|
  | train | Jan 1 – Sep 13 | 24,405 |
  | validation | Sep 14 – Oct 19 | 4,910 |
  | test | Oct 21 – Dec 31 | 10,181 |

  A gap of one window length separates every pair of periods, so no two windows
  from different periods share a time step. All columns are standardised with
  training statistics only.
* **Bank (the futures a forecast resamples from):** every 4th week of the training
  period is held out of training: 5,841 windows from late January to early
  September. Climatology and the regression baselines' noise use the same bank.
  The alternative, the training futures themselves, was checked separately (below).
* **Test:** 1,000 random test windows. All numbers are on the test period.
* **Early stopping:** the validation energy score (300 validation windows) is
  checked every 250 steps, and the best checkpoint is kept. The same rule applies to
  every model.
* **Training:** batch 256.
  * MLPs: lr 3e-4, 10k steps (2 hidden layers of 256 + linear shortcut, as in goal 2).
  * Linear models: lr 1e-3, 15k steps.

  Both were chosen on the validation period only. With lr 1e-3, MLPs over-fit even
  faster (best validation energy 0.907 vs 0.859 for L² joint).
* **Models:** the full grid {cosine, L²} × {one-sided, two-sided, joint}, plus
  normalised cosine, MSE + noise and Gaussian NLL. Each is run with linear and MLP
  networks, 2 seeds. That is 36 runs per setup and 132 runs in all.

**Reading the numbers.** "Skill vs persistence" = how much lower the energy score
is than "the last observed value, repeated" (0 = no better, higher is better). This
is the headline. Skill vs climatology is in the full tables, but it flatters
everything here: climatology draws futures from January–September, and the
test period is October–December, so even persistence has 0.70 skill vs climatology.
"cov90" = how often the 90% band catches the truth (target 0.90).

Seeds agree closely: the two seeds of every main model differ by ≤ 0.04 skill. A
difference of a few points between models is real; 1 point is not.

## Setup A: all groups → future temperature

| model | linear: skill vs persist | linear: cov90 | MLP: skill vs persist | MLP: cov90 | MLP: stop step |
|---|---|---|---|---|---|
| cos 1-sided | −0.52 | 1.00 | +0.06 | 0.61 | 500 |
| cos 2-sided | −0.56 | 1.00 | +0.14 | 0.66 | 1,875 |
| cos joint | −0.41 | 1.00 | +0.15 | 0.83 | 9,375 |
| norm. cos | −0.30 | 0.92 | +0.07 | 0.77 | 250 |
| L² 1-sided | +0.35 | 0.85 | +0.15 | 0.72 | 250 |
| L² 2-sided | +0.34 | 0.85 | +0.24 | 0.70 | 750 |
| **L² joint** | **+0.38** | 0.88 | **+0.28** | 0.80 | 2,125 |
| MSE + noise | +0.44 | 0.94 | +0.25 | 0.93 | 4,625 |
| **Gauss NLL** | **+0.46** | 0.90 | +0.21 | 0.71 | 250 |

CRPS of the best models, in °C:

| model | 10 min | 1 h | 3 h | 6 h |
|---|---|---|---|---|
| persistence | 0.12 | 0.53 | 1.29 | 2.14 |
| linear L² joint | 0.16 | 0.38 | 0.83 | 1.22 |
| linear Gauss NLL | 0.11 | 0.34 | 0.71 | 1.04 |

## Setup B: pressure + wind + moisture → future rain (cross-modal)

Rain 6 h ahead is barely predictable from these inputs. Climatology already beats
persistence here (+0.17), so the comparison that matters is against climatology.
Rain is zero most of the time, so 90% coverage is high for everything and says
little. The informative numbers are below: negative-rain mass, the Brier score for
"any rain in the next 6 h" (observed in 25% of test windows), and the effective
number of bank futures (ESS).

| model | enc | skill vs clim | samples below 0 mm | Brier (any rain) | ESS min |
|---|---|---|---|---|---|
| L² joint | linear | **+0.012** | 0 | **0.183** | 2,353 |
| L² 2-sided | linear | +0.011 | 0 | **0.183** | 517 |
| norm. cos | linear | −0.004 | 0 | **0.185** | 416 |
| L² joint | MLP | −0.043 | 0 | 0.193 | 2.0 |
| norm. cos | MLP | −0.057 | 0 | 0.210 | 282 |
| cos 2-sided | MLP | −1.04 | 0 | 0.346 | **1.0** |
| cos 1-sided | MLP | −1.39 | 0 | 0.422 | **1.0** |
| MSE + noise | MLP / linear | −1.44 / −1.02 | **0.42 / 0.37** | 0.753 | — |
| Gauss NLL | MLP / linear | −0.67 / −1.16 | **0.42 / 0.47** | 0.753 | — |
| *climatology* | | 0 | 0 | 0.187 | |
| *persistence* | | −0.20 | 0 | 0.200 | |

## Hypotheses from goals 1–2

**1. L² joint and normalised cosine (MLP) beat MSE + noise and match the Gaussian-NLL MLP, with calibrated bands.**
*Partly.*
* Among MLPs, L² joint has the best energy (+0.28 vs +0.25 MSE, +0.21 Gauss NLL).
  But no MLP is calibrated; they are over-confident. Normalised cosine (MLP) is poor
  on temperature (+0.07).
* With linear networks, the Gaussian NLL wins on the absolute target (+0.46 vs
  +0.38 for L² joint).
* That changes once the target is the *change* from the last observation (see
  "Two things that matter on real data" below). Then linear L² joint is the best model
  in the study (+0.48), ahead of the Gaussian NLL (+0.47) and MSE (+0.46).

**2. Unnormalised cosine has the widest or least reliable forecasts; ESS min → 1 on spikes.**
*Yes, both ways.*
* With linear networks on temperature, its bands are hugely too wide: 90% width
  20–23 °C against 4 °C for L², coverage 1.00, and worse than persistence. This is the
  goal-1 flaw (right average, wrong spread) at full size.
* On rain, the MLP versions collapse: ESS min 1.0, so some forecasts are a single
  bank future, and skill is −1.0 to −1.4 vs climatology. Joint cosine (MLP) is the
  exception (−0.06).

**3. MLP > linear for contrastive scores; linear normalised cosine is poor.**
*The first part is reversed on this data; the second holds.*
* Every MLP is worse than its linear version, on temperature and on rain. The MLPs
  over-fit within a few hundred to a few thousand steps:
  * the validation energy bottoms out early (stop steps 250–2,250 for most);
  * after that the validation 90% coverage keeps falling, e.g. from 0.84 to 0.29
    for L² joint over 8k steps;
  * the effective number of bank futures shrinks towards 1.

  Early stopping catches the best point, but even that point is over-confident (test
  cov90 0.6–0.8). The 24k training windows overlap heavily (a new window every
  10 minutes), so there are only about 190 days of distinct weather to learn from. That is
  little for a 600-input MLP and plenty for a linear map.
* Linear normalised cosine is poor on temperature (−0.30), as predicted: normalising a
  linear embedding throws away the current level.

**4. Contrastive forecasts keep real-data shape that Gaussian baselines miss.**
*Yes on rain; on temperature the Gaussian shape is already adequate.*
* On rain, the contrastive forecasts never produce negative rain, and their PIT
  histograms are close to flat:
  * L² and normalised cosine: 0.20–0.21 of values in the outer bins (ideal 0.20),
    half in the lower half;
  * Gaussian models: 0.03–0.10 in the outer bins and 0.64–0.85 in the lower half.
    They put 37–47% of their samples below zero rain and give "rain in the next 6 h" a
    probability near 1 for every window (Brier 0.753).
* On temperature, the PIT of linear L² joint is flat (0.20 / 0.50) and so is the
  Gaussian NLL's (0.17 / 0.55). The 6-hour temperature future is close enough to
  Gaussian that shape is not where the models differ.

**5. Rain as the heavy-tail test.** *All three predictions hold:*
* unnormalised cosine collapses (ESS → 1);
* the Gaussian models put mass on negative rain;
* L² and normalised cosine keep "mostly dry, sometimes a burst".

But nobody forecasts rain well from pressure, wind and moisture at this
resolution: the best models beat climatology by ~1%.

**6. Extra modalities help most with the uncertainty.** *Modestly.* We retrained
linear L² joint and linear Gauss NLL with one group dropped
(`ablation_A_*.md`):

| dropped | L² joint: Δ skill vs clim | L² joint: 90% width (°C) | Gauss NLL: Δ skill | Gauss NLL: width |
|---|---|---|---|---|
| none | — | 4.25 | — | 4.26 |
| temperature | −0.456 | 9.53 (cov90 0.54) | −0.423 | 13.3 (cov90 0.82) |
| radiation | −0.013 | 4.57 | −0.014 | 4.71 |
| moisture | −0.010 | 4.50 | −0.005 | 4.60 |
| pressure | −0.009 | 4.28 | −0.002 | 4.35 |
| wind | −0.004 | 4.31 | +0.000 | 4.37 |
| rain | −0.004 | 4.31 | −0.001 | 4.18 |

* **The past of the target itself is almost everything.** The "temperature" row is
  the cross-modal "everything except temperature → future temperature" setup (B2 in
  the handoff).
  * The contrastive model then becomes badly over-confident (cov90 0.54): from
    radiation, moisture and wind it cannot tell what the current temperature is, and
    resampled futures do not carry a level it does not know.
  * The Gaussian NLL widens its bands to 13 °C instead and stays closer to calibrated.
* **Among the other groups, radiation and moisture help most.** Without them the
  bands are 5–8% wider, and the 6-hour CRPS is 0.07–0.11 °C worse. That is the
  "uncertainty" effect the hypothesis expected, but the size is small: each costs
  about 1 point of skill.

## Two things that matter on real data (not in the handoff's plan)

**The bank limits the contrastive forecast at short range.**
* A contrastive forecast can only output real futures from the bank. At 10 minutes
  ahead, linear L² joint has a CRPS of 0.16 °C, worse than persistence (0.12). Its
  90% band covers the truth 99% of the time at that step, against 86% at 6 h.
* The reason: with 5.8k bank futures, few start at exactly the current temperature,
  so the first steps of the forecast are too spread out. Two checks confirm this is
  the bottleneck.
* **Bigger bank** (`summary_A+trainbank.md`): resample from the 24k training futures
  instead of the 5.8k held-out ones.

  | | held-out bank | training bank |
  |---|---|---|
  | linear L² joint, skill vs persist | +0.38 | **+0.45** |
  | MLP L² joint, skill vs persist | +0.28 | **+0.43** |
  | MLP L² joint, cov90 | 0.80 | 0.83 |

  The goal-2 worry was that a bank overlapping the training pairs would make MLPs
  over-confident. It did not materialise here: the MLP's coverage did not drop. On
  this data the bank's *size* matters more than its overlap with training. We kept the
  held-out bank as the main setting because it is the honest one.
* **Forecast the change, not the level** (`summary_Arel.md`, setup "Arel"): the
  target is the next 6 h of temperature minus the last observed temperature. That
  value is known at forecast time, so this is the same forecast problem, and the
  energy score and CRPS are exactly those of (last value + forecast) against the
  real future. Bank futures now transfer between cold and warm days.

  | model | absolute target, skill vs persist | change target, skill vs persist | change target: cov90 | change target: CRPS 10 min / 6 h (°C) |
  |---|---|---|---|---|
  | **L² joint, linear** | +0.38 | **+0.48** | 0.92 | **0.08** / **1.02** |
  | Gauss NLL, linear | +0.46 | +0.47 | 0.91 | 0.08 / 1.04 |
  | MSE + noise, linear | +0.44 | +0.46 | 0.94 | 0.09 / 1.09 |
  | L² joint, MLP | +0.28 | +0.46 | 0.86 | 0.08 / 1.09 |
  | norm. cos, MLP | +0.07 | +0.38 | 0.79 | 0.08 / 1.30 |
  | cos 2-sided, linear | −0.56 | +0.21 | 0.99 | 0.09 / 1.74 |

  * The contrastive models gain far more from this than the Gaussian baselines,
    which have no bank to be limited by.
  * Linear L² joint becomes the best forecaster in the study, and calibrated: PIT
    outer bins 0.15, slightly wide.
  * The MLP L² joint nearly catches up with the linear models, though MLPs are still
    the over-confident ones (cov90 0.79–0.86).
  * Skill vs climatology is not comparable between the two targets, because
    "climatology" for the change target is "persistence + a random change", a much
    stronger reference.

## What it means for the research question

1. **The probabilistic view works on real data with no oracle.** A trained
   contrastive model, used only through its score, gives a calibrated, competitive
   probabilistic forecast of real weather. With the change target, the best one beats
   the best Gaussian baseline built from the same networks.
2. **The choice of score matters the way the theory says.**
   * L² is the reliable score.
   * Unnormalised cosine gets the spread badly wrong with linear networks and
     collapses on spiky targets with MLPs.
   * Normalisation needs a nonlinear encoder (linear normalised cosine is poor).
   * The shape advantage over Gaussian models shows up exactly where the data are
     non-Gaussian (rain).
3. **On real data, the practical limits are elsewhere:** over-fitting of flexible
   encoders on a short record, and the size and coverage of the bank. Goal 2's "MLP
   ≫ linear" did not carry over. A year of 10-minute data is too little for these
   MLPs without more regularisation.

## Caveats

* One year of data, one test period (late October – December, colder than any of
  the training period except January–March), 2 seeds. The main models are about
  +0.1 to +0.35 °C too warm on average on the test period. That is the seasonal shift
  showing up in all forecasters, not a contrastive-specific bias.
* MLP architecture and regularisation were not tuned (no weight decay or dropout);
  only the learning rate was compared, on validation. A regularised MLP may well
  beat the linear models. That is the obvious next experiment.
* Linear models were still improving slowly at 15k steps (stop steps 7k–15k).
* L² 1-sided (MLP) on rain stopped at step 0 for both seeds: training never improved
  the validation energy. Its row is effectively an untrained model.
* The "NLL width" conditional-coverage proxy uses the Gaussian-NLL MLP, which is
  itself over-confident. The past-volatility proxy tells the same story: every model
  covers calm days better than changeable ones.
* Evaluation fix during this goal: the black-box evaluator used to resample a float32
  copy of the bank. That shifted "exactly dry" futures off the truth and broke rain
  coverage. It now resamples the float64 bank. Temperature results moved by
  0.0003 in coverage. Setup-A runs were kept; setup-B contrastive runs were redone.

## Reproduce

From `test_bed/`, with `WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1`:

```bash
python experiment.py --config config_weather.yaml                    # one run (setup A, MLP L2 joint)
SETUPS="A B Arel" bash analysis/run_weather_sweep.sh                 # grids (4 jobs in parallel)
SETUPS=A ENCODERS=linear MODELS="l2_joint gauss" \
  DROPS="temperature moisture pressure wind radiation rain" bash analysis/run_weather_sweep.sh
BANK=train SETUPS=A MODELS=l2_joint bash analysis/run_weather_sweep.sh
python -m analysis.summarize_weather_sweep --setup A                 # tables
python -m analysis.summarize_weather_sweep --setup A --ablation l2_joint_linear
```

Runtime on 4 CPU cores: about 3–10 minutes per run; the whole study took about 6 hours.
