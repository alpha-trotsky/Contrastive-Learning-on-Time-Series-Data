# Goal 1 results — linear models on non-Gaussian domains

Setup: the existing linear models, i.e. cosine (inner product) and L² scores, each
trained one-sided, two-sided and joint, plus MSE regression + Gaussian residual
noise. Each was trained 15k steps on each domain at three dial settings, 2 seeds
each (126 runs).
Field setup unchanged from `config.yaml`. Contrastive models are scored only
through their own score (resampling a bank of 50k real training futures).
Full table: `test_bed/results/domain_sweep/summary.md`.
Scripts: `analysis/domain_sweep.py`, `run_domain_sweep.sh`, `summarize_domain_sweep.py`.

How to read the columns:
* **energy vs oracle**: how much worse than the perfect forecaster (0% = perfect).
* **band90**: how often the 90% band catches the truth (target 0.90), split into
  easy / mid / hard cases by how uncertain the perfect forecaster is.

## Full grid: {cosine, L²} × {one-sided, two-sided, joint} + MSE

"One-sided" = past → future only (the forecasting direction). "Two-sided" =
average of past → future and future → past (standard CLIP). "Joint" = one softmax
over every past–future pair in the batch. Energy vs the perfect forecaster, mean
of 2 seeds (0% = perfect):

| domain | dial | cos 1-sided | cos 2-sided | cos joint | L² 1-sided | L² 2-sided | L² joint | MSE + noise |
|---|---|---|---|---|---|---|---|---|
| Heavy tails | ∞ (GP) | +5.6% | +5.6% | +9.8% | +0.0% | +0.0% | +0.0% | +0.1% |
| | 10 | +12.9% | +12.9% | +19.5% | +1.2% | +1.3% | +1.2% | +1.6% |
| | 3 | ~9× worse | ~10× worse | +69.5% | +4.6% | +4.5% | +4.5% | +8.6% |
| Bent GP | 0 (GP) | +5.5% | +5.6% | +12.5% | −0.0% | −0.0% | +0.0% | +0.0% |
| | 0.5 | +16.2% | +17.1% | +23.1% | +0.2% | +0.3% | +0.3% | +1.7% |
| | 1 | +37.4% | +40.0% | +25.2% | +1.6% | +1.5% | +1.6% | +12.7% |
| Double-well | 0 (OU) | +0.3% | +0.4% | +1.5% | +0.1% | −0.1% | +0.1% | −0.1% |
| | 1 | +1.5% | +1.5% | +3.3% | +0.7% | +0.5% | +0.6% | +10.9% |
| | 1.5 | +1.3% | +2.6% | +2.7% | +0.4% | +0.1% | +0.3% | +14.1% |

90% band coverage on the hardest third of cases (target 0.90):

| domain | dial | cos 1-sided | cos 2-sided | cos joint | L² 1-sided | L² 2-sided | L² joint | MSE + noise |
|---|---|---|---|---|---|---|---|---|
| Heavy tails | ∞ | 0.98 | 0.98 | 0.97 | 0.89 | 0.89 | 0.89 | 0.89 |
| | 10 | 0.95 | 0.95 | 0.92 | 0.80 | 0.80 | 0.81 | 0.81 |
| | 3 | 0.65 | 0.64 | 0.78 | 0.76 | 0.76 | 0.77 | 0.81 |
| Bent GP | 1 | 0.93 | 0.92 | 0.93 | 0.89 | 0.89 | 0.89 | 0.80 |
| Double-well | 1.5 | 0.91 | 0.91 | 0.92 | 0.89 | 0.89 | 0.90 | 0.91 |

(Full tables, including overall coverage and band widths: `test_bed/results/domain_sweep/summary.md`.)

**The score type decides everything; how the softmax is set up barely matters.**
* One-sided vs two-sided makes no difference for either score. That matches the
  theory for cosine (same target for any weighting) and your earlier GP runs for L².
* Joint L² is as good as conditional L². Joint cosine is worse on mild data (the
  known joint shrinkage, +10–12% on the GP vs +5.6%). On heavy tails, though, it
  is much *less* catastrophic (+70% vs ~10×): the shrinkage weakens the pull
  towards extreme futures that breaks conditional cosine.
* Every L² variant sits within 0–5% of perfect on every domain. Every cosine
  variant over-covers on mild data and degrades with non-Gaussianity.

## What it means

**1. L² holds up off-Gaussian; it is within 0–5% of perfect everywhere.**
The paper's result that L² gets both the average and the uncertainty right
survives all three kinds of non-Gaussianity, with linear encoders. Its bands
stay at ~0.89–0.91 overall.

**2. L² beats the "best possible Gaussian model", and MSE does not.**
The domain check showed that any model that assumes Gaussian noise around a
linear prediction loses 11% (bent GP) and 12–15% (double-well) to the perfect
forecaster. MSE + noise lands right there (+12.7%, +14.1%). L² does not
(+1.6%, +0.4%), even though its encoders are just as linear. The reason: a
contrastive model never invents futures. It *reweights real ones*, so the
shape of its forecast (skew, two separate outcomes) comes from the data, and
the linear encoders only have to pick the weights. This is the strongest
argument for the contrastive approach as a probabilistic forecaster on real
data, and it already holds before adding nonlinear encoders.

**3. Cosine CLIP's too-wide bands get worse with non-Gaussianity, and on heavy
tails it breaks completely.**
* On the GP it shows the paper's known flaw: right average, bands ~1.5× too
  wide, 98% coverage instead of 90%.
* Bent GP: the gap grows to +40%. Its average prediction also degrades (2.8× the
  oracle's error), so it is not only the spread.
* Heavy tails df=3: the forecast collapses onto a handful of extreme training
  futures (effective sample size → 1 for some queries). It ends up ~10× worse
  than perfect and far worse than ignoring the past. This matches the prediction
  in the strategy doc. A score that is linear in v (inner product) tilts the
  data towards extremes without limit, and with heavy tails that tilt is not even
  well-defined. The whitened theory error rises with tail weight (0.14 → 0.25 →
  0.48), so the Gaussian theory's answer no longer describes what cosine learns.
* On the double-well, whose values are bounded, cosine is only mildly worse
  (+1.5–2.6%). The failure is specific to unbounded, heavy-tailed data.

**4. The band check catches what a single number hides: uncertainty that should
vary but doesn't.**
On heavy tails, how uncertain the future is depends on the past (wild past →
wild future). Linear L² and MSE use one fixed band width, so they are too wide on
easy cases and too narrow on hard ones (L² at df=3: 0.98 / 0.92 / **0.76**),
while their overall coverage still looks fine (0.888). That is the remaining 4.6%,
and it is exactly what nonlinear encoders (goal 2) should be able to fix. On the
bent GP and double-well, L²'s bands stay even across easy/hard; MSE's do not.

## Caveats

* 2 seeds, 500 test cases each. Differences under ~1% are within noise.
* Inner-product "cosine" here is the paper's version. Real CLIP normalises
  embeddings, which bounds the score and would likely prevent the heavy-tail
  collapse. That is worth one extra run before drawing conclusions about CLIP
  as used in practice.
* The bent-GP effect is muted by the fixed field setup (noise is larger than the
  field), so its gaps understate what a cleaner signal would show.

## For goal 2

* Carry **L²** forward as the main score. One-sided, two-sided or joint is a free
  choice; one-sided past → future is the simplest for forecasting. Keep cosine as
  the comparison point the paper predicts will fail, and add normalised cosine.
* The target for nonlinear encoders is clear: fix the easy/hard band imbalance on
  heavy tails, and close the last 1–5%.
