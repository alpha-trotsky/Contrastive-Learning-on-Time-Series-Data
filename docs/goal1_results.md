# Goal 1 results — linear models on non-Gaussian domains

Setup: the existing linear models (cosine CLIP = two-sided conditional inner
product; L² = one-sided on p(v|u); MSE regression + Gaussian residual noise),
trained 15k steps on each domain at three dial settings, 2 seeds each (54 runs).
Field setup unchanged from `config.yaml`. Contrastive models are scored only
through their own score (resampling a bank of 50k real training futures).
Full table: `test_bed/results/domain_sweep/summary.md`.
Scripts: `analysis/domain_sweep.py`, `run_domain_sweep.sh`, `summarize_domain_sweep.py`.

How to read the columns:
* **energy vs oracle**: how much worse than the perfect forecaster (0% = perfect).
* **band90**: how often the 90% band catches the truth (target 0.90), split into
  easy / mid / hard cases by how uncertain the perfect forecaster is.

## Headline numbers (energy vs perfect forecaster, mean of 2 seeds)

| domain | dial | cosine CLIP | L² | MSE + noise |
|---|---|---|---|---|
| Heavy tails | GP (df=∞) | +5.6% | **+0.0%** | +0.1% |
| | df=10 | +12.9% | **+1.2%** | +1.6% |
| | df=3 | **~10× worse** (energy 1.8–2.6 vs 0.20) | **+4.6%** | +8.6% |
| Bent GP | 0 (GP) | +5.6% | **−0.0%** | +0.0% |
| | 0.5 | +17.1% | **+0.2%** | +1.7% |
| | 1 | +40.0% (no better than ignoring the past) | **+1.6%** | +12.7% |
| Double-well | 0 (OU) | +0.4% | +0.1% | −0.1% |
| | 1 | +1.5% | **+0.7%** | +10.9% |
| | 1.5 | +2.6% | **+0.4%** | +14.1% |

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

* Carry **L²** forward as the main loss. Keep cosine as the comparison point the
  paper predicts will fail, and add normalised cosine.
* The target for nonlinear encoders is clear: fix the easy/hard band imbalance on
  heavy tails, and close the last 1–5%.
