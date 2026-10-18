# Taking the paper to time series: what transfers (goals 1–3)

This is a summary of the three experiment rounds that extend Baptista, Stuart &
Tran, *A Mathematical Perspective on Contrastive Learning*
(`contrastive_learning.md`), from its Gaussian, linear-encoder setting to
forecasting time series. It is meant to be read on its own. The detailed numbers
are in the per-goal write-ups:

* goal 1: `docs/goal1_nonlinear_domain_strategy.md` (the scoring method) and `docs/goal1_results.md`;
* goal 2: `docs/goal2_results.md`;
* goal 3: `docs/goal3_results.md`.

## 1. The question

The paper reads contrastive learning (CLIP and its variants) as **learning a
conditional distribution**. A trained model with score `s(past, future)` defines a
forecast:

> take a collection of real futures and reweight each one by `exp(score)`.

The paper works this out exactly when everything is Gaussian and the encoders
are linear. In that setting it makes sharp predictions about which loss gets what right:

| | the paper's prediction (Gaussian data, linear encoders) |
|---|---|
| **P1** | The standard dot-product score ("cosine", unnormalised) gets the **average** forecast right but the **spread** wrong: its uncertainty is stuck at "knowing nothing" (Cor. 5.2). |
| **P2** | The paper's new **L² score** (negative squared distance between embeddings), trained in the forecasting direction, gets **both** the average and the spread right (Thm 5.3, Cor. 5.4). |
| **P3** | For the dot product, weighting the two directions (past → future vs future → past) does not change the answer. |
| **P4** | The **joint** loss (one softmax over all pairs) shrinks the forecast towards the average, so it is a worse conditional forecaster than the one-way loss, but better at the marginals (Thm 5.6, Cor. 5.7). |
| **P5** | Retrieval picks the most likely candidate under the model (Thm 4.2), so how well it can work is capped by how much the past says about the future. |

We asked how far these carry over once the setting stops being the paper's. We
relaxed one assumption at a time:

1. **Goal 1: non-Gaussian data, still linear encoders.** Three synthetic series
   with a dial whose zero point is the paper's Gaussian case:
   * heavy tails (Student-t; occasional wild values);
   * a bent Gaussian process (skewed futures, nonlinear relation between past and future);
   * a double-well process (the series jumps between two levels, so the future
     can have two separate likely outcomes).

   Each has an exact "perfect forecaster" to compare against.
2. **Goal 2: nonlinear (MLP) encoders**, on the same three series.
3. **Goal 3: real data.** Jena weather station, 2020, 10-minute readings:
   * forecast the next 6 h of temperature from the last day of six variable groups;
   * a cross-modal setup, closer to how CLIP is used: future rain from past
     pressure, wind and moisture.

   There is no perfect forecaster here.

Every round ran the full grid: {dot product, L²} × {one-way, two-way, joint}, plus
normalised cosine (real CLIP) from goal 2 on. The comparison points were
regression models: MSE plus a noise band, and from goal 2 on a Gaussian model with
a past-dependent band width (Gaussian NLL).

## 2. How we scored forecasts (a contribution in itself)

The earlier test bed scored models by comparing internal matrices against the
paper's closed forms, or by a cosine similarity between predicted and true
futures. Neither works once there is no closed form, and the similarity score
ignores uncertainty, which is the paper's main subject.

We replaced both with a **black-box** procedure that works for any model:

* **Turn the model into a forecast** exactly as the paper defines it (its eq. 31):
  keep a "bank" of real futures and resample them with weights `exp(score)`. This
  needs nothing but score calls: no closed form, no knowledge of the encoder.
* **Score the forecast** with the energy score. It is a standard measure for
  probabilistic forecasts: it rewards being close to what happened and being
  honest about uncertainty, and a point forecast is simply a forecast with no
  spread. Alongside it we report band coverage (does the 90% band catch the
  truth 90% of the time?), per-horizon errors, and how many bank futures a forecast
  actually uses (a collapse check).

On the Gaussian test bed this procedure reproduces the paper's closed-form
answers to within sampling error, and P1 shows up directly in it: cosine
has the perfect forecaster's average error but 98% coverage at nominal 90%. On
real data it needs only the observed future. The same yardstick therefore runs
from the paper's setting all the way to the weather station.

## 3. What we found, round by round

### Goal 1: non-Gaussian data, linear encoders

* **The paper's Gaussian results hold almost exactly where they apply.**
  * Linear L² sat on the theoretical optimum.
  * Its retrieval matched the best achievable. The old evaluation had reported
    it about 5× too low, because it scored retrieval with a similarity the model
    was never trained on.
  * Cosine's apparent "gap to theory" was mostly slow drift in directions that are
    pure noise. Measured in the units the forecast depends on, it was within ~5% of
    the theory.
  * Retrieval is low for every model because the past says little about the
    future in this setup (P5).
* **L² stays within 0–5% of the perfect forecaster on all three kinds of
  non-Gaussian data**, with linear encoders. A Gaussian regression model lands at
  +11–15% on the bent and double-well series. The reason is that a contrastive
  forecast never invents futures, it reweights real ones. So skew and two separate
  outcomes come from the data, and the encoder only has to choose the weights.
* **Cosine's too-wide bands (P1) get worse off-Gaussian, and on heavy tails it
  breaks completely.** At the heaviest tails the forecast collapses onto a handful of
  extreme training futures and is ~10× worse than perfect.
* **One-way vs two-way made no difference (P3)**, for either score. Joint cosine
  showed the predicted shrinkage on mild data (P4).
* **What linear encoders cannot do:** let the band width depend on the past. On
  heavy tails every linear model is too wide on calm cases and too narrow on wild
  ones.

### Goal 2: nonlinear encoders

* **The best contrastive models get within ~1% of perfect on every series**, with
  calibrated bands: L² joint and normalised cosine.
* **Cosine's spread flaw mostly disappears.** On the Gaussian series cosine went
  from +5.6% to +1.4–2.7%, and its 90% bands on the hardest cases from 98% to 89% coverage. So P1 is a statement
  about **linear encoders**, not about the dot product.
* **Band widths now follow the past** on heavy tails. A regression model with a
  fixed noise band cannot do this.
* **The encoder learns what matters without being told.** On the bent series, the
  learned features recover the hidden "un-bent" quantity that predicts the future
  almost perfectly (R² ≈ 1), and ignore the rest of the past.
* **The shape advantage is real but conditional.**
  * On the double-well, the best Gaussian MLP (same networks, past-dependent
    width) is at +5–6% while L² is within 1%: a Gaussian puts probability between
    the two wells, where the future never goes.
  * On the bent series, which is skewed but single-peaked, the Gaussian MLP ties
    with L².
* **Unbounded scores still fail on heavy tails.** Unnormalised cosine is the worst
  contrastive model there (+7% to +23%) and still collapses onto single futures for
  some inputs. Normalised cosine does not.
* **Normalisation needs a nonlinear encoder.** Linear-then-normalise keeps only the
  *direction* of the past and loses its *size*, and the size (the current level)
  is most of the information in these series (+12–23%).

### Goal 3: real weather data

* **The probabilistic view works with no perfect forecaster to lean on.** Forecasting
  the next 6 h of temperature as a *change from the last reading*, linear L² joint
  is the best model in the study, and calibrated. It beats the last-value-repeated
  forecast by 48% on energy, just ahead of the Gaussian model built the same way
  (47%) and MSE (46%).
* **Linear cosine shows P1 at full size on real data:** right average, 90% bands
  about 5× too wide (20–23 °C against 4 °C for L²).
* **Rain (spiky, mostly zero) is the real-data heavy-tail test, and it behaves as
  goals 1–2 predicted.**
  * Unnormalised cosine collapses onto single futures.
  * The Gaussian models put ~40% of their samples on *negative* rain and say "it
    will rain" for nearly every window.
  * L² and normalised cosine keep "mostly dry, sometimes a burst", with flat
    calibration histograms.

  But nothing forecasts rain well from these inputs: the best models beat "random
  past weather" by ~1%.
* **Two real-data limits that the synthetic rounds did not show:**
  * **Little data.** The MLPs over-fit within a few hundred to a few thousand steps
    and end up over-confident, even with early stopping. On temperature, the MLP
    version of each strong model (L², MSE, Gaussian NLL) was worse than the linear
    one; only the cosine scores, which are poor with linear encoders, gained. One year of heavily overlapping 10-minute windows is only
    about 190 days of distinct weather.
  * **The bank.** At 10 minutes ahead the contrastive forecast was worse than
    "repeat the last value". With 5.8k bank futures, few start at exactly today's
    temperature, so the first steps are too spread out. Two fixes confirm this is
    the bottleneck:
    * a bigger bank (+0.38 → +0.45 skill);
    * forecasting the change from the last reading (+0.38 → +0.48), which lets
      futures from cold days serve warm days.
* **Modality ablation:** the target's own past is almost everything. Radiation and
  moisture help most of the rest, and mostly by narrowing the bands. Dropping
  temperature's own past made the contrastive forecast *over*-confident rather than
  more cautious (open question 4).

## 4. Scorecard: how the paper transfers

| paper's prediction | Gaussian, linear (paper's setting) | non-Gaussian, linear | nonlinear encoders | real data |
|---|---|---|---|---|
| Model = reweighted real futures is a usable forecaster | ✓ matches closed form | ✓ | ✓ (within ~1% of perfect) | ✓ (best model in study) |
| P2: L² gets average **and** spread right | ✓ exactly | ✓ (0–5% of perfect) | ✓ | ✓ calibrated (linear); MLPs over-confident |
| P1: dot product gets spread wrong | ✓ (bands too wide) | ✓, worse; breaks on heavy tails | ✗ mostly fixed | ✓ linear (5× too wide); collapses on rain with MLPs |
| P3: one-way vs two-way doesn't matter | ✓ | ✓ (for L² too) | ✓ (noisy on heavy tails) | ✓ linear; MLPs vary |
| P4: joint shrinks, worse conditional forecaster | ✓ for cosine | ✓ on mild data; *less* broken on heavy tails | ✗ joint is best or tied | ✗ joint is best |
| P5: retrieval capped by information | ✓ (L² on the ceiling) | — | — | — |

Read across the rows:

* **What transfers cleanly** is the framework (a contrastive model *is* a
  conditional distribution) and the L² result. They survive every relaxation we
  tried. The framework's main practical benefit, a forecast whose shape comes
  from real data, is invisible in the paper's Gaussian setting and becomes the
  main argument for the method on real, non-Gaussian data.
* **What transfers only in its own setting** is the cosine–L² contrast (P1) and
  the joint-loss penalty (P4). Both are properties of what a *linear* encoder can
  represent. A flexible encoder can compensate for the dot product's missing
  term, and then the gap closes.
* **What the paper doesn't address, and which turned out to matter most** off its
  setting:
  * whether the score is bounded (heavy tails);
  * the bank of futures the forecast is built from;
  * finite data and over-confidence;
  * the representation of the target (level vs change).

## 5. Rules of thumb: what to expect where

From the three rounds together, these are the predictions we would now make for a
new problem:

* **Linear encoders:** use L². Expect the dot product to give the right average
  with bands far too wide. Do not normalise: it throws away the level.
* **Flexible encoders, well-behaved data:** the choice of score barely matters.
  L², joint cosine and normalised cosine all end up close to the best possible.
* **Heavy-tailed or spiky targets (rain, bursts, outliers):** use a bounded score,
  i.e. L² or *normalised* cosine. Expect the unnormalised dot product to collapse
  onto a few extreme futures, and expect Gaussian baselines to put mass on
  impossible values.
* **Near-Gaussian targets** (6-hour temperature): expect contrastive and Gaussian
  forecasters to be close. The shape advantage only pays off when the future has
  shape: two outcomes, a floor at zero, strong skew.
* **Sharp, short-range forecasts:** expect the bank to be the bottleneck. Use as
  many futures as possible, and a target that transfers between situations
  (changes rather than levels).
* **Small or highly overlapping data:** expect MLP encoders to become
  over-confident quickly. Early stopping on a held-out probabilistic score is
  necessary, and on our weather data still not enough.

## 6. What matters and what doesn't

**Matters:**

* **The score family, when the encoder is linear.** L² vs dot product is the
  difference between calibrated and 5× too wide.
* **Whether the score is bounded, when the data has heavy tails.** This decides
  collapse vs no collapse, whatever the encoder.
* **The bank:** its size, and whether its futures can stand in for the case at
  hand. Worth 7–18 points of skill on real temperature.
* **Encoder flexibility relative to data size.** Flexibility helped a lot on
  plentiful synthetic data and hurt on one year of weather.
* **How you evaluate.** The old metrics under-reported L² ~5× on retrieval,
  invented a "theory floor" for cosine, and could not see uncertainty at all.

**Doesn't matter (much):**

* **One-way vs two-way training.** It made no consistent difference, as the paper
  predicts for the dot product. Empirically the same holds for L² (MLPs on real
  data vary more between the two, with no consistent winner).
* **Cosine normalisation, in most settings.** On well-behaved data with MLP
  encoders it is neither a help nor a harm. It matters in exactly two places, in
  opposite directions:
  * it **hurts** with linear encoders (loses the level);
  * it **is needed** for the dot product on heavy-tailed data.
* **Whether the bank overlaps the training data.** We expected overlap to make MLPs
  look over-confident. On weather it did not; size mattered more.
* **Batch size, for cosine's error against the theory.** It shows no trend from 64
  to 1024 negatives.

## 7. The normalisation story: where it stands

**The observation (solid).** Across goals 1–3, every catastrophic failure is an
unbounded score on heavy-tailed or spiky data:

* linear cosine on heavy tails (~10× worse than perfect);
* MLP cosine on heavy tails (+7% to +23%);
* unnormalised cosine on rain: linear (energy 1.4–1.9× climatology's) and one-
  and two-way MLP (2.0–2.4×).

The bounded scores (L² and normalised cosine) never fail like that. One caveat:
some MLP L² forecasts on rain also lean on only a couple of bank futures, but
without the loss of skill.

**The mechanism (plausible, not yet measured).** A dot-product score grows with the
size of the future. Reweighting by `exp(score)` therefore favours the biggest
futures without limit, and on heavy-tailed data a few extreme futures take all the
weight. CLIP also normalises for practical reasons (stable training, a fixed
temperature range), so the observation alone doesn't prove this is *the* reason.

**A direct check (cheap).** For unnormalised cosine on rain:

* measure the length of each future's embedding against the size of that future;
* measure how concentrated the forecast weights are.

If the long embeddings belong to the spiky futures and those futures take the
weight, the mechanism is confirmed.

**A clean theoretical argument (likely exists).** The model distribution is
`exp(score) × (distribution of futures)`, rescaled to sum to 1.

* For a score that is linear in the future (the dot product), the rescaling
  factor is finite only if the futures' distribution has light, exponential-type
  tails. For a Student-t it is infinite, so the "model distribution" the loss is
  supposed to be learning does not exist. A finite batch or bank hides this by
  collapsing onto its largest members, which is exactly what we observe.
* A score that is **bounded above** is different: L² is at most 0, and normalised
  cosine at most its scale. It gives a well-defined model for *any* distribution
  of futures.

So bounded scores are the ones for which the paper's probabilistic objective is
valid on arbitrary data. That is a stronger statement than "CLIP normalises for a
reason", it comes straight out of the paper's framework, and it could be written
up properly.

## 8. A research direction: the bank (reference set)

Forecasting by reweighting a bank of real futures is **importance sampling**. A
known result (Chatterjee & Diaconis, 2018) says the number of samples needed grows
roughly like `exp(KL)`, where KL measures how far the target distribution (here, the
forecast) is from the distribution the samples come from (here, the bank).

That explains the short-range failure in goal 3:

* At 10 minutes ahead the true forecast is very sharp, so it is far from the bank's
  broad spread. KL is large, and 5.8k futures are not enough.
* A bigger bank raises the sample count.
* The change-from-last-value target shrinks the gap, because a bank of *changes*
  is much closer to any particular forecast than a bank of *levels*.

A theory version would split forecast error into **score error** (did training find
the right weights?) plus **reference-set error**: bank size, a shift between the
bank's period and the test period, and how the future is represented. It would also
say which representations make a bank transferable. This is outside the paper,
which treats the bank as the true data distribution, and it is the limit that bit
hardest on real data.

## 9. Open questions

1. **Why does cosine work with MLP encoders?** Guess from goal 2: MLP bias terms let
   the dot product contain a piece that depends on the future alone. That is the
   piece the linear dot product is missing (the reason for P1) and that L² has
   built in. The paper's own footnote on affine encoders points the same way.
   *Cheap test:* add an explicit future-only term to linear cosine and see whether
   the spread is fixed.
2. **Over-sharpening.** Longer training made MLP forecasts over-confident:
   * goal 2: 30k steps were worse than 15k;
   * goal 3: MLPs over-fit within a few hundred to a few thousand steps.

   Is the learned temperature (the logit scale) the culprit? *Cheap test:* after
   training, tune only the temperature on validation. If that fixes calibration,
   the objective has a one-parameter sharpness problem, a cleaner story than "use
   early stopping".
3. **Why is joint consistently best?**
   * L² joint was the best or tied for best in goals 2 and 3.
   * On rain, joint cosine was the only unnormalised MLP cosine that didn't collapse.
   * In goal 1 it was also much less broken on heavy tails.

   The paper separates joint from conditional losses, but predicts the opposite
   ranking for conditional forecasting (P4). A guess: the joint softmax's shrinkage
   damps the pull towards extreme futures. Untested.
4. **Over-confidence when the input says little.** With temperature's own past
   removed, the contrastive forecast became over-confident instead of falling back
   towards "random past weather". Whether that is an estimation artefact or
   structural matters for real cross-modal use, where one modality often says little
   about the other.
5. **Does "the encoder learns the useful quantity" (goal 2) hold on real data?** We
   never checked what the weather encoders learned.

## 10. Caveats

* Synthetic rounds: 2 seeds and 500 test cases per setting, so differences under
  ~1% are noise. The fixed synthetic field setup has observation noise larger
  than the signal's own variation, which mutes the bent-series effects.
* Real data: one year, one station, one test period (late Oct – Dec, colder than
  most of training), 2 seeds.
* MLPs were barely tuned: one architecture, no weight decay or dropout, and only
  the learning rate compared. A regularised MLP may well beat the linear models on
  weather. That is the obvious next experiment.
* The normalisation mechanism (§7) and every open question (§9) are hypotheses,
  not results.
