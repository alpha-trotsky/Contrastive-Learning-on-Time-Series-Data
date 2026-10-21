# Goal 2 results — nonlinear (MLP) encoders on non-Gaussian domains

Setup: the same three domains and dials as goal 1, with the field setup in
`config.yaml` left unchanged. Every model was trained for 15k steps at batch 256, with 2 seeds.
Training and test data are identical to goal 1, so linear and MLP numbers are
directly comparable.

* **Encoders:** 2 hidden layers of width 256, GELU, plus a linear shortcut (so the
  MLP model contains the linear one). `models/mlp_clip.py`.
* **Scores:** the full grid {cosine, L²} × {one-sided, two-sided, joint}, plus
  **normalised cosine** (real CLIP: embeddings scaled to length 1 before the dot
  product, two-sided).
* **Non-contrastive baselines** (`models/mlp_regressor.py`):
  * **MSE + noise**: an MLP predicts the average future, and one fixed Gaussian
    noise band is added around it.
  * **Gaussian NLL**: an MLP predicts the average *and* a band width per future
    step, so the width depends on the past. The shape is always Gaussian. This is
    the "best nonlinear Gaussian model".
* Contrastive models are still scored only through their own score, by reweighting
  a bank of 50k real futures. **Change from goal 1:** the bank is now a separate set
  of fresh futures, not taken from the training data, because a flexible encoder
  could memorise its training pairs. For linear L², switching the bank moves the
  numbers by at most 0.2 points of energy and 0.01 coverage (checked on 4 domains),
  so the goal-1 linear numbers stay comparable.

Full tables: `test_bed/results/domain_sweep/summary_mlp.md` (MLP, plus linear → MLP
comparison) and `summary.md` (linear, now including linear normalised cosine and
linear Gaussian NLL).

How to read the columns (same as goal 1):
* **energy vs oracle**: how much worse than the perfect forecaster (0% = perfect).
* **band90**: how often the 90% band catches the truth (target 0.90). "Hard" = the
  third of cases where the perfect forecaster is most uncertain.

## Headline table: energy vs the perfect forecaster, linear → MLP (mean of 2 seeds)

| domain | dial | cos 1-sided | cos 2-sided | cos joint | norm. cos | L² 1-sided | L² 2-sided | L² joint | MSE + noise | Gauss NLL |
|---|---|---|---|---|---|---|---|---|---|---|
| Heavy tails | ∞ (GP) | +5.6 → +2.7 | +5.6 → +1.4 | +9.8 → +0.3 | +13.1 → +0.5 | +0.0 → +0.3 | +0.0 → +0.3 | +0.0 → +0.1 | +0.1 → +0.6 | +0.2 → +0.4 |
| | 10 | +12.9 → +3.4 | +12.9 → +4.7 | +19.5 → +0.4 | +14.5 → +0.9 | +1.2 → +2.5 | +1.3 → +0.8 | +1.2 → +0.4 | +1.6 → +2.5 | +1.6 → +0.6 |
| | 3 | ~9× → +6.8 | ~10× → +23.3 | +69.5 → +13.3 | +23.0 → **+0.9** | +4.6 → +4.3 | +4.5 → +4.0 | +4.5 → **+0.7** | +8.6 → +9.5 | +45.9 → +22.2* |
| Bent GP | 0 (GP) | +5.5 → +1.0 | +5.6 → +0.5 | +12.5 → +0.1 | +12.5 → +0.4 | −0.0 → +0.2 | −0.0 → +0.1 | +0.0 → +0.1 | +0.0 → +0.5 | +0.1 → +0.4 |
| | 0.5 | +16.2 → +1.2 | +17.1 → +0.9 | +23.1 → +1.4 | +12.0 → +0.5 | +0.2 → +0.7 | +0.3 → +0.3 | +0.3 → +0.2 | +1.7 → +2.6 | +0.2 → +0.7 |
| | 1 | +37.4 → +5.9 | +40.0 → +3.9 | +25.2 → +0.6 | +12.2 → +0.9 | +1.6 → +3.0 | +1.5 → +1.1 | +1.6 → +0.7 | +12.7 → +14.7 | +27.5 → +0.7 |
| Double-well | 0 (OU) | +0.3 → +0.4 | +0.4 → +0.3 | +1.5 → +0.3 | +2.0 → +0.2 | +0.1 → −0.0 | −0.1 → +0.0 | +0.1 → +0.3 | −0.1 → +0.3 | +0.1 → +0.2 |
| | 1 | +1.5 → +1.3 | +1.5 → +0.9 | +3.3 → +0.6 | +2.6 → +0.8 | +0.7 → +0.6 | +0.5 → +0.9 | +0.6 → +0.7 | +10.9 → +6.7 | +10.3 → +5.9 |
| | 1.5 | +1.3 → +0.6 | +2.6 → +0.3 | +2.7 → +0.5 | +0.7 → +0.3 | +0.4 → +0.4 | +0.1 → +0.2 | +0.3 → +0.0 | +14.1 → +7.6 | +13.5 → +5.2 |

(all in %; \* one seed +0.8%, the other +43.6%)

## 90% band coverage on the hard third of cases, linear → MLP (target 0.90)

| domain | dial | cos 2-sided | norm. cos | L² 1-sided | L² joint | MSE + noise | Gauss NLL |
|---|---|---|---|---|---|---|---|
| Heavy tails | ∞ | 0.98 → 0.89 | 0.92 → 0.89 | 0.89 → 0.89 | 0.89 → 0.89 | 0.89 → 0.89 | 0.89 → 0.89 |
| | 10 | 0.95 → 0.86 | 0.85 → 0.88 | 0.80 → 0.86 | 0.81 → 0.89 | 0.81 → 0.81 | 0.81 → 0.88 |
| | 3 | 0.64 → 0.87 | 0.79 → 0.89 | 0.76 → 0.87 | 0.77 → **0.89** | 0.81 → 0.81 | 0.89 → 0.90 |
| Bent GP | 1 | 0.92 → 0.87 | 0.93 → 0.89 | 0.89 → 0.87 | 0.89 → 0.89 | 0.80 → 0.80 | 0.92 → 0.92 |
| Double-well | 1.5 | 0.91 → 0.90 | 0.89 → 0.90 | 0.89 → 0.90 | 0.90 → 0.90 | 0.91 → 0.92 | 0.91 → 0.94 |

## What happened, hypothesis by hypothesis

**1. Heavy tails: the bands can now depend on the past.** Yes.
With linear encoders every model had one band width for all cases, so it was too
wide on easy cases and too narrow on hard ones (linear L² at df=3: 0.98 / 0.92 /
0.76). With MLP encoders every contrastive score except joint cosine lands at
0.87–0.89 on the hard third at df=3, and the easy third drops back to ~0.90. Joint
cosine keeps the old imbalance (0.98 / 0.92 / 0.81). The MSE + noise baseline cannot do
this (stays 0.81) because its noise band is fixed by design.
The energy gap only closes for **L² joint (+4.5% → +0.7%)** and **normalised cosine
(+0.9%)**. One- and two-sided L² fix the bands but not the energy (+4%). Their
average prediction gets worse (1.1× the oracle's error) and some test cases collapse onto
a single bank future (see point 5).

**2. Bent GP: the encoder learns to un-bend the data.** Yes, clearly.
We fit a straight line from the learned past-features to the true un-bent
quantities, on held-out cases. At full warp:

| fit target | from the raw past | from linear-encoder features | from MLP features (L² variants) |
|---|---|---|---|
| un-bent past (all 50 steps) | R² 0.68 | ≤ 0.68 (can't beat raw) | 0.71–0.79 |
| **un-bent best forecast** (the part of the past that matters) | R² 0.78 | ≤ 0.78 | **0.99–1.00** |

So the encoder recovers exactly the part of the hidden un-bent past that predicts
the future, and doesn't bother with the rest. That is what the theory says the
best score needs. The best contrastive models end up within 0.6–1.1% of perfect
on the bent GP (linear: ~1.6%).

**3. L² beats the best nonlinear Gaussian model on the double-well.** Yes.
At λ = 1 and 1.5 every L² variant is within 0.0–0.9% of perfect. The Gaussian NLL
MLP is at +5.9% / +5.2%, and MSE + noise at +6.7% / +7.6%. Same MLP, same data: the
Gaussian models lose because they put probability between the two wells, where the
future never is. The contrastive forecast reweights real futures, so it keeps the
two separate outcomes.
The Gaussian NLL MLP *does* match L² on the bent GP (+0.7%). There the futures are
skewed but single-peaked, which a Gaussian with past-dependent width can
approximate well. It is also unreliable on heavy tails: one seed at +0.8%, the
other at +43.6%, from a few cases with extreme pasts where its predicted width
explodes.

**4. Cosine (unnormalised): nonlinear encoders mostly fix it, except on heavy tails.**
* The paper's known flaw (right average, bands too wide) largely disappears. On
  the GP, cosine goes from +5.6% to +1.4–2.7%, and joint cosine from +9.8% to
  +0.3%. Bands are no longer too wide (hard-third coverage 0.98 → 0.89).
  A likely reason, not yet tested directly: with an MLP (which has bias terms), the
  dot product can contain a piece that depends on the future alone. That is
  exactly the piece a linear dot product lacks and the L² score has built in.
* On heavy tails, the collapse is much milder than with linear encoders (~10× worse →
  +7% to +23%), but it is still the worst contrastive score. Its average
  prediction is 1.2–7.6× the oracle's error, and some cases still collapse onto one
  bank future. An unbounded score that grows with the size of the future keeps
  pulling weight towards extreme futures, and a better encoder doesn't change that.

**5. Normalised cosine is the practical winner along with L² joint.**
* With MLP encoders it is within 0.2–0.9% of perfect on **every** domain and dial,
  including df=3, with hard-third coverage 0.88–0.90. It also collapses least: its
  smallest effective sample size stays above 90 in every domain and dial except
  df=10 (1.9). The unnormalised scores and one/two-sided L² hit 1–2 on heavy tails
  and the bent GP.
* With **linear** encoders, normalised cosine is poor (+12–23%). A linear encoder
  followed by normalisation only sees the *direction* of the past, not its size, and
  on these series the size (the current level) is most of the information. So
  normalisation needs a nonlinear encoder to work.

## Effective sample size (collapse check)

Median / smallest effective number of bank futures used per test case (bank of
50k), MLP encoders. Smaller = fewer real futures make up the forecast; 1 = a
single future.

| domain | dial | cos 2-sided | cos joint | norm. cos | L² 1-sided | L² joint |
|---|---|---|---|---|---|---|
| Heavy tails | 3 | 3107 / 1.0 | 16702 / 2.7 | 2419 / 93 | 2860 / 1.0 | 3364 / 5.9 |
| Bent GP | 1 | 9146 / 1.2 | 13287 / 25 | 7537 / 175 | 9459 / 1.0 | 11168 / 3.5 |
| Double-well | 1.5 | 19924 / 604 | 20344 / 720 | 20131 / 651 | 19982 / 580 | 20299 / 759 |

Typical cases are fine everywhere (thousands of futures). The minimum shows that
a few extreme pasts still collapse for unnormalised scores and one-sided L².

## What it means for the research question

The probabilistic view says a trained contrastive model *is* a conditional
distribution: real futures reweighted by the score. These results say that view
carries over to nonlinear series with nonlinear encoders, and they sharpen what the
choice of score actually controls.

1. **The view works as a forecasting method with no closed form.** With MLP
   encoders, the best contrastive models are within ~1% of the perfect forecaster
   on all three kinds of non-Gaussianity, with calibrated bands. They are scored
   purely as black boxes.
2. **"Cosine gets the spread wrong" is a property of linear encoders, not of the
   dot product.** Once the encoder is flexible, cosine and L² can both represent
   the right answer, and the large goal-1 gap mostly disappears on well-behaved data.
3. **What still matters is whether the score is bounded.** The remaining failures
   are unbounded scores on heavy tails. Normalising the embeddings (as real CLIP
   does) removes them. This gives a probabilistic reason for why normalisation
   matters, and it is the cleanest new claim from goal 2.
4. **The advantage over regression is the forecast's shape, not encoder power.**
   A Gaussian MLP with past-dependent widths has the same networks and still loses
   on bimodal futures. The contrastive forecast inherits its shape from real data.
5. **The encoder learns what the past says about the future** (bent GP: R² ≈ 1 for
   the un-bent forecast), without being told what that is.

## Caveats

* 2 seeds, 500 test cases each. Differences under ~1% are within noise. One- vs
  two-sided L² also varies a lot between seeds on heavy tails (+3% to +6%).
* **Training length matters for MLPs.** On one config each of heavy tails (df=3) and
  bent GP (full warp), 30k steps was worse than 15k (heavy tails +5.6% vs +3.1%, bent
  GP +5.5% vs +1.5%, L² one-sided). The forecasts became over-confident and the
  effective sample size shrank. It is not memorisation of the bank: the held-out
  bank gives the same result. All runs here use 15k. On real data, with no oracle,
  training needs a held-out stopping rule (e.g. energy score on a validation split).
* MLP architecture and learning rate were not tuned (one setting: width 256,
  depth 2, lr 1e-3).
* The bent-GP effect is muted by the field setup, which is deliberately left
  unchanged: its observation noise (0.05) is larger than the GP's own variation
  (~0.04).

## For goal 3 (real data)

* Carry forward **L² joint** and **normalised cosine** with MLP encoders. Keep
  unnormalised cosine as the reference the theory predicts will fail on heavy tails.
* Keep **Gaussian NLL MLP** as the baseline to beat. It is strong on single-peaked
  data, so beating it on real series is the meaningful test.
* Add early stopping on a validation energy score, and report the minimum effective
  sample size as a collapse check.
