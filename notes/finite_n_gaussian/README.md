# The linear-Gaussian contrastive forecaster at finite n

Companion to `notes/score_tails` (Sec. 3.2 and 5.5). The note works out the
linear-Gaussian setting of Baptista-Stuart-Tran with **infinite** data. Here we
train on **n** samples and ask which loss gives the better forecast, and why.

Code: `test_bed/theory/finite_sample_gaussian.py` (simulation),
`test_bed/theory/summarize_finite_sample.py` (tables and figure). Results:
`test_bed/results/theory/finite_sample_gaussian/` (raw, one file per
setting) and `test_bed/results/theory/finite_sample_gaussian.{md,png}`.

Reproduce (from `test_bed/`, about 1.5 h on 4 cores):

    python -m theory.finite_sample_gaussian
    python -m theory.summarize_finite_sample

## 1. What is exact

### The joint loss is the one-way loss plus a penalty

By the chain rule for KL, for any score with finite normaliser,

    J_joint(s) = J_{v|u}(s) + KL( mu_u || nu^s_u ),

where `nu^s_u(du) = Z(u)/Z mu_u(du)` is the model's marginal of the past. So the
joint loss is the one-way loss plus a penalty for getting the distribution of
pasts wrong. (This is the decomposition behind Stuart-Wolf Thm 2.6.) The
two-way loss is likewise the one-way loss plus a reverse one-way term.

* If the score family can zero the penalty without hurting the forecast
  (flexible encoders, or the "inner product + quadratics" family), the losses
  have the same minimiser. This is Prop 3.1 again.
* If it cannot, the penalty pulls the forecast away from the truth. That is
  the bias in Table 2.

### What the penalty does in the scalar L2 case

Unit-variance scalars with correlation rho, and `s = -1/2 (g u - h v)^2`.
The forecast is Gaussian with mean `m u` and precision `p`, where

    m = g h / (1 + h^2),   p = 1 + h^2.

The one-way loss is the Gaussian KL between this forecast and the truth
`N(rho u, 1 - rho^2)`. The penalty works out to

    KL(mu_u || nu^s_u) = 1/2 ( x - log(1 + x) ),    x = m^2 p / (p - 1).

(Checked numerically against the exact joint KL to 6 digits.) Read off what it
rewards:

* It grows with the mean coefficient `m`, so it **shrinks the mean** towards 0.
* For fixed `m` it **decreases** as `p` grows, so it pushes towards a
  **sharper** forecast, not a wider one.

The joint optimum (Prop 3.7) shows both: at rho = 0.5 it has m = 1/3 (truth
0.5) and p = 1.5 (truth 1.33). So the joint L2 loss is a shrinkage estimator
for the mean but **not** a brake on over-confidence. Any finite-sample
advantage it has must come from the mean shrinkage.

### Why the penalty is large at weak signal

At the truth, `x = m^2 p/(p-1) = 1` for every rho, so the penalty there is
`(1 - log 2)/2 = 0.153` nats, a constant. The information available is
`I(u;v) = -1/2 log(1 - rho^2) ~ rho^2/2`, which goes to 0. When the signal is
weak, the joint loss cannot afford to sit at the truth, which is why it loses
3/8 of `I(u;v)` as rho -> 0 (Prop 3.7).

## 2. Simulation

**Setup.** u, v in R^d, each N(0, I), with correlation rho in each coordinate.
Three score families (dot, L2, inner product + quadratics), each trained with
the one-way, two-way and joint losses on the same n training pairs (full
batch: every other pair in the training set is a negative). Each fit is scored
by the exact forecast error E_u KL(true v|u || fitted v|u), reported as a
fraction of I(u;v): 0 = perfect, 1 = no better than the marginal
("climatology"). 5 scalar settings (rho = 0.1 to 0.9) and d = 2, 4, 8 at
rho = 0.5; n = 32 to 2048; 70 to 200 training sets per cell.

Full tables: `test_bed/results/theory/finite_sample_gaussian.md`. Figure
(L2 score): `test_bed/results/theory/finite_sample_gaussian.png`.

### Finding 1: with a family that can represent the truth, the loss still does not matter at finite n

For "inner product + quadratics", which can represent log r exactly, the three
losses give the same error at every n to within noise. The only visible
difference is at the smallest n, where joint is slightly better (by up to about
12%, e.g. d = 8, n = 64: 1.29 vs 1.47). So the thesis extends from infinite
data to finite data, in this setting: **the loss only matters through what the
score family can represent.**

### Finding 2: with the rigid L2 family, one-way wins except in a narrow small-n regime

| setting | joint better than one-way for | crossover n |
|---|---|---|
| d = 1, rho = 0.1 | n <= 128 | ~200 |
| d = 1, rho = 0.3 to 0.9 | never (n >= 32) | below 32 |
| d = 2, rho = 0.5 | never (n >= 32) | below 32 |
| d = 4, rho = 0.5 | n = 32 | ~40 |
| d = 8, rho = 0.5 | n <= 64 | ~100 |

(paired differences, significant at 2 standard errors; full numbers in the tables.)

The joint and two-way errors level off at their population bias (the dashed
lines in the figure) within a few hundred samples; one-way keeps falling like
1/n. Joint only wins while one-way's estimation error is larger than joint's
bias: when the **signal is weak** (rho = 0.1) or **n is comparable to the number
of parameters** (d = 8 has 100 forecast parameters; joint wins below n ~ 100).
The crossover moves the way a bias-variance trade-off predicts: later for
weaker signal and for more parameters.

So the joint loss does act as a regulariser at small n, but in the Gaussian
it is a small effect confined to that regime. It is a shrinkage of the mean
(Sec. 1), not protection against over-confidence.

### Finding 3: at weak signal and small n, every contrastive forecast is worse than climatology

At rho = 0.1, every family and loss has error above I(u;v) (worse than just
forecasting the marginal) up to n = 64, and only drops below at n = 128. With n = 32 the error
is 2.4 to 3.7 times I(u;v). This is the regime of the rain experiment, where
the best model beats climatology by 1%.

### Side observations

* **One-way L2 converges faster than the textbook rate.** Its error is about
  0.5 to 0.85 times k/(2n), where k is the number of forecast parameters
  (k/(2n) is the rate of a well-specified maximum-likelihood fit). Not yet
  explained; the empirical loss uses the training futures as the bank, which
  is not exactly maximum likelihood.
* **Below n = number of parameters, everything blows up.** d = 8, n = 32: mean
  errors of 10^3 to 10^6 times I(u;v). The fitted forecasts become arbitrarily
  sharp. Joint is the least bad.
* **P3 holds at finite n.** For the dot product, one-way and two-way agree at
  every n, not only at the population optimum.

## 3. What this suggests for the weather results (Sec. 5.5 of the note)

The Gaussian says joint beats one-way only when n is small relative to the
number of parameters, or the signal is weak. The weather experiments are in
that regime: MLP encoders have many parameters relative to the training
data (they over-fit within a few hundred steps), and rain has almost no signal. So the
weather result is consistent with this picture, but the Gaussian effect is
small, so the size of the weather gap needs another ingredient (most likely
the MLP itself, which this linear study does not cover).

## 4. Open questions

1. **Explain the one-way rate.** Derive the asymptotic error of the empirical
   one-way estimator (the bank is the training futures) and check the
   0.5 to 0.85 factor.
2. **Predict the crossover.** Joint's error = bias + its own variance; one-way's
   = variance only. A formula for both variances would give the crossover n
   as a function of rho and d.
3. **Mini-batches.** Training uses batches of K pairs, not the full set. Does
   batch size change the picture?
4. **Beyond Gaussian.** Repeat the training-size sweep on the heavy-tailed and
   bimodal domains with MLP encoders.
