# The linear-Gaussian contrastive forecaster at finite n

Companion to `notes/score_tails` (Sec. 3.2 and 5.5). The note works out the
linear-Gaussian setting of Baptista-Stuart-Tran with **infinite** data. Here we
train on **n** samples and ask which loss gives the better forecast, and why.

Code: `test_bed/theory/finite_sample_gaussian.py` (simulation),
`test_bed/theory/summarize_finite_sample.py` (tables and figure). Results:
`test_bed/results/theory/finite_sample_gaussian.{json,md,png}`.

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

_Results pending._
