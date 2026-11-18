# Score tails in contrastive forecasting

Theory note on which design choices in a CLIP-style contrastive forecaster
matter: the loss, the divergence, or the score.

## Files

| file | what it is |
|---|---|
| `main.tex`, `main.pdf` | Paper-style rewrite, in the style of Baptista–Stuart–Tran and Stuart–Wolf. Theory first, then experiments, with proofs in the appendix. |
| `original_note.pdf` | The original note, *"For flexible encoders, neither the contrastive loss nor the divergence matters; the score matters only through its tails"*. |

Rebuild the PDF with `pdflatex main.tex` (run it twice).

## The idea in plain language

- **r(u,v)** = p(v | u) / p(v): how many times more likely the future v is once
  the past u has been seen. log r is the pointwise mutual information.
- **The model.** A score s(u,v) defines a forecast. Take every possible future
  and weight it by e^{s(u,v)}:
  ν(dv|u) = e^{s(u,v)} μ_v(dv) / Z(u), where Z(u) = ∫ e^{s(u,v)} μ_v(dv).
- **The bank forecast.** In practice μ_v is replaced by a bank of N observed
  futures, weighted by softmax(s). This is self-normalised importance sampling.

### Half 1: the loss and the divergence drop out

- **Prop 3.1.** Every loss vanishes exactly when the score hits log r, up to a
  free constant that depends on the loss:
  - one-way: s = log r + c(u)
  - reverse one-way: s = log r + c'(v)
  - two-way and joint: s = log r + const

  The divergence D never appears in these conditions.
- **Losses disagree only under misspecification.** If the score family can
  represent log r, every (loss, divergence) pair has the same minimiser, the
  true forecast. When the losses disagree, the family is too rigid. The linear
  Gaussian case of Baptista–Stuart–Tran is an example (Sec. 3.2):
  - the dot product has no future-only term, so its forecast spread is stuck
    at climatology;
  - linear L² couples past and future only by shrinking both marginals, so it
    can lose up to 3/8 of the mutual information.
- **Thm 3.4 (flexible encoders).** Combines Stuart–Wolf Thm 4.1/4.2
  (universal approximation in joint KL) with their Thm 2.6 (each conditional
  KL ≤ joint KL). One flexible score then makes every KL loss variant small
  at once, on compact domains.
- **Prop 3.5.** KL barely charges for forecast mass moved far away; energy
  distance does. So different divergences agree at the optimum but can
  disagree in the tails.

### Half 2 (the main contribution): the score matters through its tails

| | unbounded score (raw dot product) | bounded above (L²) | bounded on both sides (normalised cosine) |
|---|---|---|---|
| Z(u) finite / model exists | only with light-tailed futures | always (Prop 4.1) | always |
| bank forecast converges (Thm 4.5) | no, on heavy tails | yes | yes |
| what goes wrong | collapses onto the most extreme future (Prop 4.6) | none of these | sharpness capped at 2κ (Prop 4.7) |

- **Prop 4.2.** A score linear in the future needs a finite moment generating
  function. A Student-t's is infinite, so every informative dot-product
  parameter has infinite loss.
- **Prop 4.3.** ReLU encoders don't help, because they grow linearly at
  infinity.
- **Remark 4.8.** Training can't catch this: a batch of 256 almost never
  contains a far-tail future, so the empirical loss stays finite.
- **Prop 4.4.** Bounded scores have a price: they can be *exact* only when r is
  bounded. They can still approximate in KL.

### Saturating scores and the embedding dimension (Secs. 4.3–4.4)

- **min(N, A) law (Thm 4.21).** A bounded score that saturates *high* along
  extreme futures over-weights them by a tail amplification A. A bank of N
  futures exposes the over-weighting up to min(N, A). Larger banks expose
  more of it.
- **Logarithmic tails.** The Student-t kernel and log-quadratic scores have
  the right tail for elliptical heavy-tailed data. The log-quadratic family
  contains the truth with *linear* encoders (Prop 4.16).
- **Bank energy loss.** Adding the energy score of a bank forecast to
  training places extreme futures low. It sees amplification only up to the
  training bank size K (Thm 4.23).
- **Information split (Prop 4.28).** For a Student-t series the information
  splits exactly into a correlation part and a shared-scale part. The scale
  part is 1.20 nats for ν = 3 and 50/50 windows.
- **Dimension (Thm 4.30, Cor 4.31, Thm 4.32).** A nonlinear encoder needs k + 1
  embedding dimensions (k = number of canonical correlations). A linear
  encoder needs the full dimension. A rank-d_z linear embedding caps the tail
  index at ν + d_z.
- **Odd interactions (Prop 4.33).** The standard linear scores (inner
  product, tanh, normalized cosine, L²) cannot use the scale at all. At zero
  correlation they learn nothing.

### Experiments (Sec. 5)

Run from `test_bed`:
- `python -m analysis.summarize_tail_sweep` produces `results/tail_sweep/summary.md`.
- `python -m analysis.summarize_dim_sweep` produces `results/dim_sweep/summary.md`.
- `python -m analysis.paper_assets` regenerates the figures and the generated table blocks of `main.tex`.

Results:
- **Bank sweep, ν = 3 (5 seeds).**
  - The unbounded inner product turns around and degrades by orders of magnitude.
  - Linear tanh goes from +28% to +84%.
  - The logarithmic scores level off near the oracle:
    - linear log-quadratic: +0.3%;
    - MLP Student-t kernel: +0.6%.
- **min(N, A) in trained models.** The ray probe of A predicts the
  over-forecast factor past by past, with rank correlation 0.96.
- **Bank energy loss.**
  - The MLP tanh plateau drops from +6.0% to +1.3%.
  - Its over-forecast factor drops from 3.6 to 0.5.
- **Real data (BTC, 3 seeds; Jena rain and Peachtree Creek discharge, 2 seeds).**
  - On rain the unbounded inner product fails (+330% to +850%).
  - With the bank energy loss the same MLP beats climatology by 6.6%.
  - On discharge the MLP tanh fails at large banks in one of two seeds.
  - On BTC only the linear log-quadratic beats climatology among linear models (−3.8%), as the odd-interaction result predicts.
- **Embedding dimension and correlation (1 seed).**
  - D1 holds: the odd-interaction linear scores are flat in d_z and capture nothing at φ = 0.
  - D3 holds: MLPs saturate at d_z = 2.
  - D2 holds only partly: linear Student-t improves slowly, far below the ceiling.
  - The tail cap is not visible at p = 10⁻⁴.
  - New finding: for the linear tanh score, the over-forecast at p = 10⁻⁴ grows with d_z, from 2.3 to 47.
- **Finite data.** On weather data the joint L² loss beats one-way L², by
  over-fitting, not by population bias.

## Relation to the two reference papers

**Baptista, Stuart & Tran (arXiv:2505.24134).** Supplies the whole framework:
- the tilted model (eqs. 13–16);
- the losses as KL divergences (Thm 3.2, Thm 3.8, Rem. 3.12);
- general divergences (Opt. Probs. 3.4/3.6, Rem. 3.5);
- the bank forecast (eq. 31);
- the Gaussian predictions P1–P4 (Sec. 5).

Their Gaussian, linear setting hides both effects studied here. The families
are rigid, so the loss appears to matter. Gaussian tails keep every moment
generating function finite, so the tail problem never shows up.

**Stuart & Wolf (arXiv:2608.17203).**

| | Stuart & Wolf | this note |
|---|---|---|
| loss variants | conditional losses ≤ joint KL (Thms 2.5/2.6) | all variants have the same zero set (Prop 3.1) |
| divergences | KL only; energy score listed as future work | any divergence (Prop 3.1); where KL and energy distance differ (Prop 3.5) |
| unbounded domains | Thm 3.3 truncates the score to 0 outside a compact set K. Prop A.3 needs \|s\| ≤ B outside K, with an e^B error term. The CLIP theorems are stated only for compact domains. | Without that tail bound the model doesn't exist (Props 4.2, 4.3) and the bank collapses (Prop 4.6). The condition is necessary, not technical. |
| normalised cosine | universality needs τ → 0 | the bounded score guarantees the bank works but caps sharpness at 2κ = 2/τ (Prop 4.7) |
| finite samples / training | open questions | bank-size results (Thm 4.5, Prop 4.6), the batch argument (Remark 4.8), all experiments |

## Open items before sending

- Fill in the author block in `main.tex`.
- More seeds for the dimension sweep (currently 1), and on the real series.
- Optional proposition: tail amplification of the tanh score grows with d_z
  (the ℓ1 versus ℓ2 argument of Rem 4.19, currently in the text of Sec 5.9).
- Finite-sample guarantees are deliberately deferred.
