# Score tails in contrastive forecasting

Theory note on which design choices in a CLIP-style contrastive forecaster
matter: the loss, the divergence, or the score.

## Files

| file | what it is |
|---|---|
| `main.tex`, `main.pdf` | Paper-style rewrite, in the style of Baptista–Stuart–Tran and Stuart–Wolf. Theory first, then experiments, with proofs in the appendix. |
| `original_note.pdf` | The original note, *"For flexible encoders, neither the contrastive loss nor the divergence matters; the score matters only through its tails"*. |
| `references/stuart_wolf_2026_expressivity.tex` | LaTeX source of Stuart & Wolf, *Expressivity in Multimodal Contrastive Learning*, arXiv:2608.17203 (Aug 2026), kept for reference. |

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

### Experiments (Sec. 5)

- **Loss variants.** With linear encoders the loss and the score both matter.
  With MLPs the gap between loss variants shrinks (bent GP: 14.8 → 5.3).
- **Heavy tails.** Every catastrophic failure is the unbounded dot product on
  heavy tails.
- **Bank sweep.** On heavy tails the dot product gets *worse* as the bank
  grows (MLP +7% → +46%; linear +33% → +3358%). Bounded scores improve and
  level off.
- **tanh ceiling.** Much better than the raw dot product, but it still
  degrades with bank size. The refinement (empirical, not proved): the score
  must also not *reward* extreme futures.
- **Finite data.** On weather data the joint L² loss beats one-way L². This
  is a hypothesis about over-fitting, not a contradiction of the infinite-data
  theory.

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
- Check Thm 3.4, Prop 4.4 and the opening of Sec. 4. These are new relative
  to the original note.
- The scripts cited in the reproducibility paragraph are not in this repo:
  - `test_bed/theory/misspecification.py`
  - `test_bed/analysis/bank_sweep.py`
  - `test_bed/results/bank_sweep/summary.md`
- Untested: the divergence half (train with energy distance), the
  "don't reward extremity" refinement as a theorem, and the bank sweep on
  rain.
