# Paper Revision Notes

Pending fixes to apply to the paper. Each item references the target section and the exact change.

---

## Section 5.5 (Cone Problem)

**Action:** Add the following paragraph after the first paragraph of Section 5.5:

> If the mean vector of the state space is located far from the origin — as occurs when VICReg's variance term pushes state magnitudes to norms of 14-19 — then variations around that mean, even if spread across all 384 dimensions, are minuscule relative to the distance from the origin. All vectors therefore point in approximately the same direction, making cosine similarity blind to the emotional differences encoded in the perturbations around the mean.

---

## Section 9 — LSTM Bullet

**Action:** Change:

> "an LSTM upgrade path is documented"

to:

> "an LSTM upgrade is a natural next step"

---

## Section 9 — Diagnostic Terminology

**Action:** Replace:

> "state ablation"

with:

> "a diagnostic that tests whether zeroing the state vector changes retrieval output (state contribution check)"

**Action:** Replace:

> "norm drift"

with:

> "a diagnostic that monitors state vector magnitude stability over long session sequences (norm stability check)"

---

## Section 7.1 — Judge Description

**Action:** Update the judge description to reflect the new three-judge majority-vote setup:

- Three judge models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b`
- Majority vote (2/3 agreement required)
- Responses presented as "Response 1" and "Response 2" (not A/B) to avoid positional bias
- If all three judges disagree, verdict is TIE
- Inter-judge agreement rate is recorded per scenario

---

## Section 7.3 and 7.4 — Results Tables

**Action:** PLACEHOLDER — Update results tables with new majority-vote verdicts and inter-judge agreement rate once the new evaluation run is complete.

---
