# Kokoro — Improvements Report

**Date:** 2026-07-03 (Cycle 2 results appended same day)
**Scope:** Full audit + implemented improvements across retrieval, decoder, training, data, diagnostics, and evaluation — followed by the Cycle-2 world-model rebuild and full re-measurement.
**Verification status:** All 89 unit tests pass; MemoryStore (36/36), WorldMemory (16/16), and StateDecoder (35/35) self-tests pass on the new checkpoints; full diagnostic suite and both LLM evaluation sets re-run on the new model.

---

# CYCLE 2 — The world-model rebuild: what was done and what the numbers now say

Cycle 1 (Parts A–F below) found that the MLP transition model's recurrent state contributed nothing (ablation gap +0.0001) and that all published validation metrics were computed on a 100%-contaminated split. Cycle 2 fixed both and re-measured everything.

## 2.1 Changes

1. **GRU world model** (`kokoro/transition.py::TransitionModelGRU`, now the default checkpoint `checkpoints/transition_v2.pt`). The user state is a GRUCell hidden state — bounded, gated accumulation of history — decoupled from prediction via an MLP head (`predict()`). `predict_from_state()` and `load_transition_checkpoint()` give every consumer one arch-agnostic path; the legacy MLP class remains loadable.
2. **State-utility margin loss** (`training/train.py`): at every warm step, predictions from the accumulated state must beat predictions from a zeroed state by ≥ `util_margin` cosine, weight `util_weight`. This term optimizes exactly what `diagnostics/state_ablation.py` measures. **Ablation within Cycle 2:** the same GRU trained with weak incentives (util=10, margin=0.1, no VAD head) converged to a gap of **+0.005** — architecture change alone does not produce state use; the objective has to demand it.
3. **Auxiliary next-session VAD prediction head** (`vad_head`, weight 10): the next session's *text* is topically unpredictable, but its *emotional position* follows the arc. This gives the recurrent state a task where history genuinely pays, and it trains the exact quantity the deployed system consumes.
4. **Vectorized training** (`rollout_batch`): batched rollouts replaced ~213 per-step forward calls per batch with ~8, making CPU retraining a ~25-minute job (60 epochs, 10k trajectories).
5. **Checkpoint selection by (val_loss − ablation_gap)**: an epoch that predicts marginally better while ignoring its state can no longer win the save; the per-epoch ablation gap is logged in the training history.
6. **Data regenerated leakage-free-by-construction**: `trajectories_10k_v2.json` (10,000 train) + `trajectories_10k_v2_val.json` (2,000 val) share **zero** source conversations, include the low-arousal augmentation (arousal to −0.80) and the two deep arcs. All diagnostics now default to the disjoint val file.
7. **All remaining ad-hoc checkpoint loaders** (demo, validate, arc_separation, eval_worldmodel, arc_ood_eval, longitudinal_sim, probe_generalization) routed through the arch-detecting loader; two environment bugs fixed along the way (CUDA auto-detect segfault in diagnostics subprocesses → CPU-pinned encoder; scipy/torch OpenMP clash → numpy Spearman in topic_leakage).
8. **Judge-parser bug found and fixed** (`experiments/run_llm_eval.py`): `qwen3-32b` emits `<think>` blocks that the old parser read as the verdict — which is why it appeared to vote for Kokoro on *every* scenario in the previous evaluation. Think blocks are now stripped and the judge token budget raised so the verdict survives truncation.

## 2.2 Representation results (all on the conversation-disjoint val set)

| Metric | GRU v2 (honest) | MLP legacy (leaky split) |
|---|---|---|
| **State-ablation gap** (`state_ablation.py`) | **+0.1575** (normal 0.557 / ablated 0.715) | +0.0001 |
| Prediction vs naive "repeat last session" (`eval_worldmodel.py`) | **+0.182 cosine; model better on 90.1% of steps** | — |
| Probe valence r | **0.7695** | 0.698 (biased) |
| Probe arousal r | **0.6722** | 0.542 (biased) |
| Arousal-primary vs valence-primary arc loss gap (`per_arc_val_loss.py`) | **−0.0165 (arousal arcs better)**; the two new deep arcs are the two best-predicted | flagged worse |
| Topic leakage (emotional vs semantic axis Spearman) | **0.041** (independent) | — |
| Norm drift over 50 sessions (`norm_drift.py`) | **1.00×** (GRU-bounded) | unconstrained |
| Participation ratio | 37.8 / 384 | 339.7 (leaky; not comparable) |
| Arc separation ratio (`arc_separation.py`) | 1.227 (shuffled control 1.003) | 3.111 (leaky) |
| Longitudinal sim (`longitudinal_sim.py`) | valence MAE 0.247 warm; +20.9% improvement after warm-up; **10/10 arc changes detected, 0.7-session lag** | — |
| Val cosine prediction loss | 0.554 | 0.503 (MLP, no utility constraint) |

Notes on the two numbers that went *down*: the val prediction loss is ~0.05 higher than the unconstrained MLP's — that is the explicit price of the utility margin (the model sacrifices a little one-step prediction to be genuinely trajectory-dependent), and it is the honest trade the paper should present. PR and arc-separation dropped because (a) the legacy numbers were computed on contaminated data and (b) bounded GRU states compress cosine geometry; neither quantity is used by the deployed retrieval path (which operates in decoded (v,a) space).

Training config of record: 60 epochs, batch 32, VICReg 25/25/1 (γ=1), util_weight 50, util_margin 0.2, vad_weight 10; best epoch 43 selected by val_loss − gap; ~1.29M parameters.

## 2.3 Response-level evaluation (counterbalanced 3-judge panel + recency baseline)

Protocol: every judge scores each pair in both presentation orders (order-flipped vote → TIE, recorded as position-inconsistent); `<think>` stripped before parsing; exact sign test + Wilson CI; Condition C = last-k sessions, no state summary.

| Set | vs semantic-only (A) | vs recency (C) | retrieval diverges | position-inconsistent judgings |
|---|---|---|---|---|
| Standard (20, 4–5 sessions) | B 8 / A 6 / TIE 6 (p=0.79) | B 7 / C 3 / TIE 10 (p=0.34) | 12/20 (was 7/20 with MLP) | 18/60 |
| Long-history (20, 10–11 sessions) | B 4 / A 8 / TIE 8 (p=0.39) | B 7 / C 2 / TIE 11 (p=0.18) | 15/20 (was 11/20) | 22/60 |
| **Pooled (40)** | **B 12 / A 14 / TIE 14 (p=0.85)** | **B 14 / C 5 / TIE 21 (p=0.064)** | 27/40 | **40/120 (33%)** |

Within the standard set, when retrieval actually diverged: B 7 / A 3 / TIE 2. In the long set the pattern reversed (A 7 / B 3 when diverging) — emotionally-current memories sometimes lose to semantically precise ones in judge preference; the two sets cancel to a pooled dead heat.

**Honest interpretation:**
- The prior headline win rates (83.3%, then 45–54.5%) were substantially **protocol artifacts**: a third of individual judgings flip with presentation order, and one of three judges was being mis-parsed into a constant Kokoro vote.
- Under the debiased protocol, **Kokoro vs semantic-only is a dead heat at n=40** — this claim must be retired from the paper as currently framed.
- **Kokoro beats the recency baseline 14–5 (21 ties, p=0.064)** — the emotional mechanism contributes something neither topical matching nor recency provides. This is the defensible response-level claim, and it did not exist as a measurable comparison before this cycle.
- The evaluation's dominant outcome is now TIE (14/40 vs A, 21/40 vs C): the judge panel frequently cannot distinguish the responses. Resolving this needs more scenarios and human raters, not another LLM pass.

## 2.4 What the contribution now is

The paper's center of gravity moves from "LLM judges prefer our responses" to three measurable, reproducible claims:

1. **A recurrent emotional world model whose state provably matters** — ablation gap +0.158 (trained via an explicit state-utility objective; the ablation showing architecture alone yields +0.005 is itself a finding), beating a naive persistence baseline on 90% of steps, with both circumplex axes linearly decodable (r = 0.77 / 0.67) on a leakage-free split.
2. **A data intervention that breaks a documented coverage ceiling** — synthetic deep-deactivation sessions in both valence polarities lift arousal decodability from 0.542 (biased) to 0.672 (honest) and make arousal-primary arcs the *best*-predicted arcs.
3. **An evaluation methodology result** — position counterbalancing + reasoning-parser hygiene + a recency baseline overturn the previous conclusions; 33% of LLM judgings were order-dependent. This is a cautionary result other companion-memory papers need.

---

---

## Part A — The Audit: what the system was, and where its ceiling sat

### A.1 What the system does

Kokoro tracks a user's emotional trajectory across sessions with a learned recurrent state (MLP transition model, VICReg-trained on 10k synthetic trajectories built from EmpatheticDialogues), decodes that state into (valence, arousal) via a linear probe, and uses those coordinates to (a) bias memory retrieval toward emotionally phase-matched past sessions and (b) inject a natural-language state summary into the companion LLM's system prompt. Headline claims: probe r = 0.698/0.542, arc separation 3.111, 83.3% LLM-judge win rate on long histories.

### A.2 The five findings that most limited credibility

**1. Train/val leakage was total (measured: 100%).** Trajectory construction reuses each source conversation in up to 8 trajectories (`max_uses_per_conv=8`). The split in `train.py`/`train_probe.py` was at the trajectory level only. Measured on `trajectories_10k.json` with the training seed: **every single val trajectory shared at least one source conversation — i.e. identical text and identical cached embedding — with the train set.** Every published val metric (val loss 0.5087, probe r 0.698/0.542, PR 339.7, arc separation 3.111) was computed on partially-seen data and is optimistically biased to an unknown degree. This is the single most dismissal-worthy fact about the previous results.

**2. The "world model" claim was unsupported — and the diagnostic that would have shown it was broken.** Four diagnostics (`state_ablation`, `topic_leakage`, `norm_drift`, `per_arc_val_loss`) rebuilt the transition model while **dropping `session_dim` from the checkpoint config**, so they crashed on the production VAD checkpoint (LayerNorm 384 vs stored 387 weights) — the monitoring suite could not monitor the model it was built for. After repairing the loaders, `state_ablation` runs and reports **gap = 0.0001** (threshold 0.05): zeroing the recurrent state at every step changes prediction loss by nothing. The current checkpoint is, for prediction purposes, a session-only encoder. The trajectory information the paper attributes to the learned state actually comes from (a) the decoded per-session (v, a) values and (b) the `arc_history` linear slope — both of which survive without the 384-dim state.

**3. The evaluation had an uncontrolled confound and no statistics.**
- **No recency baseline.** The long-history scenarios are two-phase (old emotion → new emotion, same keywords). Emotional retrieval wins by pulling Phase-2 sessions — but *"just take the most recent k sessions"* pulls Phase-2 sessions too, with zero learned machinery. Without Condition C (recency), the 83.3% result does not establish that the world model does anything a `list[-3:]` can't.
- **Position bias unaddressed.** Judges always saw Response 1 = A, Response 2 = B. LLM judges have well-documented slot preferences; every verdict carried this confound.
- **No significance testing.** "10/20 = 50%" and "5/6 = 83.3%" were reported as bare rates. 5/6 has a two-sided sign-test p = 0.22 — not significant — and the paper never said so.

**4. Retrieval scored two incommensurable axes and had silent-failure paths.**
- Semantic cosine lives in [-1, 1]; the VAD emotional score lives in ≈[0, 1]. Blending them with α means the *effective* weighting is not the nominal α.
- `get_context("")` (called at the end of **every** `update()`) fed the raw transition-model state vector into retrieval *as if it were a MiniLM embedding* — cosine between two unrelated vector spaces, i.e. noise ranked confidently.
- The cosine fallback for the emotional axis is documented (by the project's own findings) to be non-discriminative, yet remained the default when v/a weren't passed.
- No recency or time modeling at all — for *emotional* relevance, the most recent sessions are the most obvious signal, and the system had no way to use it.

**5. Fragile calibration and data ceilings.**
- Decoder thresholds were hardcoded constants, regex-patched into `decoder.py` source by the probe trainer. Any probe/checkpoint mismatch silently misclassifies every user ("positive" vs "neutral" etc.).
- The arousal floor: EmpatheticDialogues' lowest-arousal labels sit at ≈ −0.30, so arcs specifying arousal −0.5…−0.7 silently sampled −0.25 sessions. The model has *never seen* a genuinely deactivated session labeled as such — the documented cause of arousal r = 0.542.
- The stale test suite (1 failing test, 3 tests asserting the removed L2 normalization) signalled that the tests weren't run against the shipped code.

### A.3 Smaller defects found

- `_ARC_HISTORY_MAX = 20` caps trend history silently; fine, but interacts with long-history claims.
- `MemoryStore` timestamps use `time.time()` (~15 ms resolution on Windows) — same-tick inserts had ambiguous order.
- `vicreg_loss` crashed with an unhelpful stack trace when a batch contained no trajectory of length ≥ 2.
- `predict_next()` decodes with `arc_history=[]`, and the experiment `_InMemoryStore` decoded the retrieval state with empty history — harmless today (trend unused there) but a trap.
- `MemoryStore.add_session` *required* a state vector, making the store unusable as a plain RAG index (and breaking one shipped test).

---

## Part B — Changes implemented (all verified)

### B.1 Retrieval core (`kokoro/retrieval.py`) — smarter, honest scoring

| Change | Why it raises impact |
|---|---|
| Semantic cosine rescaled to [0, 1] before blending | α now means what the paper says it means; the two axes are commensurable. Rankings at α∈{0,1} are unchanged (monotonic transform), so prior per-condition results remain interpretable. |
| `query_embedding=None` supported (semantic axis disabled) | Removes the state-as-MiniLM-query hack; `get_context("")` now ranks by emotional (+ recency) proximity, which is meaningful. |
| **Recency axis**: `score = (1-γ)·hybrid + γ·exp(-rank_age/τ)`, default γ=0 | Gives the production system the same signal the new recency *baseline* uses — a deployed system should never lose to `list[-3:]`. Timestamps made strictly monotonic (counter tiebreaker) so ordering is well-defined. |
| **Adaptive alpha** (`adaptive=True` → `adaptive_alpha()`) | When a user's stored history is emotionally homogeneous (dispersion of stored (v,a) below 0.15), the emotional axis carries no ranking information and only injects decoder noise — alpha falls back to 1.0. When the history spans distinct phases (dispersion ≥ 0.45), the configured blend is kept. This directly targets the eval's observed failure mode: A won on `stable_positive`/`stable_negative` scenarios precisely because the emotional axis had nothing to add there. It converts a fixed hyperparameter into a per-user policy — a genuine mechanism, and an ablation the paper can report. |
| `state_vector` optional in `add_session` | The store works as a plain semantic RAG index (deployability; fixes the shipped failing test). |
| Results now include `recency_score` and `effective_alpha` | Every retrieval decision is auditable. |

`WorldMemory` exposes `recency_weight` and `adaptive_alpha` constructor parameters and passes decoded (v, a) through as before.

### B.2 Decoder calibration (`kokoro/decoder.py`, `training/train_probe.py`)

Thresholds now **travel with the probe checkpoint** (`thresholds` key, written by `compute_thresholds()` in the probe trainer; the existing `valence_arousal_probe.pt` was patched to carry the current calibration). `StateDecoder` reads them at load time and falls back to the module constants only for legacy checkpoints. The regex source-patch survives solely to keep the fallback constants in sync. A retrain can no longer silently invalidate every classification the system makes.

### B.3 Training pipeline (`training/split.py` new, `training/train.py`, `training/train_probe.py`)

- **`conv_disjoint_split()`**: greedy conversation-partition split. On the 10k set it yields **4658 train / 232 val / 5110 dropped, with zero shared conversations** (verified). It also *measures and logs* the legacy contamination (100%) so the bias of old numbers is documented, and `--allow-leaky-split` reproduces the old behaviour for controlled comparison. Both `train.py` and `train_probe.py` now use it by default.
- **`--holdout-conv-fraction`** in `data/construct_trajectories.py`: the by-construction fix — reserves a fraction of source conversations before generation and emits a separate `<out>_val.json` built exclusively from them. No trajectories wasted; use this for the next data regeneration.
- `vicreg_loss` raises a clear error on empty batches; arc-holdout OOD mode now composes with the clean split.

### B.4 Data (`data/low_arousal_pool.py` new, `data/arc_templates.py`, `data/construct_trajectories.py`)

- **Synthetic deep-low-arousal pool**: template-composed first-person sessions in the two missing circumplex regions — Q4-deep (exhaustion/numbness/shutdown, v∈[−0.85,−0.35], a∈[−0.80,−0.35]) and Q2-deep (serenity/deep rest, v∈[+0.30,+0.75], a∈[−0.80,−0.40]). Q2-deep is essential: without it, "deep low arousal" would become a proxy for negative valence and re-entangle the axes. Enabled with `--augment-low-arousal` (default 600 sessions).
- **Two new zones** (`shutdown`, `serene`) and **two new arousal-primary arcs** (`depressive_shutdown`: Q3 → deep Q4; `unwinding_to_serenity`: Q1 → deep Q2) that actually sample the new region.
- This is the maximum push available *within the existing data constraints*: it converts the documented "data ceiling, not a bug" limitation into a testable intervention. It does not replace a real low-arousal corpus (see Part E).

### B.5 Diagnostics (`diagnostics/common.py` new; four scripts repaired)

- `load_transition_model()` / `make_encoder()` / `build_embedding_cache()` — one correct, checkpoint-faithful loading path that honours `session_dim` and switches VAD features on iff the checkpoint used them.
- `topic_leakage`, `state_ablation`, `norm_drift`, `per_arc_val_loss` now load the production checkpoint (previously: hard crash) and slice embeddings to `[:state_dim]` wherever a 384-dim state is compared against a possibly-387-dim embedding (previously: shape errors or silently wrong math).
- `per_arc_val_loss` classifies the two new deep arcs as arousal-primary.
- **First real run of the repaired `state_ablation`: gap = +0.0001 → FLAG.** See Part C.

### B.6 Evaluation (`experiments/build_contexts.py`, `experiments/run_llm_eval.py`)

- **Condition C — recency baseline**: `build_contexts.py` emits `context_c` (the last top-k session summaries, newest first, no state summary) for every scenario. `run_llm_eval.py --with-recency-baseline` judges Kokoro (B) against C and reports the C-vs-B win rate and sign-test p separately. This is the confound-killer: if B beats A but not C, the honest conclusion is "recency suffices"; if B beats both, the world-model claim survives its strongest cheap competitor.
- **Position counterbalancing**: every judge now evaluates each pair in *both* presentation orders. A judge that names the same winner in both orders casts that vote; a judge that flips is recorded `position_consistent: false` and votes TIE. Positional preference — the dominant known bias of LLM judges — can no longer produce a verdict.
- **Statistics**: exact two-sided binomial sign test (ties excluded) and a 95% Wilson interval on the B win rate, printed in the summary and computable from `results.json`. Pure stdlib, preserving the torch-free Stage-2 design.
- `_InMemoryStore` (both copies) updated to the new retrieve contract (None query, [0,1] semantic scale, recency, adaptive alpha), so experiment and production scoring stay in lockstep.

### B.7 Tests

- The stale `test_retrieval` failure (required `state_vector`) is fixed by B.1.
- Three `test_transition` tests asserting the *removed* L2 normalization were rewritten to assert the actual contract (finite, non-zero, magnitude unconstrained — with the VICReg rationale in the docstring). **89/89 tests pass.**

---

## Part C — What the system can do now that it couldn't before

1. **Defend its headline result against the cheapest alternative.** The evaluation can now answer "is this better than just showing recent memories?" — previously it structurally could not.
2. **State results with uncertainty.** Win rates come with CIs and p-values; verdicts are position-debiased. A reviewer can no longer dismiss the eval in one sentence.
3. **Report honest training metrics.** The next retrain produces the first uncontaminated val loss / probe r / PR numbers in the project's history, with the legacy bias quantified alongside (100% contaminated val set under the old split).
4. **Know when its own mechanism is useless.** Adaptive alpha turns the "stable users don't benefit" finding from a discussion-section caveat into a shipped policy: the system now degrades gracefully to semantic retrieval exactly where the emotional axis was hurting it.
5. **Monitor the model it ships.** The diagnostic suite runs against the production checkpoint for the first time — and immediately earned its keep (state-ablation flag).
6. **Train toward genuinely low-arousal states.** The circumplex training coverage now extends to a ≈ −0.8 in both valence polarities.
7. **Operate as a sane RAG store** (optional state vectors, meaningful empty-query retrieval, deterministic recency ordering).

## Part C′ — The finding that must reshape the paper

The repaired state-ablation diagnostic shows the recurrent state contributes **nothing** to next-session prediction (gap 0.0001 vs 0.05 threshold). Combined with the retrieval design (which uses decoded (v,a), not the raw state) and the trend computation (which uses `arc_history`, not the raw state), the honest system description is:

> Kokoro's demonstrated value comes from **(1) per-session VAD-coordinate retrieval** and **(2) trend summarization over the decoded (v,a) sequence** — not from the 384-dimensional learned state as a *memory*. The transition model currently functions as a session-level (v,a) estimator.

This is still a publishable system (the mechanisms work and the long-history divergence examples are real), but the "world model" framing must be either (a) weakened to match, or (b) earned by a retrain that moves the ablation gap — LSTM/GRU transition (upgrade path already documented in `transition.py`), longer trajectories, and possibly an explicit state-utility term in the loss. The MLP-with-zero-cold-start + short (5–9 session) trajectories + split LayerNorm (which rescales the state to the same magnitude as the fresh session) is a plausible mechanistic story for why the state is ignorable.

---

## Part D — What to re-run, and what to expect

Ordered; each step's outputs feed the paper.

1. **Regenerate data** (CPU, minutes):
   `python -m data.construct_trajectories 10000 --augment-low-arousal --holdout-conv-fraction 0.2`
   → `trajectories_10k.json` + `trajectories_10k_val.json`, conversation-disjoint, arousal down to −0.8. Expect the two new arcs to appear in the arc distribution and zero zone-fallback for deep waypoints.
2. **Retrain transition model** on the new data (train file for train, `_val` file for val — or rely on the built-in disjoint split):
   `python -m training.train --data data/trajectories_10k.json --epochs 100`
   Expect: **val loss worse than 0.5087** (it's finally honest) — report it as the first uncontaminated number, alongside the legacy figure with the contamination caveat.
3. **Retrain probe** (`python -m training.train_probe`). Expect valence r to drop somewhat from 0.698 (leakage removed) and **arousal r to rise** (floor broken) — the headline data-intervention result. Thresholds are embedded in the checkpoint automatically.
4. **Re-run diagnostics** (now all runnable): `state_ablation` (the number to watch — if the gap stays ≈ 0 after retraining, adopt the weakened framing or move to the LSTM), `per_arc_val_loss` (watch the two new deep arcs), `topic_leakage`, `norm_drift`, `arc_separation`, `probe_generalization`.
5. **Rebuild contexts + run the eval with the new protocol**:
   `python experiments/build_contexts.py --long` then `python experiments/run_llm_eval.py --long --with-recency-baseline` (needs `GROQ_API_KEY`; judge calls are 6× per comparison now — 2 orders × 3 judges).
   Expect the A-vs-B win rate to move toward 50–70% once position bias is removed (counterbalancing typically deflates one-sided results), and the **B-vs-C margin to be the paper's new headline** — smaller than 83% but defensible. Also run with `adaptive_alpha=True` in `build_contexts.py`'s WorldMemory as an ablation arm; expect it to recover most A-wins on stable arcs.
6. **Scale the standard set** toward ~50 scenarios before claiming significance: with the sign test now built in, 20 scenarios can only detect effects ≥ ~75/25.

## Part E — What would push further with external resources

- **Real multi-session user data** (or platform-derived synthetic approximations): the only way to learn non-templated dynamics; would also let the LSTM upgrade be evaluated for real (state-ablation gap on real longitudinal data is the decisive world-model test).
- **A genuine low-arousal corpus** (e.g. depression/fatigue support forums with consent, or DailyDialog's neutral register, human-annotated for VAD): replaces the synthetic pool; expect a further arousal-r gain and better probe generalization to deployment text.
- **Human evaluation**: even 3 raters × 50 scenarios with the counterbalanced protocol would decouple the result from LLM-judge idiosyncrasy; report Cohen's/Fleiss' kappa against the LLM panel to validate the cheap judge for future iterations.
- **A stronger production judge suite**: current three Groq models correlate (two are Llama family); one non-Llama, non-Qwen judge (e.g. Claude) would strengthen the cross-model claim.

## Part F — Paper updates required (delta to research_report.md / paper_notes.md)

1. **Methods/split**: describe the conversation-disjoint split; disclose that all previously reported val metrics used a 100%-contaminated val set; re-report §6/§7 tables from the retrain.
2. **§7 diagnostics**: add the state-ablation result and the weakened world-model framing (or the LSTM retrain that rebuts it). This cannot be omitted — it is the paper's most attackable point and pre-empting it is worth more than the 83% headline.
3. **§8 evaluation**: new judging protocol (counterbalanced 3-judge panel, position-consistency reporting), statistics (sign test + Wilson CI), and the recency baseline Condition C with its result.
4. **§3 data**: low-arousal augmentation, the two new arcs, updated coverage figure (circumplex scatter before/after augmentation makes a strong figure).
5. **§2/§8 retrieval**: normalized-axis scoring formula, recency term, and adaptive alpha (with its ablation). The adaptive-alpha mechanism is a new, citable contribution: *dispersion-gated hybrid retrieval*.
6. **Reproducibility**: note `--allow-leaky-split` exists solely to reproduce legacy numbers.

---

*All changes verified on 2026-07-03: pytest 89/89; `python -m kokoro.retrieval` 36/36; `python -m kokoro.memory` 16/16; `python -m kokoro.decoder` 35/35; `build_contexts.py --test` end-to-end incl. `context_c`; `state_ablation` end-to-end on the production checkpoint; disjoint split verified zero shared conversations (4658/232/5110).*
