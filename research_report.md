# Kokoro: Emotionally-Aware Memory Retrieval for AI Companions
## Comprehensive Research Report — v2 (GRU world model, honest evaluation)

**Project:** Kokoro — Emotional Trajectory Memory for AI Companions
**Report type:** Full technical and empirical report (single source of truth for the paper)
**Date:** July 2026
**Status:** Complete system — v2 world model trained, all diagnostics re-run, debiased evaluation complete
**Supersedes:** the June 2026 report (v1). All v1 headline numbers were computed on a contaminated validation split and/or a position-biased judging protocol; wherever a v1 number is quoted here it is labeled *(legacy, biased)* and kept only for contrast. `improvements_report.md` documents the two improvement cycles; `paper_updates.md` maps this report onto paper sections; `codebase_map.md` is the file-level navigation companion.

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Training Data](#3-training-data)
4. [Training Objective and Procedure](#4-training-objective-and-procedure)
5. [Project History: Collapse, Leakage, and the Dead State](#5-project-history-collapse-leakage-and-the-dead-state)
6. [Diagnostic Results (v2 checkpoint, conversation-disjoint validation)](#6-diagnostic-results)
7. [Evaluation Pipeline](#7-evaluation-pipeline)
8. [Evaluation Results](#8-evaluation-results)
9. [Discussion](#9-discussion)
10. [Known Limitations](#10-known-limitations)
11. [Repository Guide and Reproduction](#11-repository-guide-and-reproduction)
12. [Citations](#12-citations)

---

## 1. Motivation and Problem Statement

AI companion systems that maintain long-term memory of users face a fundamental gap: when retrieving relevant past sessions to provide context, purely semantic (topic-based) retrieval ignores the user's emotional trajectory. A user whose keyword "work stress" appears both in a period of acute burnout and in a later period of healthy challenge will receive the same retrieved memories regardless of their current emotional state. Most companion systems remember *facts*; none remember *how the user has been doing*.

Kokoro addresses this with three coupled mechanisms:

1. **A learned world model** — a recurrent state, updated once per session, that tracks the user's emotional trajectory across sessions.
2. **Emotionally-aware retrieval** — stored sessions are ranked by a blend of topical relevance and emotional proximity to the user's *current* decoded state, so a recovered user asking about "the big sprint" retrieves their recovery-phase memories, not their burnout-phase ones.
3. **State summary injection** — a natural-language description of the trajectory ("declining across the past 7 sessions, currently negative and low-energy") placed in the companion LLM's system prompt.

The research questions, in the order the evidence now supports them:

- **RQ1 (representation):** Can a compact recurrent state trained on synthetic emotional-arc data (a) demonstrably carry trajectory information beyond the current session, and (b) decode into interpretable circumplex coordinates (valence, arousal)?
- **RQ2 (forecasting):** Does that state predict where the user is *heading* better than trivially repeating where they *are*?
- **RQ3 (response quality):** Do the retrieval and summary mechanisms produce companion responses that blind judges prefer over (i) semantic-only retrieval and (ii) a recency baseline?

RQ1 and RQ2 now have strong affirmative answers (§6). RQ3 has a nuanced answer that overturned this project's own earlier claims: against semantic-only retrieval, a statistical dead heat at n=40 under a debiased protocol; against the recency baseline, a consistent directional win (14–5 with 21 ties, p=0.064). §8 details why the earlier win rates (83.3%, then 45–54.5%) were substantially artifacts of judging protocol.

---

## 2. System Architecture

Pipeline at a glance (see `codebase_map.md` for the full ASCII diagram and per-file tracebacks):

```
session turns ─► SessionEncoder (MiniLM + VAD lexicon, 387-d)
                       │
       state_t ────► TransitionModelGRU ────► state_{t+1}  (384-d GRU hidden)
                       │                          │
                       │                    StateDecoder (linear probe 384→2)
                       │                          │  (valence, arousal) + trend + summary
                       ▼                          ▼
                 StateStore (SQLite)        MemoryStore (ChromaDB)
                 state + arc_history        session text + emb + (v,a)
                                                  │
                    get_context(msg) ─► hybrid retrieval ─► LLM system prompt
```

### 2.1 SessionEncoder (`kokoro/encoder.py`, `kokoro/vad.py`)

Encodes a completed session (list of `{role, content}` turns) into a fixed vector.

- Base: `all-MiniLM-L6-v2` [Reimers & Gurevych, 2019] — 384-dim sentence embeddings, weighted mean pooling over turns with **user turns weighted 2.0×** (user words carry the primary affect signal).
- ParlAI artifact decoding (`_comma_` → `,` etc.) is encoder responsibility, so any data source produces clean embeddings.
- **VAD features** (production configuration): a 145-word curated subset of the Warriner/NRC-VAD lexicon [Warriner et al., 2013] scores user turns for (valence, arousal, dominance), appended as 3 extra dims → **387-dim output**. Rationale: the frozen MiniLM embedding has a cosine arousal gap of only ~0.112 between high- and low-arousal sessions, vs ~1.43 for the lexicon channel (≈13× stronger signal). The lexicon dims serve two roles: transition-model input, and training targets for the auxiliary trajectory head (§4). They are **stripped before storage** — the semantic retrieval axis uses pure 384-dim MiniLM.

### 2.2 TransitionModelGRU (`kokoro/transition.py`) — the world model

The v2 production model. The central design decision: **the user state and the next-session prediction are different objects.**

```
emb (387) → LayerNorm → GRUCell(387 → 384) → h'   (the user state)
h' → Linear(384→512) → GELU → Dropout → Linear(512→384) → z   (prediction head)
h' → Linear(384→2) → (v̂, â)_{t+1}                  (auxiliary trajectory head)
```

- `forward(state, emb) → h'` — the persisted state. GRU gating bounds every coordinate in (−1, 1), so state magnitude is stable over arbitrarily long deployments (measured: 1.00× norm ratio over 50-session rollouts, §6.6). Cold start is the zero vector.
- `predict(state) → z` — unconstrained-magnitude prediction of the next session's MiniLM embedding; this is what the VICReg objective shapes, so the variance target (std ≥ 1) never fights the bounded state.
- ~1.286M parameters. Public API identical to the legacy MLP (`forward`, `initial_state()`, `STATE_DIM=384`).
- `load_transition_checkpoint(path)` auto-detects architecture (`model_config["arch"]`, else state-dict keys) — one loading path used by WorldMemory, all diagnostics, the probe trainer, and the demo. `predict_from_state(model, state)` gives arch-agnostic predictions (GRU head; identity for the legacy MLP, whose state *was* its prediction).

**Why the legacy MLP failed** (kept loadable for provenance): its state was literally its prediction of the next session embedding. Any trajectory information not expressible in MiniLM space was discarded every step, and the measured state-ablation gap was **+0.0001** — the "world model" was a session-only encoder. §5.3.

### 2.3 StateDecoder (`kokoro/decoder.py`)

- Linear probe `Linear(384 → 2)` maps the state to (valence, arousal). Deliberately linear: it tests *decodability* of the representation, not probe capacity.
- Trend = least-squares slope of valence over `arc_history` (last ≤20 decoded (v,a) pairs from StateStore); classified improving/declining/stable at ±0.02.
- Natural-language summary composed from (valence class, arousal class, trend) — e.g. *"User has maintained a consistently negative and low-energy state across the past 7 sessions, suggesting persistent low mood or withdrawal."*
- **Classification thresholds travel inside the probe checkpoint** (`thresholds` key, percentiles of the val-set prediction distribution, written by the probe trainer). A retrain can no longer silently invalidate classification; module constants exist only as a fallback for legacy checkpoints. Current calibration: valence positive > +0.060 / negative < −0.429; arousal activated > +0.328 / low-energy < −0.019.
- Warm-up gate: no summary until `session_count ≥ 3`.

### 2.4 MemoryStore (`kokoro/retrieval.py`) — hybrid retrieval

ChromaDB-backed per-user session store. Retrieval score:

```
score_i = (1−γ) · [ α′·sem_i + (1−α′)·emo_i ] + γ·recency_i
```

- **Semantic axis** `sem_i`: cosine(query embedding, stored MiniLM embedding), **rescaled from [−1,1] to [0,1]** so α blends commensurable quantities. When no query message exists (e.g. `get_context("")` after an update), the axis is disabled rather than fed a bogus vector.
- **Emotional axis** `emo_i = 1 − ‖(v,a)_current − (v,a)_i‖₂ / √8`: L2 proximity in the decoded circumplex. This is deliberately *not* state-to-state cosine — a key v1 discovery (§5.2) was that all state vectors lie in a tight angular cone (cos ≈ 0.98–1.00 regardless of emotional phase), making state-cosine useless for discrimination, whereas the decoded 2-D coordinates separate a burned-out session (v=−0.76) from a recovered one (v=+0.25) cleanly. The relevant similarity space is the human-interpretable circumplex, not the model's internal geometry.
- **Recency axis** `recency_i = exp(−rank_age / τ)` (τ=5 sessions; newest = 1.0), weight γ (default 0 — off — so published comparisons stay clean; a deployed system should not lose to `list[-3:]`). Storage timestamps carry a monotonic tiebreaker so same-tick inserts order correctly.
- **Adaptive alpha** (`adaptive=True`): `α′ = adaptive_alpha(α, stored (v,a))`. When the stored history is emotionally homogeneous (mean L2 dispersion from centroid < 0.15) the emotional axis carries no ranking information and only injects decoder noise, so α′ → 1.0 (pure semantic); above dispersion 0.45 the configured α is kept; linear in between. This converts the v1 evaluation finding "stable users don't benefit and sometimes get hurt" into a shipped per-user policy. Not enabled in the reported evaluation (kept as an ablation arm).
- `state_vector` is optional at insert time — the store works as a plain semantic RAG index. Results expose per-axis scores and `effective_alpha` for auditability. Defaults: top_k=5, α=0.6.

### 2.5 StateStore (`kokoro/store.py`)

SQLite (WAL) persistence per user: state vector (raw bytes + dim check), `session_count`, latest (v,a), `arc_history` (JSON, capped at 20 pairs, oldest first), ISO-8601 timestamps. Atomic per-call connections; safe for multi-process readers.

### 2.6 WorldMemory (`kokoro/memory.py`) — public API

```python
memory = WorldMemory(user_id="alice")          # optional: alpha, top_k, recency_weight,
memory.update(session_turns)                   #   adaptive_alpha, retrieval_fn, paths
ctx = memory.get_context(user_message)
# ctx: state_summary, relevant_memories, valence, arousal, trend, session_count, ready
```

`update()`: encode → GRU step → decode (v,a) → persist state+history → index session (MiniLM slice, decoded coordinates, summary text = first 200 chars of user turns). `get_context(msg)`: decode current state → retrieve hybrid top-k (decoded (v,a) supplied to the emotional axis) → return context dict. `predict_next()`: decode the current state as a next-session forecast (the training objective orients the state toward session t+1). Checkpoint's `session_dim` auto-configures the encoder's VAD setting. `retrieval_fn` plugs in external vector stores.

---

## 3. Training Data

### 3.1 Source pool

**EmpatheticDialogues** [Rashkin et al., 2019]: ~19k multi-turn conversations, each labeled with one of 32 emotions. `data/prepare.py` maps each label to circumplex coordinates (Russell 1980; Posner et al. 2005) — e.g. `joyful`(+0.85,+0.65), `devastated`(−0.80,−0.30), `terrified`(−0.80,+0.80). These coordinates select conversations during trajectory construction; they are never model supervision directly (though the session-level lexicon VAD, a noisy correlate, supervises the auxiliary head).

**The arousal floor and its fix.** ED's deepest low-arousal labels bottom out at ≈ −0.30 arousal. Arc waypoints specifying genuine deactivation (exhausted −0.6, shutdown −0.7) silently fell back to ≈ −0.25 sessions — the model had *never seen* a genuinely deactivated session labeled as such, the documented cause of the v1 arousal-probe ceiling (r=0.542). `data/low_arousal_pool.py` adds 600 template-composed first-person sessions in the two missing regions:

- **Q4-deep** (65%): exhaustion, numbness, depressive shutdown — v ∈ [−0.85, −0.35], a ∈ [−0.80, −0.35]
- **Q2-deep** (35%): deep calm, serenity, rest — v ∈ [+0.30, +0.75], a ∈ [−0.80, −0.40]

Both polarities are required so "deep low arousal" does not become a proxy for negative valence, which would re-entangle the axes. Ground-truth arousal coverage is now **[−0.80, +0.80]**. The texts are synthetic and templated — disclosed as such; a real low-energy corpus remains the honest upgrade (§10).

### 3.2 Arc templates (`data/arc_templates.py`) — 16 types

Each template is a sequence of circumplex zones (center ± tolerance 0.25) grounded in the emotional-trajectory literature (Bonanno resilience arcs, Tedeschi & Calhoun post-traumatic growth, Clark & Beck GAD patterns, Kübler-Ross/Stroebe grief):

| Valence-primary | Mixed | Arousal-primary (valence ~flat, arousal moves) |
|---|---|---|
| gradual_decline, slow_recovery, grief_arc, stable_positive, stable_negative, post_traumatic_growth, social_confidence_growth | acute_stress_stabilization, chronic_low_grade_anxiety, weekend_oscillation, relapse_dip | excitement_to_contentment (Q1→Q2), anxiety_to_depression (Q3→Q4), depression_to_anxiety (Q4→Q3), **depressive_shutdown (Q3→deep-Q4)**, **unwinding_to_serenity (Q1→deep-Q2)** |

The five arousal-primary arcs exist because without them, next-session prediction is solvable by tracking valence alone and the arousal dimension receives no gradient. The two **deep** arcs (new in v2) reach the augmented region; without augmentation their deep waypoints fall back to shallow sessions via tolerance relaxation. A v1 bug worth remembering: `depression_to_anxiety` originally contained an `unsettled(−0.1,+0.2)` waypoint whose +0.5 valence swing re-entangled the axes; all its waypoints now stay at valence ≤ −0.4.

### 3.3 Trajectory construction (`data/construct_trajectories.py`)

Weighted-samples a template, picks pool sessions inside each zone (one tolerance-relaxation retry of +0.15), never reuses a conversation within a trajectory, soft-caps each conversation at 8 uses across trajectories, and assigns jittered week offsets. Output JSON per trajectory: `{trajectory_id, arc_name, n_sessions, sessions:[{session_index, week_offset, conv_id, emotion_label, valence, arousal, turns}]}`.

### 3.4 The leakage disclosure and the disjoint split

**Every v1 validation metric was computed on a contaminated split.** Because each source conversation appears in up to 8 trajectories and the v1 split was at the trajectory level, a measurement on the v1 10k set showed **100% of validation trajectories shared at least one source conversation — identical text, identical cached embedding — with training.** Val loss 0.5087, probe r 0.698/0.542, PR 339.7, and arc separation 3.111 are all optimistically biased to an unknown degree.

Two fixes ship:
- `training/split.py::conv_disjoint_split()` — greedy conversation-partition split of an existing trajectory file (assign each trajectory to the side its committed conversations belong to; straddlers dropped), plus `leakage_report()` quantifying the contamination. `--allow-leaky-split` reproduces the legacy behavior for controlled comparison only.
- **Leakage-free by construction** (used for v2): `construct_trajectories --holdout-conv-fraction 0.2` reserves 20% of source conversations *before* generation and builds a separate val file exclusively from them.

### 3.5 The v2 dataset of record

`trajectories_10k_v2.json` + `trajectories_10k_v2_val.json`, generated with `--augment-low-arousal --holdout-conv-fraction 0.2`, seed 42:

| Property | Train | Val |
|---|---|---|
| Trajectories | 10,000 | 2,000 |
| Sessions/trajectory | 7.6 (6–9) | 7.6 (6–9) |
| Unique source conversations | 13,464 | (held-out 20% of pool) |
| **Shared conversations across splits** | **0** | **0** |
| Ground-truth arousal range | [−0.80, +0.80] | same |
| Synthetic low-arousal session usages | 3,772 | (proportional) |
| Arc types | all 16 | all 16 |

---

## 4. Training Objective and Procedure

### 4.1 The objective (`training/train.py`)

Rollout: from the zero state, `h_{t+1} = GRU(h_t, e_t)`; prediction `z_t = head(h_t)` targets `e_{t+1}[:384]` (normalized). Over a mini-batch of 32 trajectories (~213 valid steps):

```
L = 25·L_sim + 25·L_var + 1·L_cov + 50·L_util + 10·L_vad
```

- **L_sim** (VICReg invariance): `mean(1 − cos(z, e_next))`. Cosine, not MSE — MSE against unit-norm targets recreates a soft unit-sphere constraint that deadlocks against the variance term (the exact failure of training Run 1, §5.1).
- **L_var** (VICReg variance): `mean(ReLU(1 − std(z, dim=0)))` — the anti-collapse force. Operates on unconstrained z, never fighting the bounded state.
- **L_cov** (VICReg covariance): off-diagonal covariance penalty, decorrelating dimensions so arousal cannot be re-expressed as a valence variant.
- **L_util — the state-utility margin loss (the ablation-gap regularizer, new in v2).** For every *warm* step (t ≥ 1), the same session is also pushed through the model with a **zeroed state**, giving an ablated prediction z⁰. Then

  ```
  L_util = mean_warm[ ReLU( 0.2 − ( cos(z, e_next) − cos(z⁰, e_next) ) ) ]
  ```

  is zero only when history-carrying predictions beat history-free ones by ≥ 0.2 cosine. This optimizes *exactly* the quantity `diagnostics/state_ablation.py` measures; a model that ignores its recurrent memory cannot minimize it. Cold-start steps are excluded (there the two paths coincide by construction).
- **L_vad — auxiliary trajectory forecasting (new in v2).** `MSE(vad_head(h_t), (v,a)^lex_{t+1})`, targets being the lexicon VAD dims already present in the next session's embedding (dims 384:386). Rationale: the next session's *text* is dominated by unpredictable topical content, so L_sim alone gives history little leverage; the next session's *emotional position* follows the observed arc — this is the task that makes memory worth carrying, and it is the quantity the deployed system actually consumes.

**Checkpoint selection: `val_loss − ablation_gap`.** Both are in cosine units; this picks the epoch with the best prediction quality *attributable to state use*. The per-epoch gap is logged in `*.history.json`.

### 4.2 The decisive ablation: architecture is not enough

Two v2 training runs, identical data and architecture:

| Run | util_weight / margin | vad_weight | State-ablation gap (train metric) | Val cosine loss |
|---|---|---|---|---|
| GRU, weak incentives | 10 / 0.1 | 0 | **+0.005** | 0.503 |
| **GRU, final** | 50 / 0.2 | 10 | **+0.175–0.187** | 0.554 |
| *(MLP legacy, for scale)* | — | — | *+0.0001* | *0.503 (leaky)* |

The weak-incentive GRU converged to essentially the same dead state as the MLP. **On near-Markovian synthetic arcs, recurrence alone does not produce state use — the objective must demand it.** The ~0.05 increase in val prediction loss is the explicit, quantified price of genuine trajectory dependence (the utility term deliberately trades one-step prediction sharpness for state reliance, including making cold-start predictions worse — which is what "relying on memory" means).

### 4.3 Procedure and efficiency

- AdamW lr 1e-3 → 1e-5 (cosine), weight decay 1e-4, grad clip 1.0, batch 32 trajectories, 60 epochs, CPU.
- **Vectorized rollouts** (`rollout_batch`/`_pad_batch`): trajectories padded to (B, T_max, 387) and stepped in parallel with validity masks — ~8 batched cell calls per step-batch instead of ~213 scalar forwards. 16,773 unique sessions pre-encoded once (13s). Full retrain ≈ 25 minutes on CPU.
- Run of record: best epoch 43 (selection score 0.3675 = val 0.5544 − gap 0.1869); final-epoch val 0.5517, gap 0.1752; PR 37.8.
- **Probe training** (`training/train_probe.py`): rolls all trajectories through the frozen model, collects (state, v, a) per session, trajectory-level split honored via `--val-data`; fits the linear probe (MSE, 100 epochs); computes percentile thresholds and **saves them into the probe checkpoint**; patches the decoder's fallback constants; runs a polar-arcs sanity check (stable_positive +0.327 vs stable_negative −0.314 — PASS).

---

## 5. Project History: Collapse, Leakage, and the Dead State

Three generations of failure, each caught by a diagnostic this repo now ships. Kept because the paper's methods narrative depends on them.

### 5.1 Generation 1: dimensional collapse (v0 → v1)

The original model (cosine objective + L2-normalized output + joint LayerNorm on [state‖emb]) collapsed to an effectively 1-dimensional state: **PR 1.4/384**, decoder thresholds 0.012 apart (both negative — the model always said "slightly negative"), probe r ≈ 0 on both axes, identical predictions for polar-opposite arcs. Root causes, in severity order: (1) cosine prediction loss has zero-gradient collapsed minima; (2) L2-normalized output caps per-dim std at 1/√384 ≈ 0.051, irreconcilable with VICReg's std ≥ 1; (3) joint LayerNorm let the high-norm state (~14–19) drown the unit-norm session embedding — the model ignored session content; (4) no arousal-primary arcs — no gradient could separate the axes. Fix sequence: VICReg with **cosine** similarity (the first attempt used MSE and deadlocked at val 0.9609 for 36 epochs — MSE against unit targets recreates the sphere constraint), L2-norm removal, split LayerNorm, three arousal-primary arcs. Result: PR 1.4 → 22.1 → 339.7, probe r 0.698/0.542 *(all on the leaky split)*.

### 5.2 The cone discovery (v1)

Even at PR 339.7, all state vectors lie in a tight angular cone (pairwise cos ≈ 0.98–1.00): the mean vector sits far from the origin, so variation around it — however high-dimensional — barely moves the direction. State-cosine is therefore blind for retrieval. The fix that defines the production emotional axis: score in decoded (v,a) space with L2 distance. General lesson: **for retrieval, the relevant similarity lives in the interpretable coordinate space, not the internal representation space.** (v1 also fixed six pipeline bugs, the most serious being that Condition B's state summary was silently never injected — the mechanism under test was absent from its own evaluation.)

### 5.3 Generation 3: the dead state and the contaminated split (v1 → v2)

Cycle-1 audit findings (2026-07-03, `improvements_report.md` Parts A–F):

- Four diagnostics could not even load the production checkpoint (they dropped `session_dim` when reconstructing the model). Once repaired, `state_ablation` reported **gap +0.0001**: zeroing the recurrent state at every step changed prediction loss by nothing. The MLP "world model" was a session-only encoder; the trajectory information the paper attributed to the state actually came from per-session decoded (v,a) plus the arc-history slope.
- The train/val split was **100% contaminated** at the conversation level (§3.4).
- The evaluation had no recency baseline, no statistics, and a fixed presentation order for judges.

Cycle 2 (this report's subject) fixed all three: GRU + state-utility objective + auxiliary forecasting head (§4), leakage-free data (§3.5), debiased evaluation with a recency control (§7). Cycle 1 also shipped the retrieval upgrades (normalized axes, recency term, adaptive α, None-query handling), checkpoint-embedded decoder thresholds, and the low-arousal data pool.

---

## 6. Diagnostic Results

All numbers: v2 checkpoint (`transition_v2.pt`, epoch 43) + v2 probe, on the **conversation-disjoint** validation file, CPU. Every diagnostic is runnable via `python -m diagnostics.<name>` with current defaults.

### 6.1 State ablation (`state_ablation.py`) — RQ1(a), the headline

Rolls each val trajectory twice: normally, and with the state zeroed at every step.

| | Normal | Ablated (no history) | Gap |
|---|---|---|---|
| Mean cosine prediction loss | 0.5570 | 0.7145 | **+0.1575** |

Threshold 0.05; legacy MLP scored +0.0001. Zeroing memory demonstrably degrades prediction — the state carries trajectory information. Caveat handled honestly in §9: the gap is *trained* (Goodhart risk), which is why 6.2 matters.

### 6.2 Prediction vs persistence baseline (`eval_worldmodel.py`) — RQ2, the untrained corroboration

Model prediction cos(predict(h_t), e_{t+1}) vs naive baseline cos(e_t, e_{t+1}) ("tomorrow feels like today"):

| Metric | Value |
|---|---|
| Model mean cosine | 0.4424 |
| Persistence baseline | 0.2602 |
| **Delta** | **+0.1822** |
| Steps where model > baseline | **90.1%** |
| Per-arc delta | positive on **all 16 arcs** (+0.135 … +0.226) |

This quantity is *not* in the training loss (the loss shapes z against e_{t+1}, but the baseline comparison is free), making it the independent check on 6.1.

### 6.3 Linear probe (`train_probe.py` output) — RQ1(b)

| Axis | v2 (honest split) | v1 (leaky split) |
|---|---|---|
| Valence Pearson r | **0.7695** | 0.698 |
| Arousal Pearson r | **0.6722** | 0.542 |

Both axes improved *despite* removing the leakage advantage — attributable to the GRU + auxiliary head (which explicitly shapes (v,a) information into the state) and the broken arousal floor. Val MSE 0.096. Thresholds embedded in the checkpoint (§2.3).

### 6.4 Per-arc prediction loss (`per_arc_val_loss.py`)

Arousal-primary arcs now average **lower** loss than valence-primary arcs (0.5427 vs 0.5591, gap **−0.0165**; v1 flagged the opposite). The two new deep arcs are the two best-predicted overall (unwinding_to_serenity 0.516, depressive_shutdown 0.523); worst is social_confidence_growth (0.602). The arousal axis is no longer the model's weak side.

### 6.5 Retrieval-axis independence (`topic_leakage.py`)

Spearman rank correlation between emotional-axis and semantic-axis retrieval orderings over 150 queries: mean **0.041**, median 0.018, 90th pct 0.233 (flag threshold 0.8). The emotional axis is not semantic retrieval in disguise. (Implementation note: uses a numpy Spearman — importing scipy before sentence-transformers segfaults on the reference environment.)

### 6.6 Norm stability (`norm_drift.py`)

Mean ‖state‖ over 50-session rollouts: 18.58 at step 5 → 18.67 at step 50, ratio **1.00×** (flags at >10× or <0.1×). With the GRU this is structural — the hidden state is bounded — retiring the v1 concern about unconstrained MLP outputs drifting over long deployments.

### 6.7 Arc separation (`arc_separation.py`)

Final-state clustering by arc type (between-arc / within-arc cosine distance): model **1.227** vs last-session baseline 1.088, EWMA 1.115, shuffled-label control 1.0035 ± 0.004 (−18.2% from real labels — the structure is real). Far below the legacy 3.111, for two stated reasons: the legacy number was computed on contaminated data, and bounded GRU states compress cosine geometry. Neither matters operationally — production retrieval ranks in decoded (v,a) space, not state-cosine space (§5.2) — but the paper should report the drop rather than hide it.

### 6.8 Probe generalization to naturalistic text (`probe_generalization.py`)

24 handcrafted companion-AI conversations (nothing like EmpatheticDialogues stylistically). Quadrant means: Q1 (+0.34, +0.10), Q2 (−0.57, +0.08), Q3 (−0.54, −0.37), Q4 (+0.35, −0.17); valence ordering Q1>Q3 **PASS**, arousal ordering Q2>Q4 **PASS**; large majority of scenarios directionally correct, including deep-deactivation cases decoding to arousal ≈ −0.5 (impossible pre-augmentation; the v0 model's output range on this set was 0.034 wide).

### 6.9 Longitudinal tracking simulation (`longitudinal_sim.py`)

20 simulated users over full arcs: warm-phase valence MAE **0.247** (+20.9% better than cold-start), and **10/10 mid-trajectory arc switches detected with mean lag 0.7 sessions** — the property a deployed companion actually needs ("notice the user turned a corner roughly one session after it happens").

### 6.10 Capacity metrics

Participation ratio **37.8/384** (target >20). Not comparable to the leaky-MLP 339.7: different data, and PR over bounded gated states measures something different from PR over free prediction vectors. The collapse-detection role of PR (≫1) is satisfied.

Also available: `arc_ood_eval.py` (train with `--holdout-arcs`, evaluate on withheld arc types) — wired but not part of the v2 run of record.

---

## 7. Evaluation Pipeline

### 7.1 Conditions

| Condition | Retrieval | State summary |
|---|---|---|
| **A** — semantic baseline | α=1.0 (pure semantic) | no |
| **B** — Kokoro | α=0.6 hybrid | yes |
| **C** — recency baseline | last k=3 session summaries | no |

C is the critical control introduced in v2: the long-history scenarios are two-phase by design (old emotion → new emotion, same topic keywords), so *any* recency heuristic also pulls phase-current memories. Without C, "beats semantic-only" cannot be attributed to the world model rather than to trivial recency.

### 7.2 Scenario sets (`experiments/scenarios.py`, `scenarios_long.py`)

- **Standard**: 20 hand-written scenarios, 4–5 sessions, 8 arc types, each ending in an ambiguous message ("still here lol") with an `expected_awareness` rubric.
- **Long-history**: 20 scenarios, 10–11 sessions, explicit Phase-1 (distress, keyword-dense) → Phase-2 (recovery, same topic) structure; the new message reuses Phase-1 keywords to create maximal tension between topically-matched-but-stale and emotionally-current memories.

### 7.3 Two-stage execution

**Stage 1** (`build_contexts.py`, torch process, no network): per scenario, a fresh WorldMemory (v2 checkpoint) ingests all sessions via `update()`; contexts A (α=1.0), B (α=0.6 + summary), C (last-3) are extracted with retrieval-divergence stats → `contexts[_long].json`. A pure-numpy `_InMemoryStore` mirrors production scoring exactly (normalized axes, live decoder for (v,a), recency/adaptive parity).

**Stage 2** (`run_llm_eval.py`, network process, no torch): responses generated by `llama-3.3-70b-versatile` (temp 0.7, ≤300 tokens, Groq); judged by **three models** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b`) at temp 0.0.

### 7.4 Debiasing protocol (new in v2)

1. **Position counterbalancing**: every judge scores each pair in *both* presentation orders (1=X,2=Y and 1=Y,2=X). Consistent winner → that vote; verdict flips with order → the judge votes TIE and is recorded `position_consistent=false`; one-order-TIE → the decisive verdict, flagged order-sensitive.
2. **Reasoning-trace hygiene**: `<think>…</think>` blocks stripped before verdict parsing, judge token budget 2048. **Erratum this fixed:** the previous run's parser read qwen3-32b's chain-of-thought as its verdict, which is the real explanation for the "qwen votes Kokoro on every scenario" anomaly flagged in the v1 write-up — a parser artifact, not a judge opinion; it also invalidates the v1 per-judge analysis.
3. **Statistics**: majority vote over the three debiased judges; exact two-sided binomial sign test on decisive verdicts; 95% Wilson interval on win rates.

---

## 8. Evaluation Results

**Date:** 2026-07-03. v2 checkpoint throughout. Files: `experiments/results.json`, `results_long.json`.

### 8.1 Headline tables

| Comparison | Standard (20) | Long-history (20) | **Pooled (40)** |
|---|---|---|---|
| Kokoro (B) vs semantic-only (A) | B 8 / A 6 / TIE 6 (p=0.79) | B 4 / A 8 / TIE 8 (p=0.39) | **B 12 / A 14 / TIE 14 (p=0.85)** |
| Kokoro (B) vs recency (C) | B 7 / C 3 / TIE 10 (p=0.34) | B 7 / C 2 / TIE 11 (p=0.18) | **B 14 / C 5 / TIE 21 (p=0.064)** |
| B win rate vs A, Wilson 95% | 40% [22, 61] | 20% [8, 42] | 30% |
| Scenarios where A/B retrieve different memories | 12/20 (overlap 80%) | 15/20 (overlap 65%) | 27/40 |
| Judge agreement (unanimous/majority/none) | 12 / 8 / 0 | 7 / 12 / 1 | — |
| **Position-inconsistent individual judgings** | 18/60 | 22/60 | **40/120 (33%)** |

Subset views: standard set, retrieval-diverging scenarios: **B 7 / A 3 / TIE 2**; long set, diverging: A 7 / B 3 / TIE 5 (the sets cancel to the pooled dead heat: 10/10/7). Against recency within long-set diverging scenarios: B 5 / C 2 / TIE 8.

### 8.2 What must be said plainly

1. **The previous win rates do not survive debiasing and are retracted.** The 83.3% (6 hand-picked scenarios, single judge, fixed order), and the subsequent 45%/54.5% (20 scenarios, fixed order, mis-parsed third judge) were substantially protocol artifacts: one-third of all individual judgings in the current run flip with presentation order, and one of three judges had been voting via a parser bug.
2. **Kokoro vs semantic-only is a statistical dead heat at n=40.** The standard-set divergence subset favors B (7–3–2); the long-set divergence subset favors A (7–3–5) — emotionally-current memories sometimes lose to semantically precise ones on the judges' "acknowledges the user's history" criterion. RQ3(i) is unresolved at this sample size.
3. **Kokoro beats the recency baseline in both sets independently** — pooled 14–5 with 21 ties, p=0.064 on 19 decisive comparisons. This is the defensible response-level claim: the emotional mechanism contributes something that neither topical matching nor "just show recent sessions" provides, and recency is the comparison a deployed system actually faces.
4. **TIE is the modal outcome** (14/40 vs A; 21/40 vs C). At 2–3-sentence response length, the judge panel frequently cannot distinguish conditions at all — itself a finding about LLM-judge sensitivity, and the reason resolving RQ3 needs a larger scenario set and human raters rather than another LLM pass.
5. Mechanical health: retrieval divergence increased under the v2 model (12/20 and 15/20 scenarios, vs 7/20 and 11/20 under the MLP) — the emotional axis is doing more re-ranking; judge panels reached at least majority consensus in 39/40 scenarios.

---

## 9. Discussion

**Where the value demonstrably is.** The strong results are representational and predictive: a state that provably carries history (ablation +0.158, corroborated by the untrained +0.182 advantage over persistence on 90% of steps), decodes both circumplex axes well on honest data (r 0.77/0.67), tracks simulated users with 0.7-session arc-change lag, and stays bounded forever. The response-preference test — the weakest link methodologically — is a dead heat vs semantic-only and a directional win vs recency. The paper should be built on the former and honest about the latter.

**Trained vs emergent state use.** The ablation gap exists because the objective demands it; the weak-incentive GRU run (+0.005) proves recurrence alone buys nothing on this data. Two readings, both worth stating: (a) Goodhart concern — the gap is optimized, so its value as *evidence* rests on the persistence-baseline result, which is not trained; (b) the deeper reading — synthetic template arcs are near-Markovian (the current session's zone almost suffices to predict the next zone), so history is cheap to ignore. **Whether real users' emotional dynamics reward memory more richly is exactly the question real longitudinal data would answer**, and the state-utility objective guarantees the architecture will exploit such structure where it exists.

**Why emotionally-current memories sometimes lose.** Judges reward specific autobiographical detail. A topically-precise burnout memory can beat an emotionally-matched recovery memory even when the latter's framing is more appropriate. This is partly a genuine trade-off (retrieval should perhaps blend one topical anchor with emotionally-current context) and partly judge-rubric bias; the divergent standard/long subset results suggest sensitivity to scenario construction.

**Cosine is the wrong space twice over.** Both v1 discoveries generalize: internal-representation cosine failed for retrieval (the cone), and state-cosine arc separation compressed under the bounded GRU — while decoded-coordinate retrieval and probe decodability held or improved. Systems that retrieve or compare in a learned space should check whether an interpretable low-dimensional decode is the better metric space.

**Evaluation methodology as a contribution.** 33% order-flipped judgings and a reasoning-trace parser artifact were together large enough to manufacture an 83% headline. Position counterbalancing, `<think>` stripping, per-judge consistency reporting, a recency control, and sign tests should be table stakes for LLM-judged companion-memory evaluations; ours reversed our own conclusions.

---

## 10. Known Limitations

1. **RQ3 vs semantic-only is unresolved** (dead heat, n=40, tie-dominant). Needs ~100+ scenarios, human raters, and power analysis. The recency-baseline margin (p=0.064) narrowly misses 0.05.
2. **Synthetic deep-arousal text.** The floor-breaking sessions are template-generated; style diversity is limited. Honest upgrade: consented fatigue/depression support corpora or human-annotated DailyDialog.
3. **Synthetic trajectories overall.** Template arcs are simpler and more Markovian than real emotional life; all representation results are upper-bounded by this. Real multi-session histories are the decisive test (and the state-utility objective is designed to exploit them).
4. **State utility is trained, not emergent** — with a deliberately weakened cold start as the flip side (first-session predictions are worse than a session-only model's; the 3-session warm-up gate already covers the deployment impact).
5. **Adaptive α and recency γ are shipped but not in the reported eval** — they are ablation arms, not measured claims.
6. **Lexicon VAD targets are noisy** — a 145-word lexicon's session scores are a crude (v,a) proxy; the auxiliary head inherits that noise ceiling.
7. **Judge panel homogeneity** — two of three judges are Llama-family; a non-Llama/non-Qwen judge would strengthen cross-model claims.
8. **Warm-up period** — no summary or retrieval until session 3.

---

## 11. Repository Guide and Reproduction

### 11.1 Artifacts of record

| File | Contents |
|---|---|
| `checkpoints/transition_v2.pt` | **Production world model** — GRU, epoch 43, session_dim=387, ablation_gap 0.187 (train metric), `model_config.arch="gru"` |
| `checkpoints/valence_arousal_probe_v2.pt` | Linear probe, r=0.770/0.672, embedded `thresholds` |
| `checkpoints/transition_v1.pt`, `valence_arousal_probe.pt` | Legacy MLP lineage (loadable via arch detection; historical only) |
| `data/trajectories_10k_v2.json` / `_val.json` | Conversation-disjoint train/val (10,000 / 2,000) |
| `checkpoints/train_v2.log`, `probe_v2.log`, `leakage_v2.log`, `arcsep_v2.log`, `probegen_v2.log`, `longsim_v2.log` | Evidence trail for every number in §6 |
| `experiments/contexts[_long].json`, `results[_long].json` | Evaluation inputs/outputs, §8 |

### 11.2 Module map (details in `codebase_map.md`)

`kokoro/` — encoder, vad, transition (GRU + MLP + loader + `predict_from_state`), decoder, store, retrieval, memory. `data/` — prepare (ED + circumplex map), low_arousal_pool, arc_templates (16), construct_trajectories. `training/` — train (VICReg+util+vad, vectorized), train_probe, split (disjoint + leakage report), validate (Table-1 baselines). `diagnostics/` — common (arch-aware loading) + 9 scripts (§6). `experiments/` — scenarios ×2, build_contexts, run_llm_eval (counterbalanced + stats + recency arm), run_evaluation (legacy single-process). `tests/` — 89 tests, all passing. `figures/`, `demo.py`, `examples/` — visualization, Gradio circumplex demo, API samples.

### 11.3 Full reproduction

```bash
# 1. Data (leakage-free by construction, low-arousal augmented)
python -m data.construct_trajectories 10000 --augment-low-arousal \
    --holdout-conv-fraction 0.2 --out data/trajectories_10k_v2.json

# 2. World model (~25 min CPU)
python -m training.train --data data/trajectories_10k_v2.json \
    --val-data data/trajectories_10k_v2_val.json \
    --checkpoint checkpoints/transition_v2.pt --epochs 60 \
    --util-weight 50 --util-margin 0.2 --vad-weight 10

# 3. Probe + threshold calibration
python -m training.train_probe --data data/trajectories_10k_v2.json \
    --val-data data/trajectories_10k_v2_val.json \
    --checkpoint checkpoints/transition_v2.pt \
    --out-probe checkpoints/valence_arousal_probe_v2.pt

# 4. Diagnostics (defaults already point at v2 + disjoint val)
python -m diagnostics.state_ablation        # gap: +0.1575
python -m diagnostics.eval_worldmodel       # +0.182 vs persistence
python -m diagnostics.per_arc_val_loss      # arousal arcs better
python -m diagnostics.topic_leakage         # r = 0.04
python -m diagnostics.norm_drift            # 1.00x
python -m diagnostics.arc_separation
python -m diagnostics.probe_generalization
python -m diagnostics.longitudinal_sim

# 5. Evaluation (requires GROQ_API_KEY in .env)
python experiments/build_contexts.py            # + --long
python experiments/run_llm_eval.py --with-recency-baseline    # + --long
```

---

## 12. Citations

**MiniLM / Sentence-Transformers:** Reimers & Gurevych (2019), *Sentence-BERT*, EMNLP. Wang et al. (2020), *MiniLM*, NeurIPS.
**VICReg:** Bardes, Ponce & LeCun (2022), *VICReg: Variance-Invariance-Covariance Regularization*, ICLR.
**GRU:** Cho et al. (2014), *Learning Phrase Representations using RNN Encoder–Decoder*, EMNLP.
**EmpatheticDialogues:** Rashkin, Smith, Li & Boureau (2019), ACL.
**VAD norms:** Warriner, Kuperman & Brysbaert (2013), *Behavior Research Methods* 45(4). Mohammad (2018), *Word Affect Intensities*, LREC.
**Circumplex:** Russell (1980), *J. Personality & Social Psychology* 39(6). Posner, Russell & Peterson (2005), *Development & Psychopathology*.
**Emotional trajectories:** Bonanno (2004); Tedeschi & Calhoun (1996); Clark & Beck (2010); Stroebe & Schut (1999).
**LLM-judge position bias:** Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS.
**RAG:** Lewis et al. (2020), NeurIPS. **LayerNorm:** Ba, Kiros & Hinton (2016). **ChromaDB:** Chroma (2023). **Llama/Groq:** Meta AI (2023).

---

*End of report. All §6 metrics are from `transition_v2.pt` (GRU, epoch 43) + `valence_arousal_probe_v2.pt` on the conversation-disjoint validation set; all §8 results are from the 2026-07-03 counterbalanced evaluation run. Log files listed in §11.1 reproduce every table.*
