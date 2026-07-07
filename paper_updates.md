# Paper Updates — required changes, section by section

**Date:** 2026-07-03
**Basis:** GRU world-model rebuild (`checkpoints/transition_v2.pt` + `valence_arousal_probe_v2.pt`), conversation-disjoint data (`trajectories_10k_v2[_val].json`), full diagnostic re-run, and the debiased 40-scenario LLM evaluation. All numbers below are reproducible from the commands in README §Training and §Evaluation.

This document maps onto the section structure of `research_report.md` (the paper source). Items marked **[MUST]** are correctness issues; **[ADD]** are new content; **[REFRAME]** are positioning changes.

---

## §1 Motivation / Abstract

- **[REFRAME]** The claim hierarchy inverts. Lead with the representation result (a recurrent emotional state that provably carries trajectory information and decodes both circumplex axes), then the recency-baseline win, then present the semantic-only response comparison as unresolved at current evaluation power. Do NOT lead with an LLM-judge win rate — ours is a dead heat once debiased, and §8/§11 explain why the old numbers were artifacts.
- **[ADD]** One sentence: "We also report a negative methodological finding: 33% of individual LLM-judge decisions in our setting flip with response presentation order, and a reasoning-model judge was silently mis-parsed into a constant vote — position counterbalancing and reasoning-trace stripping reversed our own earlier conclusions."

## §2 System Architecture

- **[MUST]** Replace the MLP transition model description with the GRU:
  - state = GRUCell hidden (bounded (−1,1)^384, gated), input = LayerNorm(session embedding, 387-dim with VAD features)
  - prediction head: Linear(384→512)→GELU→Linear(512→384), decoupled from the state
  - auxiliary head: Linear(384→2) predicting the NEXT session's (valence, arousal)
  - ~1.29M parameters; public API unchanged (`forward(state, emb) → state`, zero cold start)
- **[MUST]** Update the retrieval score: semantic cosine rescaled to [0,1]; emotional axis = VAD-coordinate L2 proximity; optional recency term γ·exp(−rank_age/τ); optional dispersion-gated adaptive α (α→1 when stored (v,a) dispersion < 0.15). Cite the formula from README.
- **[ADD]** Note that decoder thresholds are calibrated percentiles stored *inside* the probe checkpoint (no hardcoded constants to go stale).

## §2.3 / §5 Training Objective

- **[MUST]** New loss: `L = 25·L_sim + 25·L_var + 1·L_cov + 50·L_util + 10·L_vad`.
  - `L_util = ReLU(0.2 − (cos(z_t, e_{t+1}) − cos(z⁰_t, e_{t+1})))` averaged over warm steps, where z⁰ is the prediction from a zeroed state on the same session. State the design intent: this optimizes the state-ablation gap directly; a model that ignores its memory cannot minimize it.
  - `L_vad = MSE(vad_head(h_t), (v,a)_{t+1})` — the trajectory-forecasting task where history actually pays.
- **[ADD — key ablation table]** Three rows, same data and architecture except where noted:
  | Model | util/vad terms | state-ablation gap | val cosine loss |
  |---|---|---|---|
  | MLP (legacy) | — | +0.0001 | 0.503* |
  | GRU | util=10, margin=0.1, no vad | +0.005 | 0.503 |
  | **GRU (final)** | util=50, margin=0.2, vad=10 | **+0.158** | 0.554 |
  (*leaky split.) The middle row is the paper's most instructive finding: **recurrence alone does not produce state use on near-Markovian synthetic arcs — the objective must demand it.** Present the 0.05 val-loss increase as the explicit, quantified price of trajectory dependence.
- **[MUST]** Checkpoint selection criterion: val_loss − ablation_gap (state why).

## §3 Training Data

- **[MUST]** 16 arc types (add `depressive_shutdown` Q3→deep-Q4 and `unwinding_to_serenity` Q1→deep-Q2), two new zones (`shutdown`(−0.6,−0.7), `serene`(+0.5,−0.6)).
- **[MUST]** Low-arousal augmentation: 600 template-generated sessions in Q4-deep (v∈[−0.85,−0.35], a∈[−0.80,−0.35]) and Q2-deep (v∈[+0.30,+0.75], a∈[−0.80,−0.40]); training arousal coverage now [−0.80, +0.80] (was floored at −0.30). Emphasize both polarities are needed so deep deactivation is not a valence proxy. Disclose the text is synthetic and templated.
- **[MUST — disclosure]** The v1 split was contaminated: with `max_uses_per_conv=8` and trajectory-level splitting, **100% of validation trajectories shared source conversations (identical text and embeddings) with training**. All v1 validation metrics were optimistically biased. The v2 pipeline reserves 20% of source conversations *before* trajectory construction (`--holdout-conv-fraction`), yielding 10,000 train / 2,000 val trajectories with zero shared conversations.

## §4–§6 Collapse history / Fixes / Training History

- Keep the dimensional-collapse narrative (it is sound) but add Run 4 (GRU + weak incentives → gap 0.005) and Run 5 (final). Report the new PR (37.8/384) with the caveat that PR on bounded GRU states is not comparable to PR on unconstrained MLP outputs, and that the collapse-era comparison (1.4 → 339.7) belongs to the MLP lineage.

## §7 Diagnostic Results — replace the tables with the v2 numbers

All on the conversation-disjoint val set, current checkpoint:

| Diagnostic | Result | Old (for contrast) |
|---|---|---|
| State ablation gap | **+0.1575** (0.557 normal / 0.715 ablated) | +0.0001 |
| Prediction vs persistence baseline | **+0.182 cosine, better on 90.1% of steps**; positive delta on all 16 arcs | — |
| Probe Pearson r (valence / arousal) | **0.770 / 0.672** | 0.698 / 0.542 (leaky) |
| Per-arc loss, arousal−valence gap | **−0.017** (arousal arcs better; deep arcs best two) | flagged |
| Topic leakage (mean Spearman) | 0.041 | — |
| Norm over 50 sessions | 1.00× (bounded) | — |
| Arc separation (vs shuffled) | 1.227 vs 1.003 | 3.111 (leaky, MLP) |
| Longitudinal sim | valence MAE 0.247 warm; 10/10 arc changes, lag 0.7 sessions | — |

- **[ADD]** State ablation is now BOTH a diagnostic and a training target; discuss Goodhart risk explicitly (the gap is trained — its value is that the *mechanism* is verified: zeroing memory demonstrably degrades prediction). The prediction-vs-persistence result (+0.182, untrained quantity) is the independent corroboration.

## §8 Evaluation Pipeline

- **[MUST]** Document the new protocol: 3 judges × 2 presentation orders each; order-flip ⇒ that judge votes TIE (recorded `position_consistent=false`); `<think>` blocks stripped before verdict parsing; judge max_tokens 2048; majority vote; exact binomial sign test (ties excluded) + 95% Wilson CI.
- **[MUST]** Add Condition C (recency baseline: last k session summaries, no state summary) and why it is the critical control for two-phase scenarios.
- **[MUST — erratum]** The qwen3-32b "voted B on every scenario" anomaly flagged in the previous write-up was a **verdict-parser artifact** (its `<think>` deliberation was parsed as the verdict). State this plainly; it also invalidates the per-judge analysis of the previous run.

## §10–§11 Evaluation Results — full rewrite

Replace all win-rate tables with:

| Comparison | Standard (20) | Long (20) | Pooled (40) |
|---|---|---|---|
| Kokoro vs semantic-only | B 8 / A 6 / T 6 (p=0.79) | B 4 / A 8 / T 8 (p=0.39) | **B 12 / A 14 / T 14 (p=0.85)** |
| Kokoro vs recency | B 7 / C 3 / T 10 | B 7 / C 2 / T 11 | **B 14 / C 5 / T 21 (p=0.064)** |
| Retrieval divergence | 12/20, mean overlap 80% | 15/20, mean overlap 65% | 27/40 |
| Position-inconsistent judgings | 18/60 | 22/60 | **40/120 (33%)** |
| Judge agreement | 12 unanimous / 8 majority | 7 / 12 / 1 none | — |

Required narrative points:
1. The 83.3% (6-scenario) and 45%/54.5% (20-scenario, single-order) results do not survive counterbalancing + parser fixes; retract them explicitly rather than quietly dropping them.
2. vs semantic-only: dead heat; standard-set divergence subset favors B (7–3–2), long-set divergence subset favors A (7–3–5) — discuss why emotionally-current memories can lose on specificity.
3. vs recency: 14–5–21 — the one response-level comparison with a consistent direction in both sets; frame as the deployable claim ("the world model earns its place over trivial recency"), noting p=0.064 at n=19 decisive.
4. Tie-dominance (35–52% of comparisons) is itself a finding about LLM-judge sensitivity at 2–3-sentence response length.

## §12 Discussion

- **[REFRAME]** "Where the value is": representation and forecasting (state summary generation, trend detection, next-session (v,a) prediction with 0.7-session arc-change lag) rather than response-preference wins.
- **[ADD]** Trained-vs-emergent state use; the near-Markovian structure of template arcs as the likely reason history is cheap to ignore without an explicit objective; expected behavior on real longitudinal data is the open question.

## §13 Limitations — update

Drop "MLP not LSTM" and the arousal-floor items (fixed); add: synthetic deep-arousal text; trained (not emergent) state utility with weaker cold-start; judge-panel insensitivity / tie rates; adaptive-α not enabled in the reported eval (present as ablation-ready mechanism).

## §14–§15 Inventory / Next Steps

- Inventory: transition_v2.pt (GRU, epoch 43, ablation_gap 0.187 train-metric), valence_arousal_probe_v2.pt (thresholds embedded), trajectories_10k_v2[_val].json, new modules (training/split.py, data/low_arousal_pool.py, diagnostics/common.py).
- Next steps (rewrite): (1) 100+ scenario eval with human raters and per-scenario power analysis; (2) real longitudinal corpora — the decisive test of trained state utility; (3) enable adaptive-α and recency-γ arms as reported ablations; (4) real low-arousal corpus replacing templates; (5) judge-panel diversity (non-Llama/Qwen judge) with order-consistency reporting as a standard metric.

## §16 Citations to add

- Cho et al. (2014) — GRU.
- Zheng et al. (2023), "Judging LLM-as-a-Judge" — position bias in LLM judges (supports §8).
- Wang et al. (2023) or similar on positional bias / counterbalancing in pairwise LLM evaluation.
