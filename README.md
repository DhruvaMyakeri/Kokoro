# Kokoro 心

**Emotional trajectory memory for AI companions.**

Kokoro is a memory layer that tracks how a user has been feeling across conversations, not just what they said. It gives your AI companion a sense of the person's emotional history so responses feel like they come from something that actually knows them.

Most companion systems remember facts. Kokoro remembers how someone has been doing.

---

## The problem

Every AI companion today resets emotionally at the start of each conversation. It might remember that you have a dog named Bruno or that you work in finance. It does not know that you have been grinding through burnout for three weeks, that last Tuesday felt like a turning point, or that your message today, _"still here lol"_, carries more weight than it looks like.

---

## How it works

**Each session gets encoded.** When a conversation ends, Kokoro encodes the full session into an embedding and runs it through a learned transition model that updates a continuous state vector, a compressed representation of the user's recent emotional trajectory.

**The state is decoded into emotional coordinates.** A linear probe maps the state vector to valence and arousal (Russell's circumplex model), then generates a plain-English summary of the trajectory.

**Retrieval is emotionally weighted.** When the user sends a new message, Kokoro retrieves relevant past sessions using a hybrid score:

```
score_i = (1-γ) · [α · semantic(query, session_i) + (1-α) · emotional(state, session_i)] + γ · recency_i
```

The semantic axis is cosine similarity rescaled to [0, 1]; the emotional axis is VAD-coordinate L2 proximity, comparing the user's current decoded (valence, arousal) against each stored session's coordinates; the optional recency axis (γ, default 0) is an exponential decay over storage order. Default blend: α = 0.6. With `adaptive_alpha=True`, α falls back toward 1.0 when the user's stored history is emotionally homogeneous (the emotional axis then has nothing to rank).

**The LLM receives both.** The state summary is injected into the system prompt; the retrieved memories provide session-level context.

---

## Installation

```bash
git clone https://github.com/DhruvaMyakeri/kokoro
cd kokoro
pip install -r requirements.txt
```

---

## Quickstart

```python
from kokoro import WorldMemory

memory = WorldMemory(user_id="alice")

# At the end of each conversation session
memory.update(session_turns)

# Before each LLM reply
context = memory.get_context(user_message)

# Use it
response = llm(
    system=context["state_summary"],
    memories=context["relevant_memories"],
    message=user_message,
)
```

`session_turns` is a list of `{"role": "user"|"assistant", "content": "..."}` dicts.

---

## What `get_context()` returns

```python
context = memory.get_context("still here lol")

{
    "state_summary":      "User has been trending negatively across recent sessions, "
                          "with affect noticeably declining and low energy levels.",
    "relevant_memories":  [
        "finished the sprint but honestly just feel empty. not relieved, just empty. "
        "i used to love coding. now i just stare at the screen.",
        "missed a deadline for the first time in like two years. just couldn't focus.",
        "snapped at a junior dev today for something stupid. felt awful after.",
    ],
    "valence":            -0.336,
    "arousal":            -0.118,
    "trend":              "declining",
    "session_count":      5,
    "ready":              True,
}
```

---

## Architecture

| Component            | What it does                                                                                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SessionEncoder`     | Encodes a conversation session using `all-MiniLM-L6-v2` + optional Warriner VAD lexicon features → 384 or 387-dim output                                                                                                                                                          |
| `TransitionModelGRU` | The world model. The user state is a GRU hidden state (gated, bounded accumulation of trajectory history), decoupled from a separate prediction head. Trained with VICReg + a **state-utility margin loss** (predictions from accumulated state must beat zero-state predictions) + an auxiliary next-session (valence, arousal) prediction head. Trained on 10,000 synthetic trajectories across 16 arc types with a conversation-disjoint validation split. |
| `StateDecoder`       | Linear probe (384→2) mapping state vector to (valence, arousal). Classification thresholds travel inside the probe checkpoint. Generates plain-English trend summary.                                                                                                             |
| `MemoryStore`        | ChromaDB. Hybrid retrieval: normalized semantic cosine + VAD-coordinate L2 emotional distance + optional recency decay, with optional dispersion-gated adaptive alpha.                                                                                                            |
| `WorldMemory`        | Public API. Orchestrates encode → transition → decode → store → retrieve.                                                                                                                                                                                                        |

---

## Key results

All representation metrics below are computed on a **conversation-disjoint validation set** (train and val trajectories share zero source conversations). Earlier published numbers were computed on a split where 100% of validation trajectories shared source conversations with training; those legacy figures are shown for reference but are optimistically biased.

| Metric                                                        | Current (GRU, honest split)              | Legacy (MLP, leaky split) |
| ------------------------------------------------------------- | ---------------------------------------- | -------------------------- |
| **State-ablation gap** (does the recurrent state matter?)     | **+0.158** (threshold: 0.05)             | +0.0001 (state unused)     |
| Next-session prediction vs naive "repeat last session"        | **+0.182 cosine, better on 90.1% of steps** | not measured               |
| Probe valence r                                               | **0.770**                                | 0.698                      |
| Probe arousal r                                               | **0.672**                                | 0.542                      |
| Arousal-primary vs valence-primary arc prediction loss        | **−0.017 (arousal arcs predicted better)** | flagged worse              |
| Ground-truth arousal coverage in training data                | **down to −0.80**                        | floor at −0.30             |
| Emotional/semantic retrieval axis correlation (topic leakage) | **0.04** (independent)                   | —                          |
| State norm over 50-session rollouts                           | **1.00× (bounded by construction)**      | unbounded MLP output       |
| Arc-change detection (simulated users)                        | **10/10 detected, 0.7-session lag**      | —                          |

### Response-level evaluation (honest protocol)

The LLM-judge evaluation now uses a **position-counterbalanced 3-judge panel** (every judge scores each pair in both presentation orders; an order-flipped vote counts as TIE), a **recency baseline** (Condition C: just show the last k sessions), and exact sign tests with Wilson intervals. Under this protocol:

| Comparison (40 scenarios pooled: 20 standard + 20 long-history) | Result                          |
| ---------------------------------------------------------------- | -------------------------------- |
| Kokoro (B) vs semantic-only (A)                                   | B 12 / A 14 / TIE 14 — no significant difference (p = 0.85) |
| Kokoro (B) vs recency baseline (C)                                | **B 14 / C 5 / TIE 21** (p = 0.064) |
| Standard set, scenarios where retrieval diverges (12/20)          | B 7 / A 3 / TIE 2                |
| Position-inconsistent individual judgings (neutralized)           | **40/120 (33%)**                 |

Three things worth being direct about:

1. **Previously reported win rates were substantially judge-protocol artifacts.** A third of individual judgings flip with presentation order, and the earlier pipeline's verdict parser mis-read the reasoning-model judge's `<think>` output (which is why `qwen3-32b` appeared to vote for Kokoro on every scenario). With both fixed, Kokoro vs semantic-only is a statistical dead heat at this sample size.
2. **Kokoro consistently beats the recency baseline** (14–5 with 21 ties, p = 0.064) — the emotional mechanism adds signal that neither topical matching nor "just show recent sessions" provides, which is the comparison a deployed system actually faces.
3. **The representation-level results are now the headline**, and they are strong: the recurrent state demonstrably carries trajectory information (ablation gap 3× threshold), decodes both circumplex axes well on honest data, and tracks simulated users with sub-session detection lag. The response-level preference test needs a larger scenario set (and human raters) to resolve differences the judge panel currently scores as ties — 21/40 recency comparisons and 14/40 semantic comparisons are ties.

---

## Evaluation

Three retrieval conditions are compared:

- **Condition A (baseline):** α=1.0, semantic-only retrieval, no state summary
- **Condition B (Kokoro):** α=0.6, hybrid retrieval + emotional state summary in system prompt
- **Condition C (recency baseline):** the last k session summaries, no state summary — the cheapest competitor to emotional retrieval

**Standard set:** 20 scenarios, 4–5 sessions each. **Long-history set:** 20 scenarios, 10–11 sessions each, with explicit emotional phase shifts (the new message reuses Phase 1 topic keywords, creating retrieval tension between an emotionally outdated but topically matched memory and an emotionally current one).

**Judging protocol:** three judge models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b`), each scoring every pair in **both presentation orders**. A judge that names a different winner when the order flips is recorded as position-inconsistent and votes TIE. Reasoning-model `<think>` output is stripped before verdict parsing (an earlier parser bug made `qwen3-32b` appear to vote for Kokoro on every scenario). Majority vote across the three debiased judges; exact binomial sign test and 95% Wilson interval on win rates.

Reproduce: `python experiments/build_contexts.py [--long]` then `python experiments/run_llm_eval.py [--long] --with-recency-baseline` (requires `GROQ_API_KEY`).

Full technical and empirical report: [`research_report.md`](research_report.md); change log: [`improvements_report.md`](improvements_report.md); paper deltas: [`paper_updates.md`](paper_updates.md).

---

## Training

The transition model is trained on 10,000 synthetic emotional arc trajectories with:

```
L = 25·L_sim + 25·L_var + 1·L_cov + 50·L_util + 10·L_vad
```

- `L_sim/L_var/L_cov` — VICReg (cosine invariance + variance + covariance) on the prediction head output
- `L_util` — **state-utility margin loss**: predictions from the accumulated state must beat predictions from a zeroed state by ≥0.2 cosine at every warm step. This directly optimizes the state-ablation gap; a model that ignores its recurrent memory cannot minimize it. (Without this term, the GRU converged to a gap of +0.005 — architecture alone does not buy state use.)
- `L_vad` — auxiliary next-session (valence, arousal) prediction. Next-session *text* is dominated by unpredictable topical content; next-session *emotional position* follows the arc — this is the task that makes history worth carrying.

Sessions are sourced from EmpatheticDialogues (Rashkin et al., 2019), augmented with a synthetic deep-low-arousal pool (`data/low_arousal_pool.py`) that extends arousal coverage from the ED floor of −0.30 down to −0.80 in both valence polarities, across 16 arc types. Train and validation trajectories share **zero source conversations** (the legacy trajectory-level split left 100% of validation trajectories sharing conversations with training).

To reproduce:

```bash
python -m data.construct_trajectories 10000 --augment-low-arousal --holdout-conv-fraction 0.2 --out data/trajectories_10k_v2.json
python -m training.train --data data/trajectories_10k_v2.json --val-data data/trajectories_10k_v2_val.json --checkpoint checkpoints/transition_v2.pt --epochs 60 --util-weight 50 --util-margin 0.2 --vad-weight 10
python -m training.train_probe --data data/trajectories_10k_v2.json --val-data data/trajectories_10k_v2_val.json --checkpoint checkpoints/transition_v2.pt --out-probe checkpoints/valence_arousal_probe_v2.pt
```

---

## Diagnostics

```bash
python -m diagnostics.arc_separation      # arc clustering metric (current: 3.111)
python -m diagnostics.probe_generalization # probe on naturalistic conversations
python -m diagnostics.per_arc_val_loss    # per-arc validation loss
python -m diagnostics.state_ablation      # state contribution check
python -m diagnostics.norm_drift          # state vector norm stability
python -m diagnostics.topic_leakage       # semantic/emotional axis separation
```

---

## Limitations

- **Response-level preference is unresolved vs semantic-only.** Under the debiased judging protocol, Kokoro vs semantic-only retrieval is a statistical dead heat at 40 scenarios (12–14–14). The advantage over the recency baseline (14–5–21, p = 0.064) is the defensible response-level claim. Resolving the semantic-only comparison needs a larger scenario set and human raters.
- **Deep-low-arousal text is synthetic.** The arousal-coverage fix uses template-generated sessions; the honest upgrade is a real low-energy corpus (fatigue/depression support text with consent, or human-annotated DailyDialog).
- **Synthetic trajectories.** Trained on arc templates, not real multi-session user histories. The state-utility and VAD-prediction results show the model *can* carry trajectory information; whether real users follow learnable arcs is untested.
- **State utility is trained, not emergent.** The ablation gap exists because the loss demands it (the same GRU without the utility term converged to +0.005). This is by design — it also means the cold-start (first-session) prediction is deliberately weaker than a session-only model's.
- **Warm-up period.** State summary is `None` for the first two sessions.
- **Emotional retrieval can backfire.** The hybrid axis sometimes surfaces an emotionally current but topically weaker session; judges sometimes prefer the semantically precise memory. Dispersion-gated adaptive alpha (`adaptive_alpha=True`) mitigates the homogeneous-history case but is not enabled in the reported eval.
- **Evaluation scale.** 40 scenarios with high tie rates is small; results are directionally informative, not definitive.

---

## Dependencies

- Python 3.10+
- `torch >= 2.2.0`
- `sentence-transformers >= 2.7.0`
- `numpy >= 1.26.0`
- `chromadb >= 0.5.0`
- `datasets >= 2.19.0`
- `tqdm >= 4.66.0`

Optional for evaluation:

- `groq >= 0.4.0`

---

## Citation

```bibtex
@misc{kokoro2026,
  title   = {Kokoro: Emotional Trajectory Memory for AI Companions},
  author  = {Myakeri, Dhruva},
  year    = {2026},
  note    = {Paper in preparation},
  url     = {https://github.com/DhruvaMyakeri/kokoro},
}
```

---

## License

MIT
