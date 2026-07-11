# Kokoro 心

**Emotional trajectory memory for AI companions.**

Kokoro is a memory layer that tracks how a user has been feeling across conversations, not just what they said. It maintains a learned recurrent "world model" of the user's emotional trajectory, decodes it into interpretable valence/arousal coordinates, and uses those coordinates to retrieve emotionally relevant memories and to brief the companion LLM on the user's direction of travel.

Most companion systems remember facts. Kokoro remembers how someone has been doing.

Full technical report: [research_report.md](research_report.md). Change history: [improvements_report.md](improvements_report.md). Codebase navigation: [codebase_map.md](codebase_map.md).

---

## The problem

Every AI companion today resets emotionally at the start of each conversation. It might remember that you have a dog named Bruno or that you work in finance. It does not know that you have been grinding through burnout for three weeks, that last Tuesday felt like a turning point, or that your message today, _"still here lol"_, carries more weight than it looks like.

There is a sharper version of this problem: purely semantic retrieval actively fetches the *wrong* memories after an emotional phase shift. A user who burned out over "the big sprint" and then recovered will, on mentioning "sprint" again, get their burnout-era memories back, because those match the keywords. Kokoro's hybrid retrieval is built to fetch the recovery-era ones instead.

---

## How it works

**Each session gets encoded.** When a conversation ends, Kokoro encodes the full session with `all-MiniLM-L6-v2` (user turns weighted 2x) plus a 3-dim valence/arousal/dominance signal from a Warriner VAD lexicon, giving a 387-dim vector.

**A GRU world model updates the user state.** The user's state is a 384-dim GRU hidden state: a gated, bounded accumulator of trajectory history, updated once per session. A separate prediction head forecasts the next session's embedding, and an auxiliary head forecasts the next session's (valence, arousal). The state and the prediction are deliberately different objects; the previous MLP design conflated them and its state provably carried no information (see Training below).

**The state is decoded into emotional coordinates.** A linear probe maps the state to (valence, arousal) on Russell's circumplex, a trend is fit over the recent history, and a plain-English summary is generated ("declining across the past 7 sessions, currently negative and low-energy"). Classification thresholds are calibrated percentiles stored inside the probe checkpoint, so a retrain can never leave the decoder stale.

**Retrieval is emotionally weighted.** When the user sends a new message, past sessions are ranked by:

```
score_i = (1-y) * [ a * semantic_i + (1-a) * emotional_i ] + y * recency_i
```

- `semantic_i`: cosine similarity of MiniLM embeddings, rescaled to [0, 1]
- `emotional_i`: 1 - L2distance((v,a)_current, (v,a)_i) / sqrt(8), proximity in the decoded circumplex. This is deliberately not state-to-state cosine: all learned states sit in a tight angular cone (cos ~0.98-1.00 regardless of mood), so the interpretable 2-D space is the right metric space.
- `recency_i`: exponential decay over storage order, weight `y` (default 0, off)
- with `adaptive_alpha=True`, `a` falls back toward 1.0 when the stored history is emotionally homogeneous and the emotional axis has nothing to rank

**The LLM receives both.** The state summary goes into the system prompt; the retrieved memories provide session-level context.

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

response = llm(
    system=context["state_summary"],
    memories=context["relevant_memories"],
    message=user_message,
)
```

`session_turns` is a list of `{"role": "user"|"assistant", "content": "..."}` dicts. Useful constructor options: `alpha`, `top_k`, `recency_weight`, `adaptive_alpha`, `retrieval_fn` (plug in your own vector store), `checkpoint_path`/`probe_path`.

### What `get_context()` returns

```python
{
    "state_summary":      "User has been trending negatively across recent sessions, "
                          "with affect noticeably declining and low energy levels.",
    "relevant_memories":  ["finished the sprint but honestly just feel empty...",
                           "missed a deadline for the first time in two years...",
                           "snapped at a junior dev today. felt awful after."],
    "valence":            -0.34,
    "arousal":            -0.12,
    "trend":              "declining",
    "session_count":      5,
    "ready":              True,
}
```

There is also `memory.predict_next()`, which decodes the current state as a forecast of the next session's emotional position.

---

## Architecture

| Component            | What it does                                                                                                                                                                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SessionEncoder`     | Session text to 384-dim MiniLM embedding + 3 VAD lexicon dims (387 total). ParlAI artifact cleanup built in.                                                                                                                                     |
| `TransitionModelGRU` | The world model (~1.29M params). GRUCell hidden state = user state (bounded, drift-free); MLP head predicts next-session embedding; auxiliary head predicts next-session (v, a). Arch-detecting loader keeps legacy MLP checkpoints loadable.     |
| `StateDecoder`       | Linear probe (384 to 2) for (valence, arousal); trend slope over arc history; natural-language summary. Thresholds embedded in the probe checkpoint.                                                                                             |
| `MemoryStore`        | ChromaDB store with the hybrid semantic + emotional + recency scoring above, per-axis score auditing, and optional adaptive alpha. Works as a plain RAG index too (state vector optional).                                                       |
| `StateStore`         | SQLite persistence: state vector, session count, last (v, a), capped arc history.                                                                                                                                                                 |
| `WorldMemory`        | Public API. Orchestrates encode, transition, decode, persist, retrieve.                                                                                                                                                                          |

---

## Key results

All representation metrics are computed on a **conversation-disjoint validation set**: train and validation trajectories share zero source conversations. Earlier published numbers used a split where 100% of validation trajectories shared source conversations with training; they are shown only for contrast and are optimistically biased.

| Metric                                                     | Current (GRU v2, honest split)              | Legacy (MLP, leaky split) |
| ---------------------------------------------------------- | -------------------------------------------- | -------------------------- |
| **State-ablation gap** (does the recurrent state matter?)  | **+0.158** (threshold 0.05)                  | +0.0001 (state unused)     |
| Next-session prediction vs "repeat last session" baseline  | **+0.182 cosine, better on 90.1% of steps**  | not measured               |
| Probe valence Pearson r                                    | **0.770**                                    | 0.698                      |
| Probe arousal Pearson r                                    | **0.672**                                    | 0.542                      |
| Arousal-primary vs valence-primary arc prediction loss     | **-0.017 (arousal arcs predicted better)**   | flagged worse              |
| Training arousal coverage                                  | **down to -0.80**                            | floor at -0.30             |
| Emotional/semantic retrieval axis correlation              | **0.04** (independent axes)                  | n/a                        |
| State norm over 50-session rollouts                        | **1.00x** (bounded by construction)          | unconstrained              |
| Arc-change detection (simulated users)                     | **10/10 detected, 0.7-session lag**          | n/a                        |

### Response-level evaluation (debiased protocol)

The LLM-judge evaluation uses a position-counterbalanced 3-judge panel (every judge scores each pair in both presentation orders; an order-flipped vote counts as TIE), a recency baseline, and exact sign tests with Wilson intervals.

| Comparison (40 scenarios: 20 standard + 20 long-history)  | Result                                                       |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| Kokoro (B) vs semantic-only (A)                             | B 12 / A 14 / TIE 14, no significant difference (p = 0.85)   |
| Kokoro (B) vs recency baseline (C)                          | **B 14 / C 5 / TIE 21** (p = 0.064)                          |
| Standard set, scenarios where retrieval diverges (12/20)    | B 7 / A 3 / TIE 2                                            |
| Position-inconsistent individual judgings (neutralized)     | **40/120 (33%)**                                             |

Three things stated plainly:

1. **Previously reported win rates (83.3%, then 45-54.5%) were substantially judge-protocol artifacts and are retracted.** A third of individual judgings flip with presentation order, and the earlier pipeline's verdict parser misread the reasoning-model judge's `<think>` output, which is why `qwen3-32b` appeared to vote for Kokoro on every scenario. With both fixed, Kokoro vs semantic-only is a statistical dead heat at this sample size.
2. **Kokoro consistently beats the recency baseline** (14-5 with 21 ties), the comparison a deployed system actually faces: the emotional mechanism adds signal that neither topical matching nor "just show recent sessions" provides.
3. **The representation-level results are the headline**: the recurrent state provably carries trajectory information, decodes both circumplex axes well on honest data, and tracks simulated users with sub-session detection lag. Resolving the response-level comparison needs a larger scenario set and human raters; ties are the modal judge outcome.

---

## Training

### Data

Source pool: EmpatheticDialogues (~19k emotion-labeled conversations), each label mapped to Russell-circumplex coordinates. Synthetic multi-session trajectories are constructed by sampling 16 arc templates (gradual decline, slow recovery, grief, burnout-style anxiety-to-depression, five arousal-primary arcs, etc.) and picking real conversations whose coordinates fall in each arc waypoint zone.

Because EmpatheticDialogues has an arousal floor of about -0.30, the pool is augmented with 600 template-generated deep-low-arousal sessions in both valence polarities (exhaustion/shutdown and deep calm/serenity), extending coverage to -0.80. Dataset of record: 10,000 train / 2,000 val trajectories, 7.6 sessions each, **zero shared source conversations across splits** (validation conversations are held out before construction).

### Objective

```
L = 25*L_sim + 25*L_var + 1*L_cov + 50*L_util + 10*L_vad
```

- `L_sim`, `L_var`, `L_cov`: VICReg (cosine invariance + variance + covariance) on the prediction head output, preventing the dimensional collapse that killed the first-generation model
- `L_util`: **state-utility margin loss.** At every warm step, predictions from the accumulated state must beat predictions from a zeroed state by at least 0.2 cosine. This directly optimizes the state-ablation gap; a model that ignores its memory cannot minimize it.
- `L_vad`: auxiliary next-session (valence, arousal) prediction, the task where trajectory history genuinely pays

Checkpoints are selected by `val_loss - ablation_gap`, so an epoch cannot win the save by ignoring its state.

### The decisive ablation

| Run                      | util / margin | vad | State-ablation gap | Val loss |
| ------------------------ | ------------- | --- | ------------------- | -------- |
| MLP (legacy)             | none          | 0   | +0.0001             | 0.503*   |
| GRU, weak incentives     | 10 / 0.1      | 0   | +0.005              | 0.503    |
| **GRU, final**           | 50 / 0.2      | 10  | **+0.158**          | 0.554    |

(*leaky split.) On near-Markovian synthetic arcs, **recurrence alone does not produce state use; the objective must demand it.** The ~0.05 higher validation loss is the explicit, quantified price of genuine trajectory dependence.

### Reproduce

```bash
# 1. Data (leakage-free by construction, low-arousal augmented)
python -m data.construct_trajectories 10000 --augment-low-arousal \
    --holdout-conv-fraction 0.2 --out data/trajectories_10k_v2.json

# 2. World model (~25 min on CPU; rollouts are vectorized across the batch)
python -m training.train --data data/trajectories_10k_v2.json \
    --val-data data/trajectories_10k_v2_val.json \
    --checkpoint checkpoints/transition_v2.pt --epochs 60 \
    --util-weight 50 --util-margin 0.2 --vad-weight 10

# 3. Probe + threshold calibration (thresholds are written into the checkpoint)
python -m training.train_probe --data data/trajectories_10k_v2.json \
    --val-data data/trajectories_10k_v2_val.json \
    --checkpoint checkpoints/transition_v2.pt \
    --out-probe checkpoints/valence_arousal_probe_v2.pt
```

---

## Evaluation

Three conditions per scenario:

- **A (semantic baseline):** alpha=1.0, semantic-only retrieval, no state summary
- **B (Kokoro):** alpha=0.6 hybrid retrieval + state summary in the system prompt
- **C (recency baseline):** the last k session summaries, no state summary

Two scenario sets: 20 standard (4-5 sessions) and 20 long-history (10-11 sessions with an explicit emotional phase shift; the new message reuses old-phase topic keywords to create maximal tension between topically matched and emotionally current memories).

Judging: three models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b`), each scoring every pair in both presentation orders; order-inconsistent votes become TIE; `<think>` blocks stripped before parsing; majority vote; exact binomial sign test and Wilson interval.

```bash
python experiments/build_contexts.py            # add --long for the long set
python experiments/run_llm_eval.py --with-recency-baseline    # add --long
```

Requires `GROQ_API_KEY` in `.env`.

---

## Diagnostics

Every number in the results tables is reproducible from a diagnostic script (defaults already point at the current checkpoint and the disjoint validation file):

```bash
python -m diagnostics.state_ablation        # state contribution: gap +0.158
python -m diagnostics.eval_worldmodel       # +0.182 vs persistence baseline
python -m diagnostics.per_arc_val_loss      # arousal arcs now predicted best
python -m diagnostics.topic_leakage         # axis independence: r = 0.04
python -m diagnostics.norm_drift            # state norm 1.00x over 50 sessions
python -m diagnostics.arc_separation        # arc clustering vs shuffled control
python -m diagnostics.probe_generalization  # probe on naturalistic conversations
python -m diagnostics.longitudinal_sim      # tracking MAE, arc-change lag
```

---

## Repository layout

```
kokoro/        core package: encoder, vad, transition (GRU world model), decoder,
               store, retrieval, memory (public API)
data/          EmpatheticDialogues loader, circumplex map, 16 arc templates,
               low-arousal pool, trajectory constructor
training/      train (VICReg + state-utility + VAD aux, vectorized), train_probe,
               split (conversation-disjoint), validate
diagnostics/   9 diagnostic scripts + shared arch-aware loading
experiments/   scenario sets, two-stage evaluation pipeline
tests/         89 unit tests
checkpoints/   transition_v2.pt (production), valence_arousal_probe_v2.pt,
               legacy v1 artifacts, training histories
figures/       figure generators; demo.py is a Gradio circumplex demo
```

---

## Limitations

- **The response-level comparison vs semantic-only retrieval is unresolved** (dead heat at n=40, tie-dominant). The recency-baseline margin (p=0.064) is the defensible response-level claim.
- **Deep-low-arousal text is synthetic.** The honest upgrade is a real low-energy corpus.
- **Synthetic trajectories.** Template arcs are simpler and more Markovian than real emotional life; real longitudinal data is the decisive test, and the state-utility objective is designed to exploit it.
- **State utility is trained, not emergent**, with a deliberately weaker cold start as the flip side (covered in deployment by the 3-session warm-up gate).
- **Adaptive alpha and the recency weight ship but are not in the reported eval**; they are ablation arms.
- **Warm-up period.** State summary is `None` for the first two sessions.

---

## Dependencies

Python 3.10+, `torch >= 2.2`, `sentence-transformers >= 2.7`, `numpy >= 1.26`, `chromadb >= 0.5`, `datasets >= 2.19`, `tqdm`. Optional for evaluation: `groq`.

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

## License

MIT
