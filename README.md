# Kokoro 心

**Emotional trajectory memory for AI companions.**

Kokoro is a memory layer that tracks how a user has been feeling across conversations — not just what they said. It gives your AI companion a sense of the person's emotional history so responses feel like they come from something that actually knows them.

Most companion systems remember facts. Kokoro remembers how someone has been doing.

---

## The problem

Every AI companion today resets emotionally at the start of each conversation. It might remember that you have a dog named Bruno or that you work in finance. It does not know that you have been grinding through burnout for three weeks, that last Tuesday felt like a turning point, or that your message today — *"still here lol"* — carries more weight than it looks like.

---

## How it works

**Each session gets encoded.** When a conversation ends, Kokoro encodes the full session into an embedding and runs it through a learned transition model that updates a continuous state vector — a compressed representation of the user's recent emotional trajectory.

**The state is decoded into emotional coordinates.** A linear probe maps the state vector to valence and arousal (Russell's circumplex model), then generates a plain-English summary of the trajectory.

**Retrieval is emotionally weighted.** When the user sends a new message, Kokoro retrieves relevant past sessions using a hybrid score:

```
score_i = α · semantic(query, session_i) + (1 - α) · emotional(state, session_i)
```

The emotional axis uses VAD-coordinate L2 distance — comparing the user's current decoded (valence, arousal) against each stored session's coordinates. Default blend: α = 0.6 (60% semantic, 40% emotional).

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

| Component | What it does |
|-----------|-------------|
| `SessionEncoder` | Encodes a conversation session using `all-MiniLM-L6-v2` + optional Warriner VAD lexicon features → 384 or 387-dim output |
| `TransitionModel` | MLP with split LayerNorm. Takes (state, session_emb) → next state. Trained on 10,000 synthetic trajectories across 14 emotional arc types. Trained with VICReg loss (variance + invariance + covariance) to prevent dimensional collapse. |
| `StateDecoder` | Linear probe (384→2) mapping state vector to (valence, arousal). Generates plain-English trend summary. |
| `MemoryStore` | ChromaDB. Stores session embeddings + decoded (valence, arousal) per session. Hybrid retrieval: semantic cosine + VAD-coordinate L2 emotional distance. |
| `WorldMemory` | Public API. Orchestrates encode → transition → decode → store → retrieve. |

---

## Key results

| Metric | Value |
|--------|-------|
| Participation Ratio (state space spread) | **339.7 / 384** (baseline: 1.4 / 384) |
| Probe valence r | **0.698** (baseline: ~0) |
| Probe arousal r | **0.542** (baseline: ~0) |
| Arc separation ratio | **3.111** vs 1.047 baseline (+197%) |
| Retrieval eval — standard (20 scenarios, 4–5 sessions) | **50% win rate** vs semantic-only |
| Retrieval eval — long-history (6 scenarios, 10–12 sessions) | **83.3% win rate** vs semantic-only |

The long-history win rate (83.3%) reflects the core mechanism: when enough sessions exist to span an emotional phase shift, hybrid retrieval pulls phase-matched memories while semantic-only retrieval pulls topically-matched but emotionally-stale ones.

---

## Evaluation

Two retrieval conditions were compared:

- **Condition A (baseline):** α=1.0 — semantic-only retrieval, no state summary
- **Condition B (Kokoro):** α=0.6 — hybrid retrieval + emotional state summary in system prompt

An LLM judge (`llama-3.3-70b-versatile`, blind to condition labels) rated which response showed more emotional awareness. Standard set: 20 scenarios, 4–5 sessions each. Long-history set: 6 scenarios, 10–12 sessions with explicit emotional phase shifts between early and late sessions.

Full results: [`experiments/eval_report.md`](experiments/eval_report.md) and [`experiments/eval_report_long.md`](experiments/eval_report_long.md).  
Full technical and empirical report: [`research_report.md`](research_report.md).

---

## Training

The transition model was trained on 10,000 synthetic emotional arc trajectories using VICReg loss:

```
L = 25·L_sim + 25·L_var + 1·L_cov
```

Sessions are sourced from EmpatheticDialogues (Rashkin et al., 2019) and selected to follow 14 arc types (gradual decline, slow recovery, grief arc, burnout, post-traumatic growth, etc.).

Three training runs resolved a severe dimensional collapse in the original model (PR: 1.4 → 22.1 → 339.7):

1. **Run 1** — VICReg with MSE similarity term: val loss stuck at 0.9609 (MSE recreated the unit-sphere constraint that was blocking variance)
2. **Run 2** — VICReg with cosine similarity term: val loss 0.5059, PR 22.1
3. **Run 3** — Split LayerNorm + arousal-primary arc templates: val loss 0.5087, PR 339.7

To retrain:

```bash
python -m training.train
python -m training.train_probe
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

- **Arousal coverage.** Ground-truth arousal in EmpatheticDialogues only spans [−0.30, +0.80]. The model cannot reliably decode very low arousal states (depressed, exhausted) until the session pool is extended with genuinely low-energy content.
- **Warm-up period.** State summary is `None` for the first two sessions.
- **Synthetic training data.** The model is trained on arc templates, not real multi-session user histories. Val loss plateaued at ~0.508 — a data ceiling, not an architecture ceiling.
- **MLP, not LSTM.** The transition model processes sessions one at a time; long-range sequential dependencies are only captured via the accumulated state vector.

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
