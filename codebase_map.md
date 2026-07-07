# Kokoro — Codebase Map (traceback memory)

Drop into any file, find it here, and trace **upstream** (where its input comes from),
**downstream** (where its output goes), and **sideways** (what else touches it).

---

## 0. Full pipeline diagram

```
                                  DATA GENERATION
┌─────────────────────────────────────────────────────────────────────────────┐
│  EmpatheticDialogues tarball (FB servers, cached ~/.cache/kokoro/)          │
│        │                                                                    │
│        ▼                                                                    │
│  data/prepare.py ──── EMOTION_TO_CIRCUMPLEX label→(v,a) map                 │
│        │  load_empathetic_dialogues() → list[LabeledSession]                │
│        │        ▲                                                           │
│        │        └── data/low_arousal_pool.py (--augment-low-arousal)        │
│        │            synthetic deep-Q4/Q2 sessions, arousal to −0.8          │
│        ▼                                                                    │
│  data/construct_trajectories.py  ◄── data/arc_templates.py                  │
│        │   (16 ArcTemplates over CircumplexZones; sampling weights)         │
│        │   [--holdout-conv-fraction → also emits *_val.json, conv-disjoint] │
│        ▼                                                                    │
│  data/trajectories_10k_v2.json (+ _val.json)  {arc_name, sessions:[...]}    │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ▼                 TRAINING
┌─────────────────────────────────────────────────────────────────────────────┐
│  training/split.py  conv_disjoint_split()  (leakage-free 80/20)             │
│        │                                                                    │
│  training/train.py                                                          │
│        │  precompute_embeddings() ── kokoro/encoder.py (MiniLM + VAD)       │
│        │  vicreg_loss(+util margin, +VAD aux) over GRU rollouts             │
│        ▼                                                                    │
│  checkpoints/transition_v2.pt  (arch=gru, session_dim=387)                  │
│        │                                                                    │
│  training/train_probe.py  (rolls states, fits Linear(384→2))                │
│        ▼                                                                    │
│  checkpoints/valence_arousal_probe_v2.pt (+ 'thresholds' calibration key)   │
│        │            └── recalibrate_decoder() patches decoder.py fallbacks  │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ▼                 INFERENCE (production API)
┌─────────────────────────────────────────────────────────────────────────────┐
│                        kokoro/memory.py :: WorldMemory                       │
│                                                                             │
│  update(session_turns):                                                     │
│    turns ─► encoder.py SessionEncoder.encode ──► emb (384 or 387)           │
│                       └─ vad.py VADLexicon (+3 dims if VAD ckpt)            │
│    (state, emb) ─► transition.py TransitionModel ──► new_state (384)        │
│    new_state ─► decoder.py StateDecoder(probe) ──► (valence, arousal)       │
│    new_state,(v,a) ─► store.py StateStore (SQLite: state, arc_history)      │
│    summary,emb[:384],(v,a) ─► retrieval.py MemoryStore (ChromaDB)           │
│                                                                             │
│  get_context(message, alpha):                                               │
│    StateStore.load ─► state ─► StateDecoder.decode(+arc_history)            │
│                                   │ state_summary, (v,a), trend             │
│    message ─► encoder.encode_text ─► query_emb (None if no message)         │
│    MemoryStore.retrieve(query_emb, (v,a), alpha, recency_weight, adaptive)  │
│         score = (1-γ)·[α'·sem₀₁ + (1-α')·emoVA] + γ·recency                 │
│         α' = adaptive_alpha(α, stored (v,a)) if adaptive                    │
│    ──► {state_summary, relevant_memories, valence, arousal, trend, ready}   │
│                    │                                                        │
│                    ▼                                                        │
│            companion LLM system prompt                                      │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ▼                 EVALUATION (two-stage, two-process)
┌─────────────────────────────────────────────────────────────────────────────┐
│  experiments/scenarios.py (20 std) / scenarios_long.py (20 two-phase)       │
│        │                                                                    │
│  Stage 1: experiments/build_contexts.py   [torch process, no network]       │
│        │  WorldMemory + _InMemoryStore(decoder)  (ChromaDB patched out)     │
│        │  ctx A (α=1.0) / ctx B (α=0.6 + summary) / ctx C (last-k recency)  │
│        ▼                                                                    │
│  experiments/contexts[_long].json                                           │
│        │                                                                    │
│  Stage 2: experiments/run_llm_eval.py     [Groq process, no torch]          │
│        │  responses A/B(/C) → 3 judges × 2 presentation orders              │
│        │  majority vote, sign test, Wilson CI                               │
│        ▼                                                                    │
│  experiments/results[_long].json → eval_report*.md                          │
└─────────────────────────────────────────────────────────────────────────────┘

  DIAGNOSTICS (offline, read checkpoints + trajectories):
  diagnostics/common.py ──loader/encoder/cache──► all diagnostics scripts
  state_ablation │ topic_leakage │ per_arc_val_loss │ norm_drift │
  arc_separation │ probe_generalization │ eval_worldmodel │
  longitudinal_sim │ arc_ood_eval          (also training/validate.py Table 1)
```

---

## 1. Core package — `kokoro/`

### `kokoro/__init__.py`
Re-exports the public API: `WorldMemory`, `StateStore`, `StateDecoder`, `MemoryStore`, `SessionEncoder`, `TransitionModel`.

### `kokoro/encoder.py`
| | |
|---|---|
| **Does** | Session (list of `{role, content}` turns) → 384-dim MiniLM embedding (user turns weighted 2×), optionally +3 VAD dims (387). Also `encode_text()` for single query strings and `decode_parlai_artifacts()` for `_comma_`-style tokens. |
| **Key symbols** | `SessionEncoder(use_vad_features=...)`, `.encode(turns)`, `.encode_text(str)`, `.output_dim`, `decode_parlai_artifacts()` |
| **Upstream** | Raw turn dicts from `WorldMemory.update`, training precompute, diagnostics; `kokoro/vad.py` for the +3 dims. |
| **Downstream** | `TransitionModel` input; `MemoryStore` semantic axis (first 384 dims only); every embedding cache in training/ and diagnostics/. |
| **Sideways** | `decode_parlai_artifacts` is re-implemented inline in `vad.py` (circular-import avoidance) and imported by training/diagnostics caches. |

### `kokoro/vad.py`
| | |
|---|---|
| **Does** | ~200-word Warriner/NRC-VAD lexicon subset; `VADLexicon.score_turns(turns, user_only=True)` → mean (v, a, d) float32(3). Explicit arousal channel (13× stronger separation than MiniLM). |
| **Upstream** | Turn text (user turns). |
| **Downstream** | Appended to embeddings by `SessionEncoder` when `use_vad_features=True` (dims 384:387 of the transition-model input; **stripped before storage/semantic use**). |

### `kokoro/transition.py`
| | |
|---|---|
| **Does** | The world model. **`TransitionModelGRU` (production)**: the user state IS a GRUCell hidden state (bounded, gated accumulation of history); a separate MLP head `predict(state)` maps state → unconstrained next-session prediction z. **`TransitionModel` (legacy MLP)**: split-LayerNorm MLP whose state was its own prediction — measured ablation gap ≈ 0, kept only to load old checkpoints. Also: `predict_from_state(model, state)` (arch-agnostic prediction — GRU head, MLP identity) and `load_transition_checkpoint(path)` (auto-detects arch from `model_config["arch"]` or state-dict keys). |
| **Upstream** | `(current_state from StateStore, session_emb from SessionEncoder)`. |
| **Downstream** | New state → `StateDecoder` probe, `StateStore.save`, `MemoryStore.add_session(state_vector=…)`; rollouts in `training/train.py`, `train_probe.py`, all diagnostics. `load_transition_checkpoint` is THE loading path — used by WorldMemory, diagnostics/common, train_probe, demo, validate, and the per-script diagnostics. |
| **Watch out** | `state_dim` (output, always 384) ≠ `session_dim` (input, 387 for VAD checkpoints). Anything comparing a state against a next-session embedding must go through `predict_from_state` AND slice `emb[:state_dim]`. GRU states are bounded in (−1, 1)^384 — norm drift is structurally impossible. |

### `kokoro/decoder.py`
| | |
|---|---|
| **Does** | `StateDecoder`: linear probe (384→2) → (valence, arousal); trend = least-squares slope over `arc_history`; `_build_summary()` composes the natural-language state summary; `decode()` returns the context dict. Classification thresholds load from the **probe checkpoint's `thresholds` key** (fallback: module constants `_VALENCE_POS` etc., kept in sync by `train_probe.recalibrate_decoder`). `is_ready()` gates on ≥3 sessions. |
| **Upstream** | `checkpoints/valence_arousal_probe.pt`; state vector from transition model; `arc_history` from `StateStore`. |
| **Downstream** | (v, a) → `MemoryStore.retrieve` emotional axis + `StateStore.save`; `state_summary` → LLM system prompt (Condition B); used live by `_InMemoryStore(decoder=…)` in `build_contexts.py`. |

### `kokoro/store.py`
| | |
|---|---|
| **Does** | `StateStore` — SQLite (WAL) persistence of per-user `state_vector` (blob), `session_count`, latest (v, a), `arc_history` (JSON, capped at 20). `save/load/get_info/reset/delete/list_users`. |
| **Upstream** | `WorldMemory.update` (after each transition step). |
| **Downstream** | `WorldMemory._get_current_state` (state for next transition + retrieval), `get_context` (arc_history/session_count → decoder trend & readiness). |
| **Watch out** | `save()` increments `session_count` every call; arc_history cap of 20 bounds trend windows. |

### `kokoro/retrieval.py`
| | |
|---|---|
| **Does** | `MemoryStore` — ChromaDB-backed hybrid store. `add_session` (state_vector optional; monotonic timestamps). `retrieve` scores: semantic = MiniLM cosine rescaled to [0,1] (disabled when `query_embedding=None`); emotional = 1 − L2((v,a)cur,(v,a)stored)/√8 (state-cosine fallback is known non-discriminative); recency = exp(−rank_age/τ) weighted by `recency_weight`; `adaptive=True` gates alpha via module-level `adaptive_alpha()` (dispersion of stored (v,a): homogeneous → α=1). Returns per-axis scores + `effective_alpha`. |
| **Upstream** | `WorldMemory` (add + retrieve with decoded v/a); test suites. |
| **Downstream** | `relevant_memories` in the context dict → LLM prompt. |
| **Sideways** | Mirrored by two experiment `_InMemoryStore` classes (`build_contexts.py`, `run_evaluation.py`) — **keep the three scoring implementations in lockstep**. |

### `kokoro/memory.py`
| | |
|---|---|
| **Does** | `WorldMemory` — the orchestrator/public API. `update(turns)`: encode → transition → decode → persist (StateStore) → index (MemoryStore, embedding sliced to 384) → return `get_context("")`. `get_context(msg, alpha)`: decode state + retrieve (query None ⇒ emotional/recency-only). `predict_next()`: decode current state as next-session forecast. Constructor auto-detects VAD from checkpoint `model_config.session_dim`; params: `top_k, alpha, min_sessions, recency_weight, adaptive_alpha, retrieval_fn` (pluggable backend). |
| **Upstream** | Both checkpoints; user session turns; current message. |
| **Downstream** | Context dict consumed by the companion LLM, `examples/*`, `demo.py`, `experiments/build_contexts.py`. |

---

## 2. Data — `data/`

### `data/prepare.py`
Loads/caches EmpatheticDialogues tarball → `list[LabeledSession(conv_id, turns, valence, arousal, emotion_label)]` using `EMOTION_TO_CIRCUMPLEX` (32 labels → (v,a); **arousal floor ≈ −0.30**). Also `load_mental_health_sessions()` (supplementary distress-region pool, unused by default). Upstream: FB download. Downstream: `construct_trajectories.py`. Sideways: `LabeledSession` dataclass is the pool contract also produced by `low_arousal_pool.py`.

### `data/low_arousal_pool.py` *(new)*
Template-generated deep-low-arousal sessions (Q4-deep depleted + Q2-deep serene, arousal −0.35…−0.80) as `LabeledSession`s. Upstream: nothing (self-contained templates). Downstream: appended to the pool in `construct_trajectories.py --augment-low-arousal`. Reason: breaks the ED arousal floor (documented cause of arousal r ceiling).

### `data/arc_templates.py`
16 `ArcTemplate`s (sequences of `CircumplexZone`s + sampling weight), incl. 5 arousal-primary arcs (3 original + `depressive_shutdown`, `unwinding_to_serenity` which need the augmented pool). Upstream: psychology-literature-grounded constants. Downstream: `construct_trajectories.py` sampling; `per_arc_val_loss.py` / `arc_ood_eval.py` classify arcs by name.

### `data/construct_trajectories.py`
Samples an arc template, picks pool sessions inside each zone (±tolerance, one relaxation retry; `max_uses_per_conv=8` soft cap — the cause of the historical train/val leakage), assigns week offsets → `Trajectory` JSON. CLI: `n`, `--out`, `--seed`, `--max-uses`, `--augment-low-arousal [--n-low-arousal]`, `--holdout-conv-fraction` (emits conversation-disjoint `<out>_val.json`). Downstream: `trajectories_10k.json` / `trajectories_sample.json` consumed by all of training/ and diagnostics/.

---

## 3. Training — `training/`

### `training/split.py` *(new)*
`conv_disjoint_split(trajs, val_fraction, seed, enforce_disjoint)` — greedy conversation-partition split (zero shared conv_ids; on 10k data: 4658/232/5110 dropped) + `leakage_report()` quantifying legacy contamination (100%). Upstream: trajectory JSON. Downstream: `train.py`, `train_probe.py` (both default to it; `--allow-leaky-split` reproduces legacy numbers).

### `training/train.py`
Trains the transition model (`--arch gru|mlp`, default gru) with `vicreg_loss` = VICReg (cosine sim + variance(γ=1) + covariance) **+ state-utility margin loss** (`--util-weight/--util-margin`: warm-step predictions must beat zero-state predictions — directly optimizes the ablation gap) **+ auxiliary next-session VAD loss** (`--vad-weight`, targets = embedding dims 384:386). `rollout_batch`/`_pad_batch` vectorize rollouts across the batch (CPU retrain ≈ 25 min). `evaluate(..., return_ablation_gap=True)` logs the gap per epoch; best checkpoint selected by `val_loss − gap`. `--val-data` accepts the conversation-disjoint `*_val.json`. Saves `model_config{state_dim, session_dim, hidden_dim, arch}` + `.history.json` (incl. per-epoch ablation_gap). Downstream: `checkpoints/transition_v2.pt` → everything at inference/diagnostics.

### `training/train_probe.py`
Rolls every trajectory through the frozen transition model (arch auto-detected via `load_transition_checkpoint`), collects (state, v, a) samples, fits `ValenceArousalProbe` (Linear 384→2, MSE), reports Pearson r (current: 0.770 valence / 0.672 arousal on the disjoint val file via `--val-data`), runs a polarity sanity check, `compute_thresholds()` → percentile thresholds **saved into the probe checkpoint**, `recalibrate_decoder()` patches decoder.py fallback constants. Downstream: `checkpoints/valence_arousal_probe_v2.pt` → `StateDecoder`, `demo.py`, `longitudinal_sim.py`.

### `training/validate.py`
Paper Table 1: separation ratio (model vs last-session baseline vs EWMA vs identity), shuffled-label control, silhouette, PR, PC1 polarity correlation. Reads trajectories + checkpoint; standalone report printer.

---

## 4. Inference extras

- **`demo.py`** — Gradio circumplex demo; loads both checkpoints directly (own loader, honours session_dim), plots the (v,a) trajectory per typed session.
- **`examples/basic_usage.py` / `with_custom_retrieval.py` / `with_openai.py`** — API usage samples over `WorldMemory` (incl. `retrieval_fn` plug-in point).
- **`PIPELINE.md`, `docs/ARCHITECTURE.md`, `README.md`** — prose docs; `research_report.md` = full empirical record; `paper_notes.md` = pending paper edits; `improvements_report.md` = this upgrade cycle.

---

## 5. Evaluation — `experiments/`

### `experiments/scenarios.py`
20 handwritten multi-session scenarios (4–5 sessions, 8 arc types) with `new_message` + `expected_awareness`. Downstream: `build_contexts.py` (default mode).

### `experiments/scenarios_long.py`
6 long scenarios (10–12 sessions), two-phase by design (Phase 1 negative / Phase 2 recovered, same topic keywords) to force semantic-vs-emotional retrieval divergence. Downstream: `build_contexts.py --long`.

### `experiments/build_contexts.py` — Stage 1 (torch process)
Patches `chromadb.PersistentClient → EphemeralClient`, swaps `WorldMemory._mem_store` for `_InMemoryStore(decoder)` (pure-numpy mirror of MemoryStore scoring: [0,1] semantic, VAD-L2 emotional via live decoder, recency, adaptive; accepts `query_embedding=None`). Per scenario: feed sessions via `update()`, then emit `context_a` (α=1.0), `context_b` (α=0.6 + state summary), `context_c` (last-k recency baseline), + retrieval-divergence stats → `contexts[_long].json`. Flags: `--test` (3 scenarios), `--long`.

### `experiments/run_llm_eval.py` — Stage 2 (network process, no torch)
Reads contexts JSON; builds system prompts (`build_system_a` memories-only, `build_system_b` +state summary); generates responses via Groq (`llama-3.3-70b`); judges with 3 models × **2 presentation orders** (`_judge_counterbalanced`: flip-inconsistent judge ⇒ TIE), majority vote → verdict; optional `--with-recency-baseline` adds the B-vs-C arm; summary prints win rates, **Wilson CI**, **exact sign test**, inter-judge agreement, per-arc breakdown → `results[_long].json`. Needs `GROQ_API_KEY`.

### `experiments/run_evaluation.py`
Legacy single-process pipeline (Stage 1+2 in one). Kept runnable (store signature updated) but superseded by the two-stage flow.

### Generated: `contexts*.json`, `results*.json`, `eval_report*.md`.

---

## 6. Diagnostics — `diagnostics/`

### `diagnostics/common.py` *(new — start here)*
`load_transition_model(path)` (honours `session_dim`), `make_encoder(cfg)` (VAD iff checkpoint used it), `build_embedding_cache(trajs, encoder)` (training-identical preprocessing). Downstream: all repaired diagnostics. Rule: compare state vs embedding only after `emb[:model.state_dim]`.

| Script | Question it answers | Key output / flag |
|---|---|---|
| `state_ablation.py` | Does the recurrent state add anything over session-only? | gap = ablated − normal loss; **flag < 0.05. Currently 0.0001 → FLAGGED** |
| `topic_leakage.py` | Is the emotional retrieval axis just semantic retrieval again? | mean Spearman r between axis rankings; flag > 0.8 |
| `per_arc_val_loss.py` | Can the model predict along the arousal axis? | arousal-primary vs valence-primary arc loss gap; flag > 0.05 |
| `norm_drift.py` | Does ‖state‖ explode/vanish over 50+ sessions (no L2 norm)? | norm ratio step50/step5; flags >10× or <0.1× (uses MSC data, synthetic fallback) |
| `arc_separation.py` | Do final states cluster by arc type? | separation ratio vs last-session & EWMA baselines + shuffled control (3.111 reported) |
| `probe_generalization.py` | Does the probe transfer to naturalistic companion text? | predicted (v,a) on 24 handcrafted scenarios vs expected quadrants |
| `eval_worldmodel.py` | Prediction vs naive "repeat last session" baseline | model_sim − baseline_sim per arc |
| `longitudinal_sim.py` | Could it track simulated users session-by-session? | (v,a) MAE/RMSE, cold-start latency, arc-change detection lag |
| `arc_ood_eval.py` | Does it generalize to arcs withheld at training (`train.py --holdout-arcs`)? | per-arc delta vs full-model baseline |

All read `data/trajectories_10k.json` + `checkpoints/transition_v1.pt` (CLI-overridable).

---

## 7. Figures & misc

- `figures/fig0_architecture.py` (system diagram), `visualize_step1..4.py` (collapse → post-fix state space → retrieval eval visuals), `animate_world_model.py`, `validate_msc.py` (MSC external-data check). Read checkpoints/history JSONs; write `figures/*.png`.
- `tests/` — pytest suites per core module (89 tests; transition tests assert the *unnormalized* output contract).
- `checkpoints/` — **`transition_v2.pt` (GRU, production: epoch 43, ablation gap +0.158)**, **`valence_arousal_probe_v2.pt`** (+`thresholds` key; r=0.770/0.672); `transition_v1.pt`/`valence_arousal_probe.pt` are the legacy MLP artifacts; `train_v2.log`/`probe_v2.log`/diagnostic logs are the evidence trail. (`kokoro/checkpoints/` is a stale copy — the code resolves `checkpoints/` at the project root; prefer the root.)
- `hehe.txt`, `report.txt`, `research_critique.txt`, `review/` — scratch/notes, not part of the pipeline.

---

## 8. Trace cheat-sheet ("I'm in X, where did this come from?")

- **A number in `results_long.json`** ← `run_llm_eval.py` (judging) ← `contexts_long.json` ← `build_contexts.py` (`WorldMemory` + `_InMemoryStore`) ← `scenarios_long.py` + both checkpoints.
- **A (valence, arousal) anywhere at inference** ← `StateDecoder._run_probe` ← probe ckpt ← `train_probe.py` ← states from transition ckpt rollouts ← `train.py` ← `trajectories_10k.json` ← `construct_trajectories.py` ← `prepare.py` label map (and, post-augmentation, `low_arousal_pool.py`).
- **The state summary string** ← `decoder._build_summary` ← thresholds (probe ckpt) + trend slope ← `arc_history` ← `StateStore.save` calls in `WorldMemory.update`.
- **Which memories the LLM saw** ← `MemoryStore.retrieve` scoring (α, γ, adaptive) ← decoded (v,a) + query embedding ← `get_context(message)`.
- **Why a diagnostic number changed** — check checkpoint (`session_dim`?), data file, and split mode (`--allow-leaky-split`?) before suspecting the script.
