# Kokoro — Architecture Reference

> Generated from source. Every class, function, default, threshold, and data format documented.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Complete Data Flow](#3-complete-data-flow)
4. [Component Deep Dive — kokoro/](#4-component-deep-dive--kokoro)
5. [Data Pipeline — data/](#5-data-pipeline--data)
6. [Training — training/](#6-training--training)
7. [Configuration and Defaults](#7-configuration-and-defaults)
8. [Data Formats](#8-data-formats)
9. [Dependencies](#9-dependencies)
10. [Call Graph](#10-call-graph)
11. [Known Limitations](#11-known-limitations)
12. [Extension Points](#12-extension-points)

---

## 1. Project Overview

Kokoro is a pluggable emotional trajectory memory layer for AI companions. It sits between a conversation history and an LLM and maintains a continuous, evolving representation of a user's emotional state across sessions. At each session boundary Kokoro encodes the conversation with a sentence transformer, feeds that embedding through a trained transition model (an MLP world model) to advance the user's state vector, decodes the state into valence/arousal coordinates via a linear probe, persists everything in SQLite, indexes the session in a ChromaDB vector store, and on the next request retrieves emotionally and semantically relevant past sessions to inject into the LLM system prompt.

**Core research claim:** A recurrent world model applied sequentially to conversation-session embeddings produces state representations whose arc-type separation ratio (between-arc / within-arc cosine distance) is 3.11× baseline and whose first principal component explains 82.6% of variance, demonstrating that emotional trajectory structure is linearly decodable from the learned representation.

**Full data flow:**

```
Raw conversation turns
        │
        ▼  SessionEncoder.encode()
   session_embedding (384-dim float32)
        │
        ▼  TransitionModel.forward(state, session_embedding)
   updated_state (384-dim, L2-normalised, unit sphere)
        │
        ├──▶  StateStore.save()         → SQLite (state + arc_history)
        │
        ├──▶  StateDecoder.decode()     → valence, arousal, trend, state_summary
        │
        ├──▶  MemoryStore.add_session() → ChromaDB (semantic + emotional index)
        │
        └──▶  MemoryStore.retrieve()   → top-k past sessions (hybrid score)
                    │
                    ▼
          context dict → LLM system prompt injection
```

---

## 2. Directory Structure

```
Worldmodel_memory/
│
├── kokoro/                     # Installable package — the public API
│   ├── __init__.py             # Exports WorldMemory; version = "0.1.0"
│   ├── encoder.py              # SessionEncoder: turns → 384-dim embedding
│   ├── transition.py           # TransitionModel: (state, emb) → state (MLP)
│   ├── store.py                # StateStore: SQLite persistence of state vectors
│   ├── decoder.py              # StateDecoder: state → natural-language summary
│   ├── retrieval.py            # MemoryStore: ChromaDB hybrid RAG
│   └── memory.py               # WorldMemory: public API integrating all above
│
├── data/                       # Dataset loading and synthetic trajectory construction
│   ├── arc_templates.py        # 11 emotional arc definitions in circumplex space
│   ├── prepare.py              # EmpatheticDialogues loader + circumplex mapping
│   ├── construct_trajectories.py # Synthetic trajectory builder + CLI
│   ├── trajectories_sample.json  # 300-500 sample trajectories (generated)
│   └── trajectories_10k.json     # 10,000 training trajectories (generated)
│
├── training/                   # Model training and validation scripts
│   ├── train.py                # Transition model training loop
│   ├── train_probe.py          # Linear probe (valence/arousal) training
│   └── validate.py             # Table 1 validation: separation + polarity metrics
│
├── figures/                    # Visualisation scripts and saved figures
│   ├── visualize_step1.py      # Fig 1-3: circumplex scatter, arc paths, distribution
│   ├── visualize_step2.py      # Fig 4-5: t-SNE embeddings, valence/PC1 correlation
│   ├── visualize_step3.py      # Fig 6-8: loss curves, state trajectories, arc separation
│   └── validate_msc.py         # OOD validation on nayohan/multi_session_chat
│
├── checkpoints/                # Trained model weights (gitignored or tracked by LFS)
│   ├── transition_v1.pt            # Transition model trained on 300 trajectories
│   ├── transition_v1_10k.pt        # Transition model trained on 10,000 trajectories
│   ├── transition_v1.history.json  # Loss history for 300-traj run
│   ├── transition_v1_10k.history.json # Loss history for 10k-traj run
│   └── valence_arousal_probe.pt    # Linear probe checkpoint
│
├── docs/
│   └── ARCHITECTURE.md         # This file
│
├── pyproject.toml              # Package metadata and dependencies
└── hehe.txt                    # (scratch file)
```

---

## 3. Complete Data Flow

### 3.1 `WorldMemory.update(session_turns)` — step by step

```
Input: session_turns = [{"role": "user"|"assistant", "content": "..."}, ...]
```

| Step | Function | Input | Output | Side effects |
|------|----------|-------|--------|--------------|
| 1 | `SessionEncoder.encode(session_turns)` | list of turn dicts | `session_emb: np.ndarray (384,) float32` | none |
| 1a | `encoder._prepare_turns(turns)` | turns | `texts: list[str], weights: list[float]` | none |
| 1b | `decode_parlai_artifacts(content)` | raw text | cleaned text | none |
| 1c | `encoder.model.encode(texts, ...)` | batch of strings | `embs: np.ndarray (n_turns, 384)` | none |
| 1d | weighted mean pool | embs + weights | `session_emb: (384,) float32` | none |
| 2 | `StateStore.load(user_id)` | user_id | `current_state: np.ndarray (384,) or None` | SQLite READ |
| 2a | if None → `TransitionModel.initial_state()` | — | `zeros: torch.Tensor (384,)` | none |
| 3 | `TransitionModel.forward(state_t, emb_t)` | `(384,), (384,)` tensors | `new_state: (384,) L2-normalised` | none |
| 3a | `torch.cat([state, session_emb], dim=-1)` | two (384,) → (768,) | (768,) | none |
| 3b | `self.net(x)` — Linear→LayerNorm→ReLU→Dropout→Linear | (768,) | (384,) | none |
| 3c | `F.normalize(x, dim=-1)` | (384,) | unit sphere (384,) | none |
| 4 | `StateDecoder.decode(new_state, arc_before, session_count+1)` | state, history, count | `{valence, arousal, trend, ...}` | none |
| 4a | `decoder._run_probe(state)` | (384,) | `(valence, arousal)` clamped to [-1.5, 1.5] | none |
| 4b | `_linear_slope(valences)` | list[float] | slope float | none |
| 4c | `_build_summary(...)` | continuous values | natural-language str | none |
| 5 | `StateStore.save(user_id, new_state, valence, arousal)` | — | — | SQLite WRITE: upsert user_states; increments session_count; appends arc_history |
| 6 | `_session_summary(session_turns)` | turns | first 200 chars of user turns joined | none |
| 7 | `MemoryStore.add_session(user_id, session_id, text, emb, v, a)` | — | — | ChromaDB WRITE: upserts document with embedding + metadata |
| 8 | `WorldMemory.get_context("")` | — | context dict | SQLite READ + ChromaDB READ |

### 3.2 `WorldMemory.get_context(current_message)` — step by step

```
Input: current_message: str  (may be empty)
```

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 1 | `StateStore.load(user_id)` | user_id | `state_vec: (384,)` or zeros |
| 2 | `StateStore.get_info(user_id)` | user_id | `{session_count, arc_history, valence, arousal, ...}` or None |
| 2a | if info is None → return cold-start dict (ready=False, all zeros/None) | — | — |
| 3 | `StateDecoder.decode(state_vec, arc_history, session_count)` | — | `{state_summary, valence, arousal, trend, ...}` |
| 4 | `ready = session_count >= self._min_sessions` | — | bool |
| 5 | if ready and current_message.strip(): `SessionEncoder.encode_text(message)` → `query_emb` | str | (384,) |
| 5a | if ready and no message: `query_emb = state_vec.copy()` | — | (384,) |
| 6 | `MemoryStore.retrieve(user_id, query_emb, state_vec, top_k, alpha)` | — | list of result dicts |
| 6a | `col.get(where={"user_id": user_id}, include=[...])` | — | all user sessions |
| 6b | `_cosine_sim_batch(query_emb, stored_embs)` | — | semantic_scores (N,) |
| 6c | `_cosine_sim_batch(state_vec, stored_embs)` | — | emotional_scores (N,) |
| 6d | `alpha * semantic + (1-alpha) * emotional` | — | combined_scores (N,) |
| 6e | `np.argsort(-combined_scores)[:top_k]` | — | top indices |
| 7 | Return dict | — | `{state_summary, relevant_memories, valence, arousal, trend, session_count, ready}` |

---

## 4. Component Deep Dive — kokoro/

### 4.1 `kokoro/__init__.py`

```python
__version__ = "0.1.0"
from kokoro.memory import WorldMemory
__all__ = ["WorldMemory"]
```

Exports only `WorldMemory`. All other classes are importable directly from their modules but are not part of the public surface.

---

### 4.2 `kokoro/encoder.py`

#### `decode_parlai_artifacts(text: str) -> str`

Replaces ParlAI punctuation tokens with real characters. Applied before encoding so the sentence transformer sees natural text regardless of source dataset.

| Token | Replacement |
|-------|-------------|
| `_comma_` | `,` |
| `_period_` | `.` |
| `_exclamation_` | `!` |
| `_question_` | `?` |
| `_semicolon_` | `;` |
| `_colon_` | `:` |
| `_apostrophe_` | `'` |
| `_newline_` | ` ` |
| `_tab_` | ` ` |
| `" {2,}"` (multi-space) | `" "` |

All patterns absorb optional preceding whitespace. Case-insensitive.

---

#### `class SessionEncoder`

**Purpose:** Converts a conversation session (list of turns) into a single 384-dimensional embedding.

**Constructor:** `SessionEncoder(model_name=MODEL_NAME, user_weight=2.0, device=None)`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model_name` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model ID |
| `user_weight` | `2.0` | Relative weight for user turns during pooling |
| `device` | `None` (auto) | `"cpu"`, `"cuda"`, or None |

Model is **lazy-loaded** on first call to `encode()` or `encode_text()`.

---

**`encode(turns: Sequence[dict[str, str]]) -> np.ndarray`**

- **Input:** List of `{"role": "user"|"assistant", "content": "..."}` dicts. At least one non-empty turn required.
- **Returns:** `(384,) float32` numpy array.
- **Raises:** `ValueError` if turns is empty or all content is blank after cleaning.
- **Algorithm:**
  1. `_prepare_turns(turns)` → cleans each turn with `decode_parlai_artifacts`, assigns weight (2.0 for user, 1.0 for assistant), drops empty turns.
  2. Single batch call to `self.model.encode(texts, ...)`.
  3. Weighted mean pool: `(embs * weights[:, None]).sum(axis=0)` after normalizing weights to sum=1.
- **Note:** Does NOT L2-normalize the output. The transition model receives raw encoder outputs.

---

**`encode_text(text: str) -> np.ndarray`**

- **Input:** Single cleaned string.
- **Returns:** `(384,) float32` numpy array.
- **Used by:** `WorldMemory.get_context()` to encode the current message for semantic retrieval.
- **Raises:** `ValueError` if text is blank.

---

**`_prepare_turns(turns) -> tuple[list[str], list[float]]`**

Internal. Returns `(texts, weights)` after cleaning and dropping empty turns.

---

**Module-level helpers:**

- `get_encoder() -> SessionEncoder` — Returns module-level singleton, loading on first call.
- `encode_session(turns) -> np.ndarray` — Convenience wrapper over `get_encoder().encode(turns)`.

---

### 4.3 `kokoro/transition.py`

#### `class TransitionModel(nn.Module)`

**Purpose:** The world model core. Maps (current_state, session_embedding) → updated_state on the unit hypersphere.

**Architecture:**

```
Input: concat(state_vector, session_emb)  →  (768,)
Layer 1: Linear(768 → 512) → LayerNorm(512) → ReLU → Dropout(0.1)
Layer 2: Linear(512 → 384)
Output:  F.normalize(x, dim=-1)           →  (384,) on unit sphere
```

**Constants:**
- `STATE_DIM = 384`

**Constructor:** `TransitionModel(state_dim=384, hidden_dim=512, dropout=0.1)`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `state_dim` | `384` | Matches all-MiniLM-L6-v2 output dim |
| `hidden_dim` | `512` | Intermediate layer width |
| `dropout` | `0.1` | Applied between the two linear layers |

Weight initialization: Xavier uniform for Linear layers, ones/zeros for LayerNorm.

---

**`forward(state: torch.Tensor, session_emb: torch.Tensor) -> torch.Tensor`**

- **Input:** `state` of shape `(..., 384)` (zeros for new user), `session_emb` of shape `(..., 384)`.
- **Returns:** Updated state of shape `(..., 384)`, **L2-normalized** (always ‖z‖ = 1).
- **No side effects.**

---

**`TransitionModel.initial_state(batch_size=1, device=None) -> torch.Tensor`**

- **Static method.**
- Returns `torch.zeros((384,))` for `batch_size=1`, else `torch.zeros((batch_size, 384))`.
- The all-zero initial state is the cold-start: the model receives `[0...0 | e_0]` on the first update.

---

**`parameter_count() -> int`**

Returns total number of trainable parameters. For default config: 768×512 + 512 + 512×384 + 384 = ~590k parameters.

---

**LSTM upgrade path (commented out):**

`TransitionModelLSTM` is defined as a docstring block inside the module docstring. Its `forward` signature is `(state, session_emb, hidden=None) -> (new_state, hidden)`. To activate: uncomment `TransitionModel = TransitionModelLSTM`. All callers in `kokoro/memory.py` and `training/train.py` are compatible with no other changes because they use the same `forward(state, emb)` call pattern (the LSTM ignores `state` and uses `hidden` instead, which is initialized internally).

---

### 4.4 `kokoro/store.py`

#### `class StateStore`

**Purpose:** Persistent, per-user emotional state storage using SQLite.

**Constructor:** `StateStore(db_path=None)`

| Parameter | Default |
|-----------|---------|
| `db_path` | `~/.kokoro/kokoro.db` |

Creates parent directory automatically. Initializes the `user_states` table if it does not exist. Enables WAL journal mode on every connection.

---

**`save(user_id: str, state_vector: np.ndarray, valence: float, arousal: float) -> None`**

- **Behavior:**
  - First call for a user: `INSERT` with `session_count=1`, `arc_history=[[valence, arousal]]`.
  - Subsequent calls: `UPDATE` — increments `session_count`, appends `[valence, arousal]` to `arc_history` (capped at 20 entries), updates `updated_at`.
- **Side effect:** SQLite write (atomic transaction).
- **Raises:** `ValueError` if `state_vector.ndim != 1`.
- `state_vector` is serialized as `float32` bytes via `ndarray.tobytes()`. `state_dim` is stored alongside for reconstruction validation.

---

**`load(user_id: str) -> Optional[np.ndarray]`**

- **Returns:** `(state_dim,) float32` numpy array, or `None` if user not found.
- **Raises:** `ValueError` if stored blob size doesn't match stored `state_dim` (corruption / model mismatch).
- Uses `np.frombuffer(...).copy()` — the `.copy()` is required because `frombuffer` returns a read-only view.

---

**`get_info(user_id: str) -> Optional[dict]`**

- **Returns:** Dict or None. Dict keys:
  - `session_count: int`
  - `valence: float`
  - `arousal: float`
  - `arc_history: list[list[float]]` — list of `[valence, arousal]` pairs, oldest first, max 20 entries
  - `created_at: str` — ISO-8601 UTC
  - `updated_at: str` — ISO-8601 UTC
  - `state_dim: int`
- No state vector loaded (efficient).

---

**`reset(user_id: str) -> bool`**

- Sets `state_vector` to zeros, `session_count` to 0, `arc_history` to `[]`, valence/arousal to 0.0.
- User record is **retained** (so the next `save` will INSERT rather than confusingly fail uniqueness).
- **Returns:** `True` if user existed and was reset, `False` if not found.

---

**`delete(user_id: str) -> bool`**

- Permanently removes user row.
- **Returns:** `True` if deleted, `False` if not found.

---

**`list_users() -> list[str]`**

- Returns all `user_id` strings ordered by `created_at` ascending.

---

**Internal helpers:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_now_iso()` | `-> str` | UTC ISO-8601 timestamp |
| `_vec_to_blob(arr)` | `np.ndarray -> bytes` | `arr.astype(float32).tobytes()` |
| `_blob_to_vec(blob, state_dim)` | `bytes, int -> np.ndarray` | `frombuffer(..., dtype=float32).copy()` |
| `_append_arc_history(json_str, v, a)` | `str, float, float -> str` | Appends `[v, a]`, trims to `_ARC_HISTORY_MAX=20` |

---

**Constants:**

| Name | Value |
|------|-------|
| `_DEFAULT_DB` | `~/.kokoro/kokoro.db` |
| `_ARC_HISTORY_MAX` | `20` |
| `_STATE_DTYPE` | `np.dtype("float32")` |

---

### 4.5 `kokoro/decoder.py`

**File encoding:** `# -*- coding: utf-8 -*-` (required because Windows cp1252 cannot decode box-drawing chars in docstrings).

#### `class _ValenceArousalProbe(nn.Module)` (internal)

```python
Linear(384, 2, bias=True)
```

Identical architecture to `ValenceArousalProbe` in `training/train_probe.py`. The `__repr__` uses `"->"` not `"→"` to avoid `UnicodeEncodeError` on Windows consoles.

---

#### `class StateDecoder`

**Purpose:** Converts a state vector + arc_history → structured context dict with natural-language summary.

**Constructor:** `StateDecoder(probe_path=None)`

| Parameter | Default |
|-----------|---------|
| `probe_path` | `<project_root>/checkpoints/valence_arousal_probe.pt` |

Loads the probe checkpoint, sets `probe.eval()`. Raises `FileNotFoundError` if checkpoint absent.

---

**`is_ready(session_count: int) -> bool`**

Returns `session_count >= _MIN_SESSIONS` (i.e., `>= 3`). Controls whether `decode()` produces a summary.

---

**`decode(state_vector: np.ndarray, arc_history: list[list[float]], session_count: int) -> dict`**

- **Input:**
  - `state_vector`: `(state_dim,)` float32, L2-normalised.
  - `arc_history`: list of `[valence, arousal]` pairs from `StateStore.get_info()`.
  - `session_count`: from `StateStore.get_info()`.
- **Returns:** Dict with keys:
  - `state_summary: str | None` — None when `session_count < 3`
  - `valence: float` — rounded to 4 decimal places
  - `arousal: float` — rounded to 4 decimal places
  - `trend: str` — `"improving"` | `"declining"` | `"stable"`
  - `trend_strength: float` — absolute linear slope over arc_history, rounded to 6 dp
  - `mean_valence: float` — mean over arc_history, or current probe output if history < 3
  - `mean_arousal: float` — same
  - `session_count: int`
  - `ready: bool`
- **Raises:** `ValueError` if `state_vector.ndim != 1`.

---

**`_run_probe(state_vector: np.ndarray) -> tuple[float, float]`**

1. `torch.from_numpy(state_vector.astype(float32)).unsqueeze(0)` → (1, 384) tensor.
2. `self._probe(x).squeeze(0).numpy()` → (2,) array.
3. `np.clip(pred, -1.5, 1.5)` — clamp to prevent extreme probe outputs from aligned vectors.
4. Returns `(float(pred[0]), float(pred[1]))` — (valence, arousal).

---

**`_linear_slope(values: list[float]) -> float`**

Closed-form ordinary least-squares slope. No scipy dependency.

```
slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
```

Returns `0.0` for sequences of length < 2 or degenerate denominator (`|denom| < 1e-12`).

---

**Threshold constants:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `_MIN_SESSIONS` | `3` | Min sessions before summary produced |
| `_MIN_TREND_ENTRIES` | `3` | Min arc_history entries to compute slope |
| `_VALENCE_POS` | `+0.30` | Valence > this → "positive" |
| `_VALENCE_NEG` | `-0.30` | Valence < this → "negative" |
| `_AROUSAL_HIGH` | `+0.30` | Arousal > this → "activated" |
| `_AROUSAL_LOW` | `-0.10` | Arousal < this → "low-energy" |
| `_SLOPE_IMP` | `+0.02` | Slope > this → "improving" |
| `_SLOPE_DEC` | `-0.02` | Slope < this → "declining" |

---

**`_build_summary(valence, arousal, trend, trend_strength, mean_valence, mean_arousal, session_count, n_history) -> str`**

Generates natural-language text from continuous values. The logic branches on `(trend, v_word, a_word)` combinations, producing 9 distinct sentence structures. Appends a historical mean note if `mean_v_word != v_word` (recent shift detected). `trend_strength` modulates adverbs: `"gradually"` if `< 0.05`, `"notably"/"noticeably"` otherwise.

---

**Helper functions:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `_classify_trend(slope)` | `float -> str` | `"improving"` / `"declining"` / `"stable"` |
| `_describe_valence(v)` | `float -> str` | `"positive"` / `"negative"` / `"neutral"` |
| `_describe_arousal(a)` | `float -> str` | `"activated"` / `"low-energy"` / `"moderate"` |

---

### 4.6 `kokoro/retrieval.py`

#### `class MemoryStore`

**Purpose:** ChromaDB-backed hybrid semantic + emotional memory store.

**Constructor:** `MemoryStore(collection_name="kokoro", persist_dir=None)`

| Parameter | Default |
|-----------|---------|
| `collection_name` | `"kokoro"` |
| `persist_dir` | `~/.kokoro/memories/` |

Creates a `chromadb.PersistentClient` and gets-or-creates the collection with `{"hnsw:space": "cosine"}`.

---

**`add_session(user_id, session_id, session_text, session_embedding, valence, arousal) -> None`**

- **Side effect:** `col.upsert(...)` — adds or replaces the document.
- Document ID: `hashlib.sha1(f"{user_id}\x00{session_id}".encode()).hexdigest()`
- Stored embedding: `session_embedding.tolist()` (float32 list)
- Metadata: `{user_id, session_id, valence: float, arousal: float, timestamp: float}`
- Validates embedding shape with `_validate_embedding()`: must be `(384,)`.

---

**`retrieve(user_id, query_embedding, state_vector, top_k=5, alpha=0.6) -> list[dict]`**

- **Raises:** `ValueError` if any embedding is not 1-D, or `alpha` not in [0, 1].
- **Algorithm:**
  1. `col.get(where={"user_id": user_id}, include=["embeddings", "documents", "metadatas"])` — fetches ALL user sessions (no HNSW top-k pre-filtering; correct for O(10-200) sessions per user).
  2. `stored_embs = np.array(result["embeddings"], dtype=float32)` — shape `(N, 384)`.
  3. `semantic_scores = _cosine_sim_batch(query_embedding, stored_embs)` — shape `(N,)`.
  4. `emotional_scores = _cosine_sim_batch(state_vector, stored_embs)` — shape `(N,)`.
  5. `combined = alpha * semantic + (1 - alpha) * emotional`.
  6. `np.argsort(-combined)[:min(top_k, N)]` → top indices.
- **Returns:** List of dicts with keys: `session_text`, `session_id`, `semantic_score`, `emotional_score`, `combined_score`, `valence`, `arousal`. Sorted by `combined_score` descending.

---

**`get_user_sessions(user_id: str) -> list[dict]`**

- Returns all sessions for user, sorted by `timestamp` ascending.
- Each dict: `{session_id, session_text, valence, arousal, timestamp}`.

---

**`delete_user(user_id: str) -> int`**

- Deletes all documents matching `user_id`.
- **Returns:** Number of documents deleted (0 if user not found).

---

**Internal helpers:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_l2_normalize(v)` | `ndarray -> ndarray` | Unit-length copy; raises if norm < 1e-12 |
| `_cosine_sim_batch(query, stored)` | `(D,), (N,D) -> (N,)` | L2-normalizes both before dot product |
| `_doc_id(user_id, session_id)` | `str, str -> str` | SHA1 hex of `user_id\x00session_id` |
| `_validate_embedding(arr, name)` | `ndarray, str -> ndarray` | Checks 1-D shape `(384,)`, returns float32 copy |

---

### 4.7 `kokoro/memory.py`

#### `class WorldMemory`

**Purpose:** The single public API class. Wires together all five components.

**Constructor:**

```python
WorldMemory(
    user_id: str,
    *,
    db_path=None,
    persist_dir=None,
    checkpoint_path=None,
    probe_path=None,
    top_k: int = 5,
    alpha: float = 0.6,
    min_sessions: int = 3,
    collection_name: str = "kokoro",
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `user_id` | (required) | Unique string identifier for the user |
| `db_path` | `~/.kokoro/kokoro.db` | SQLite path for StateStore |
| `persist_dir` | `~/.kokoro/memories/` | ChromaDB directory for MemoryStore |
| `checkpoint_path` | `checkpoints/transition_v1.pt` | TransitionModel checkpoint |
| `probe_path` | `checkpoints/valence_arousal_probe.pt` | Linear probe checkpoint |
| `top_k` | `5` | Sessions retrieved per query |
| `alpha` | `0.6` | Semantic fraction in hybrid scoring (1-alpha = emotional) |
| `min_sessions` | `3` | Warm-up period before ready=True |
| `collection_name` | `"kokoro"` | ChromaDB collection name |

**Raises:** `ValueError` for invalid `user_id`, `alpha`, `top_k`, `min_sessions`. `FileNotFoundError` if checkpoint not found.

**Initialization order:** `SessionEncoder` → `TransitionModel` (loaded from checkpoint) → `StateDecoder` → `StateStore` → `MemoryStore`.

---

**`update(session_turns: Sequence[dict]) -> dict`**

Full pipeline. See Section 3.1 for step-by-step breakdown.

- **Raises:** `ValueError` if `session_turns` is empty or all content is blank (propagated from `SessionEncoder`).
- **Side effects:** SQLite write (StateStore), ChromaDB write (MemoryStore).
- **Returns:** Result of `get_context("")` called after the update.

The `session_id` stored in MemoryStore is `f"session_{session_count}"` where `session_count` is the post-update value from `StateStore.get_info()`.

---

**`get_context(current_message: str = "") -> dict`**

- **Returns:**

```python
{
    "state_summary":     str | None,   # LLM-ready natural language; None if not ready
    "relevant_memories": list[str],    # session_text strings, top_k max
    "valence":           float,        # [-1.5, 1.5] clamped probe output
    "arousal":           float,        # [-1.5, 1.5] clamped probe output
    "trend":             str,          # "improving" | "declining" | "stable"
    "session_count":     int,
    "ready":             bool,
}
```

- When `ready=False`: `state_summary=None`, `relevant_memories=[]`.
- When `current_message` is blank and `ready=True`: uses `state_vec.copy()` as `query_emb` (state-only retrieval).
- **Side effects:** SQLite read, ChromaDB read.

---

**`reset() -> None`**

- Calls `StateStore.reset(user_id)` + `MemoryStore.delete_user(user_id)`.
- State reverts to cold-start; all memory sessions deleted.

---

**`_load_transition(path: Path) -> TransitionModel`** (static, internal)

Loads from checkpoint dict. Uses `ckpt["model_config"]` (keys `state_dim`, `hidden_dim`) to reconstruct architecture. Falls back to `{state_dim: 384, hidden_dim: 512}` if key absent. Calls `model.eval()`.

---

**`_get_current_state() -> np.ndarray`** (internal)

Returns `StateStore.load(user_id)` or `TransitionModel.initial_state().numpy()` if None.

---

**`_session_summary(turns) -> str`** (internal)

Joins `content` of all user turns with spaces, truncated to 200 characters. Returns `"(no user turns)"` if no user turns found.

---

## 5. Data Pipeline — data/

### 5.1 `data/arc_templates.py`

#### `@dataclass(frozen=True) CircumplexZone`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `valence` | `float` | (required) | Position on valence axis [-1, 1] |
| `arousal` | `float` | (required) | Position on arousal axis [-1, 1] |
| `tolerance` | `float` | `0.25` | Acceptance window half-width |
| `label` | `str` | `""` | Human-readable name (unused in model) |

**`contains(v: float, a: float) -> bool`**

Returns `abs(v - self.valence) <= tolerance and abs(a - self.arousal) <= tolerance`.

---

#### `@dataclass ArcTemplate`

| Field | Type | Meaning |
|-------|------|---------|
| `name` | `str` | Unique arc identifier |
| `description` | `str` | Human-readable arc description |
| `sessions` | `list[CircumplexZone]` | Ordered sequence of target zones |
| `typical_duration_weeks` | `tuple[int, int]` | (min, max) weeks this arc spans |
| `weight` | `float` | Sampling weight (higher = more common) |

---

#### `_ZONES` — 15 named circumplex anchors

| Key | Valence | Arousal |
|-----|---------|---------|
| `elated` | +0.8 | +0.7 |
| `excited` | +0.6 | +0.6 |
| `hopeful` | +0.5 | +0.3 |
| `content` | +0.5 | -0.3 |
| `calm` | +0.3 | -0.5 |
| `grateful` | +0.6 | -0.1 |
| `neutral` | 0.0 | 0.0 (tolerance=0.3) |
| `unsettled` | -0.1 | +0.2 |
| `anxious` | -0.4 | +0.5 |
| `distressed` | -0.6 | +0.6 |
| `panicked` | -0.7 | +0.8 |
| `angry` | -0.5 | +0.7 |
| `sad` | -0.5 | -0.3 |
| `depressed` | -0.7 | -0.5 |
| `exhausted` | -0.4 | -0.6 |
| `lonely` | -0.6 | -0.2 |

---

#### `ARC_TEMPLATES` — 11 arc types

| Name | Sessions | Duration (weeks) | Weight | End valence | Net direction |
|------|----------|------------------|--------|-------------|---------------|
| `gradual_decline` | 8 | 4–8 | 1.5 | -0.7 | declining |
| `slow_recovery` | 9 | 6–12 | 1.5 | +0.5 | improving |
| `acute_stress_stabilization` | 7 | 3–6 | 1.2 | 0.0 | stable |
| `chronic_low_grade_anxiety` | 8 | 6–16 | 1.3 | -0.4 | stable/neg |
| `weekend_oscillation` | 8 | 4–8 | 1.0 | +0.5 | oscillating |
| `grief_arc` | 9 | 8–20 | 1.0 | 0.0 | stable/neg |
| `post_traumatic_growth` | 9 | 8–16 | 0.8 | +0.8 | improving |
| `social_confidence_growth` | 7 | 4–10 | 1.0 | +0.6 | improving |
| `relapse_dip` | 9 | 6–14 | 1.1 | +0.5 | improving |
| `stable_positive` | 6 | 4–12 | 0.7 | +0.3 | stable/pos |
| `stable_negative` | 6 | 6–20 | 0.9 | -0.6 | stable/neg |

**Weighted sampling:** Each template is replicated `max(1, int(weight * 10))` times in the sampling pool.

---

#### `get_template_by_name(name: str) -> ArcTemplate`

Linear scan of `ARC_TEMPLATES`. Raises `KeyError` if not found.

---

#### `summarize_templates() -> None`

Prints a table of all templates (name, session count, duration, weight) to stdout.

---

### 5.2 `data/prepare.py`

#### `EMOTION_TO_CIRCUMPLEX: dict[str, tuple[float, float]]`

Maps all 32 EmpatheticDialogues emotion labels to `(valence, arousal)` coordinates. Values derived from Russell (1980) and Posner, Russell & Peterson (2005). Verified to match the exact label set in the `facebook/empathetic_dialogues` train split via a module-level `assert`.

**Selected mappings:**

| Label | Valence | Arousal | Quadrant |
|-------|---------|---------|---------|
| `joyful` | +0.85 | +0.65 | I |
| `grateful` | +0.65 | -0.05 | II |
| `terrified` | -0.80 | +0.80 | III |
| `devastated` | -0.80 | -0.30 | IV |
| `neutral/surprised` | +0.20 | +0.70 | I/neutral |

---

#### `@dataclass LabeledSession`

| Field | Type | Meaning |
|-------|------|---------|
| `conv_id` | `str` | Unique conversation ID from source |
| `turns` | `list[dict[str, str]]` | `[{"role": ..., "content": ...}]` |
| `valence` | `float` | From `EMOTION_TO_CIRCUMPLEX` |
| `arousal` | `float` | From `EMOTION_TO_CIRCUMPLEX` |
| `emotion_label` | `str` | Original label (debugging only, never fed to model) |

---

#### `_build_turns(group_rows: list[dict]) -> list[dict[str, str]]`

Internal. Sorts rows by `utterance_idx`, converts `speaker_idx` (0=user, 1=assistant) to role strings, drops empty utterances.

---

#### `load_empathetic_dialogues(split="train") -> list[LabeledSession]`

- **Downloads** the EmpatheticDialogues tarball from Facebook AI's servers on first call (`https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz`).
- **Cache location:** `~/.cache/kokoro/empathetic_dialogues/`
- Subsequent calls use cached CSV files. The HuggingFace `datasets` loader is not used (it uses a legacy loading script that newer `datasets` versions reject).
- **Split files:** `empatheticdialogues/{train,valid,test}.csv`
- Groups rows by `conv_id`. Drops conversations with fewer than 2 valid turns. Skips rows with unknown emotion labels.
- **Returns:** One `LabeledSession` per conversation.

---

#### `load_mental_health_sessions() -> list[LabeledSession]`

- Loads `Amod/mental_health_counseling_conversations` (HuggingFace, `datasets.load_dataset`).
- 2-turn format: user=Context, assistant=Response.
- Assigns diffuse distress prior: `valence = Uniform(-0.65, -0.20)`, `arousal = Uniform(0.10, 0.55)`, seeded at 42.
- `emotion_label` is set to `"distressed"` (synthetic label for debugging only).

---

#### `pool_statistics(sessions) -> dict` / `print_pool_statistics(sessions)`

Returns/prints: `total_sessions`, `valence_mean`, `valence_stdev`, `arousal_mean`, `arousal_stdev`, `turns_mean`, `turns_min`, `turns_max`, `emotion_distribution`.

---

### 5.3 `data/construct_trajectories.py`

#### `@dataclass TrajectorySession`

| Field | Type |
|-------|------|
| `session_index` | `int` |
| `week_offset` | `int` |
| `conv_id` | `str` |
| `emotion_label` | `str` |
| `valence` | `float` |
| `arousal` | `float` |
| `turns` | `list[dict[str, str]]` |

---

#### `@dataclass Trajectory`

| Field | Type |
|-------|------|
| `trajectory_id` | `str` (e.g. `"traj_000042"`) |
| `arc_name` | `str` |
| `n_sessions` | `int` |
| `sessions` | `list[TrajectorySession]` |

---

#### `_sessions_for_zone(pool, zone, exclude, usage_counts, max_uses) -> list[LabeledSession]`

Filters the session pool to candidates that:
1. `zone.contains(s.valence, s.arousal)` — within tolerance.
2. `s.conv_id not in exclude` — not already used in this trajectory.
3. `usage_counts[s.conv_id] < max_uses` — not over the global reuse cap.

---

#### `_week_offsets(n_sessions, duration_weeks, rng) -> list[int]`

Distributes `n_sessions` across `total_weeks = rng.randint(*duration_weeks)` with jitter of ±0–2 weeks. Ensures strictly increasing offsets.

---

#### `construct_trajectories(pool, n_trajectories=5000, max_uses_per_conv=8, seed=42, fallback_tolerance_boost=0.15) -> list[Trajectory]`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pool` | (required) | `list[LabeledSession]` from `prepare.py` |
| `n_trajectories` | `5000` | Number to attempt |
| `max_uses_per_conv` | `8` | Soft cap on source conversation reuse |
| `seed` | `42` | RNG seed for reproducibility |
| `fallback_tolerance_boost` | `0.15` | Zone tolerance expansion if no candidates found |

**Algorithm:**
1. Build weighted template list (each template replicated by `max(1, int(weight * 10))`).
2. For each trajectory attempt: pick template, iterate zones, sample a compatible session per zone.
3. If a zone has no candidates: expand tolerance by `fallback_tolerance_boost` and retry once.
4. If still no candidates: skip trajectory, record failure.
5. Assign week offsets via `_week_offsets()`.
6. Trajectories with fewer than 2 sessions are discarded.

**Invariants:** No source conversation reused within a single trajectory. Same `conv_id` appears in at most `max_uses_per_conv` trajectories globally.

---

#### `save_trajectories(trajectories, output_path, indent=None) -> None`

Serializes to JSON. `indent=None` produces compact (smaller) files.

---

#### `load_trajectories(input_path) -> list[dict]`

Returns raw dicts (not `Trajectory` objects) — this is the format used by all training scripts.

---

#### `trajectory_statistics(trajectories) -> dict` / `print_trajectory_statistics(trajectories)`

Returns: `total_trajectories`, `arc_distribution`, `sessions_per_trajectory_{mean,min,max}`, `turns_per_session_mean`.

---

#### CLI

```
python -m data.construct_trajectories <N> [--out PATH] [--seed INT] [--max-uses INT] [--sample INT]
```

Default output: `data/trajectories_sample.json` for N≤500, `data/trajectories_{N//1000}k.json` otherwise.

---

## 6. Training — training/

### 6.1 `training/train.py`

#### Training objective

At each step `t` in a trajectory of length `N`, the model predicts what the next session will feel like:

```
z_0 = zeros
z_{t+1} = TransitionModel(z_t, e_t)
loss_t = 1 - cosine_similarity(z_{t+1}, normalize(e_{t+1}))
epoch_loss = mean(loss_t for t in 0..N-2)
```

Loss range: `[0, 2]` — 0 = perfect, 1 = orthogonal (chance), 2 = opposite.
Expected well-trained range: `0.6–0.8` (limited by synthetic trajectory noise ceiling).

---

#### `precompute_embeddings(trajectories, encoder, device, batch_size=256) -> dict[str, torch.Tensor]`

- Encodes all unique sessions (deduped by `conv_id`) in a single large batch.
- Returns `conv_id → (384,) float32 tensor` on `device`.
- Uses the same weighted-pool approach as `SessionEncoder.encode()`: ParlAI artifacts decoded, user turns weighted 2.0, assistant 1.0.
- Sessions shared across trajectories are encoded exactly once.

---

#### `trajectory_loss(model, session_embeddings: list[torch.Tensor]) -> torch.Tensor`

- Runs a single trajectory through the model step by step, accumulating cosine prediction loss.
- **Raises:** `ValueError` if trajectory has fewer than 2 sessions.
- Returns scalar tensor (mean loss over N-1 steps).

---

#### `evaluate(model, trajectories, emb_cache, device) -> float`

Mean trajectory loss over a trajectory set, no gradients. Sets model to `.eval()` mode.

---

#### `train(model, train_trajectories, val_trajectories, emb_cache, device, epochs=50, lr=1e-3, weight_decay=1e-4, clip_grad_norm=1.0, log_every_n_steps=10, checkpoint_path) -> dict`

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR, `T_max=epochs`, `eta_min=lr/100` |
| Gradient clipping | max norm 1.0 |
| Training order | Shuffled each epoch |
| Checkpoint criterion | Lowest validation loss |

**Side effects:**
- Writes checkpoint to `checkpoint_path` when best val loss is found.
- Writes training history JSON to `checkpoint_path.with_suffix(".history.json")`.

**Returns:** `{"train_loss": list[float], "val_loss": list[float]}` — one entry per epoch.

---

#### CLI

```
python -m training.train [--data PATH] [--checkpoint PATH] [--epochs INT] [--lr FLOAT] [--log-steps INT]
```

Defaults: `data/trajectories_sample.json`, `checkpoints/transition_v1.pt`, 50 epochs, lr=1e-3.

---

### 6.2 `training/train_probe.py`

#### `class ValenceArousalProbe(nn.Module)`

```
Linear(384, 2, bias=True)
forward: (..., 384) → (..., 2)   output[:, 0]=valence, output[:, 1]=arousal
```

Xavier uniform weight init, zero bias init.

---

#### `precompute_embeddings(trajectories, encoder, device, chunk_size=1500) -> dict[str, torch.Tensor]`

Identical approach to `training/train.py`. Processes in chunks of 1500 sessions to cap text list size. Logs progress every 5 chunks.

---

#### `build_probe_dataset(trajectories, emb_cache, transition_model, device) -> tuple[torch.Tensor, torch.Tensor]`

1. For each trajectory, runs all sessions through the transition model sequentially.
2. After each session, collects `(state_vector, valence, arousal)`.
3. Returns `states: (N_samples, 384)` and `targets: (N_samples, 2)`.

**Split is at trajectory level** — all samples from one trajectory appear in exactly one split. This prevents leakage (see Section 11).

---

#### `pearson_r(pred: np.ndarray, target: np.ndarray) -> float`

Returns `float("nan")` if either array has `std < 1e-9` (constant prediction).

---

#### `train_probe(probe, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=512, log_every=10, device) -> dict`

| Hyperparameter | Value |
|----------------|-------|
| Loss | MSE on both valence and arousal simultaneously |
| Optimizer | AdamW, `weight_decay=0.0` (linear probe is not regularized) |
| Scheduler | CosineAnnealingLR |
| Batch size | 512 |
| Epochs | 100 |

Returns history dict: `{train_mse, val_mse, val_r_valence, val_r_arousal}` — one entry per epoch.

---

#### `print_results(probe, X_val, y_val, history, n_train_samples, n_val_samples, device)`

Prints ASCII table with: architecture, sample counts, Pearson r, MSE, MAE, overall MSE, interpretation notes, ground-truth and prediction ranges.

---

#### `sanity_check(probe, all_trajectories, emb_cache, transition_model, device)`

Finds first `stable_positive` and `stable_negative` trajectories in the dataset. Runs both through the full pipeline. Checks `probe_valence(stable_positive) > probe_valence(stable_negative)`. Prints PASS/FAIL.

---

#### Probe checkpoint format

```python
{
    "model_state_dict":  OrderedDict,          # Linear(384→2) weights
    "probe_config":      {"state_dim": 384},
    "transition_ckpt":   str,                  # Source checkpoint filename
    "val_r_valence":     float,                # Final epoch Pearson r (valence)
    "val_r_arousal":     float,                # Final epoch Pearson r (arousal)
    "val_mse":           float,                # Final epoch val MSE
    "best_val_mse":      float,                # Best val MSE across all epochs
    "epochs_trained":    int,
    "history":           {"train_mse": [...], "val_mse": [...],
                          "val_r_valence": [...], "val_r_arousal": [...]},
}
```

#### CLI

```
python -m training.train_probe [--data PATH] [--checkpoint PATH] [--out-probe PATH]
                               [--epochs INT] [--lr FLOAT] [--batch-size INT]
                               [--val-split FLOAT] [--log-every INT] [--seed INT]
```

Defaults: `data/trajectories_10k.json`, `checkpoints/transition_v1_10k.pt`, `checkpoints/valence_arousal_probe.pt`.

---

### 6.3 `training/validate.py`

Produces Table 1 (core) and Table 1 (Extended) for the research paper.

#### `ARC_END_VALENCE: dict[str, float]`

Derived from `ARC_TEMPLATES`: maps arc name → end zone valence (last session's `CircumplexZone.valence`).

#### `ARC_POLARITY: dict[str, str]`

Maps arc name → `"net_positive"` (end valence > 0) or `"net_negative"` (≤ 0).

---

#### `build_embedding_cache(trajectories, encoder, device, chunk_size=1500) -> dict[str, torch.Tensor]`

Same chunked encoding pattern as other scripts.

---

#### `get_baseline_vectors(trajectories, emb_cache) -> tuple[list[np.ndarray], list[str]]`

**Baseline:** Last session's raw encoder embedding, L2-normalized.
Has access to identical text information as the model but no sequential integration.

---

#### `get_model_vectors(trajectories, emb_cache, model, device) -> tuple[list[np.ndarray], list[str]]`

Runs all sessions through the transition model. Returns the final state vector (already L2-normalized by the model).

---

#### `cosine_distance(a: np.ndarray, b: np.ndarray) -> float`

`1.0 - np.dot(a, b)` — valid for L2-normalized vectors; range [0, 2].

---

#### `class DistanceResult(NamedTuple)`

| Field | Type |
|-------|------|
| `within_arc_dist` | `float` |
| `between_arc_dist` | `float` |
| `separation_ratio` | `float` (`between / within`) |
| `within_per_arc` | `dict[str, float]` |
| `n_within_pairs` | `int` |
| `n_between_pairs` | `int` |

---

#### `compute_separation_metrics(vectors, labels, max_pairs_per_cell=2000, seed=42) -> DistanceResult`

For each (arc_i, arc_j) cell:
- Within-arc (diagonal): sample up to `max_pairs_per_cell` unique pairs.
- Between-arc (off-diagonal): sample up to `max_pairs_per_cell` pairs from cross-product.

Returns mean within-arc and between-arc cosine distance, their ratio, per-arc within distances.

---

#### `class PolarityResult(NamedTuple)`

| Field | Type | Meaning |
|-------|------|---------|
| `r_pb` | `float` | Point-biserial r (PC1 vs net_positive/net_negative) |
| `p_value` | `float` | p-value from `scipy.stats.pointbiserialr` |
| `ci_low` | `float` | 95% CI lower bound (Fisher z-transform) |
| `ci_high` | `float` | 95% CI upper bound |
| `n_positive` | `int` | Number of net_positive arc instances |
| `n_negative` | `int` | Number of net_negative arc instances |
| `pc1_var_exp` | `float` | % variance explained by PC1 |

---

#### `compute_polarity_correlation(vectors, labels) -> PolarityResult`

1. PCA(n_components=1) on filtered state vectors.
2. `scipy.stats.pointbiserialr(binary_polarity, pc1)`.
3. Sign convention: flip PC1 so `r_pb` is always positive (positive = PC1 correlates with net_positive).
4. Fisher-z 95% CI: `z ± 1.96 / sqrt(n-3)`, `tanh()` back.

---

#### Extended baselines and controls

**`get_identity_vectors(trajectories, state_dim=384) -> (list, list)`**

All-zeros vector for every trajectory. Null floor: separation ratio = 1.0, silhouette = undefined (all vectors identical).

**`get_ewma_vectors(trajectories, emb_cache, alpha=0.3) -> (list, list)`**

No learned parameters. At each step:
```
state_t = normalize(alpha * normalize(emb_t) + (1-alpha) * state_{t-1})
```
`alpha=0.3` = 70% persistence. Result is on the unit sphere.

**`compute_silhouette(vectors, labels) -> float`**

`sklearn.metrics.silhouette_score(mat, labels, metric="cosine")`. Returns `NaN` if all vectors identical or fewer than 2 labels.

**`compute_shuffled_separation(vectors, labels, max_pairs_per_cell=2000, seed=42) -> DistanceResult`**

Same state vectors, randomly permuted labels (seed offset +999 from main seed). Tests whether label identity is load-bearing for the observed separation.

**`class EffDimResult(NamedTuple)`**

| Field | Type |
|-------|------|
| `participation_ratio` | `float` — `(Σλ)² / Σ(λ²)`, range [1, 384] |
| `var_exp_per_pc` | `list[float]` — % variance per PC1..PC5 |
| `cumulative_var_exp` | `float` — sum of above |
| `n_components_fit` | `int` — actual PCA components fitted |

**`compute_effective_dimensionality(vectors, n_pcs=5) -> EffDimResult`**

PCA(n_components=min(N-1, D)), computes participation ratio from eigenvalues.

---

#### CLI

```
python -m training.validate [--data PATH] [--checkpoint PATH]
                            [--val-fraction FLOAT] [--max-pairs INT] [--seed INT]
```

Defaults: `data/trajectories_10k.json`, `checkpoints/transition_v1_10k.pt`, val-fraction=0.2.

---

### 6.4 `figures/visualize_step1.py`

Produces three figures from the data pipeline:

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | `fig1_circumplex_scatter.{png,pdf}` | All sessions in valence/arousal space, colored by emotion label |
| Fig 2 | `fig2_arc_trajectories.{png,pdf}` | 4 arc types as paths through circumplex |
| Fig 3 | `fig3_arc_distribution.{png,pdf}` | Trajectory counts per arc type |

**`apply_style()`** — sets rcParams for publication-quality output (300 dpi, tight bbox, Arial/Helvetica font, no top/right spines, subtle grid).

**`build_emotion_colors() -> dict[str, str]`** — Interpolated colors within quadrant families (warm orange = Q1, teal = Q2, red = Q3, indigo = Q4).

**`draw_circumplex_background(ax, alpha=0.07)`** — Shades 4 quadrants, draws dashed reference axes through origin.

**`fig1_circumplex_scatter(sessions)`** — Scatter with `JITTER=0.045` Gaussian noise, centroid markers, offset labels via path effects.

**`fig2_arc_trajectories(trajectories)`** — 2×2 grid of 4 arc types. Gradient arrows (plasma/viridis/inferno/cividis colormaps). Falls back to template zone centers if no instance found.

**`fig3_arc_distribution(trajectories)`** — Horizontal bar chart, colored by delta valence (improving/declining/stable).

---

### 6.5 `figures/visualize_step2.py`

| Figure | File | Content |
|--------|------|---------|
| Fig 4 | `fig4_tsne_embeddings.{png,pdf}` | t-SNE of 500 session embeddings, colored by quadrant |
| Fig 5 | `fig5_valence_correlation.{png,pdf}` | PC1 vs ground-truth valence + PCA variance explained |

**`load_balanced_sample(sessions, n_total=500, seed=42)`** — Equal representation across all 32 emotion labels (~15 per label). Tops up from remaining sessions if any label is under-represented.

**`encode_sessions(sessions, encoder) -> np.ndarray (n, 384)`** — Batch encoding via flattened turn list, same weighted-pool approach.

**`fig4_tsne_embeddings(sessions, embeddings)`** — t-SNE with `perplexity=30`, `max_iter=1000`, `init="pca"`, `metric="cosine"`. Points colored by specific emotion, quadrant markers (o/s/^/D).

**`fig5_valence_correlation(sessions, embeddings)`** — Left panel: PC1 vs valence scatter + linear fit + Pearson r annotation. Right panel: PCA variance explained by PC1–PC5, annotated with r-values. Best-correlated PC highlighted.

---

### 6.6 `figures/visualize_step3.py`

**Strict 6-phase execution order:**

1. Load ML models (encoder + transition model)
2. Load data (history JSONs + trajectories)
3. Encode sessions (chunked, `chunk_size=1500`)
4. Compute state trajectories + PCA
5. Compute arc separation (50 samples per arc)
6. Import matplotlib + draw figures

This order prevents sentence-transformer / matplotlib CUDA memory conflicts.

| Figure | File | Content |
|--------|------|---------|
| Fig 6 | `fig6_loss_curves.{png,pdf}` | Train/val loss curves for 300-traj and 10k-traj runs |
| Fig 7 | `fig7_state_trajectories.{png,pdf}` | 3 arc types as paths in 2-D PCA state space |
| Fig 8 | `fig8_arc_separation.{png,pdf}` | Final state vectors per arc type in PCA space |

**`encode_unique_sessions(trajs, chunk_size=1500) -> dict[str, np.ndarray]`** — Chunked encoding, returns numpy arrays (not tensors).

**`run_trajectory(traj) -> np.ndarray (n+1, 384)`** — Includes the initial zero state as row 0.

`SAMPLES_PER_ARC = 50` final states sampled per arc for Fig 8.

---

### 6.7 `figures/validate_msc.py`

Out-of-distribution validation on `nayohan/multi_session_chat`. Same strict phase ordering as `visualize_step3.py`.

**Constants:**

| Name | Value |
|------|-------|
| `CHECKPOINT` | `checkpoints/transition_v1_10k.pt` |
| `SYNTH_DATA` | `data/trajectories_10k.json` |
| `N_PLOT_SEQS` | `20` |
| `SEED` | `42` |

---

**`_extract_turns_from_row(row, cols) -> list[dict]`**

Multi-format parser with four fallback levels:

| Priority | Format | Condition |
|----------|--------|-----------|
| 0 (highest) | `dialogue` + `speaker` parallel lists | `"dialogue" in cols and "speaker" in cols` |
| A | List column (utterances/conversation/dialog/...) | Items are dicts or strings |
| B | Flat `speaker1`/`speaker2` columns | One utterance per row |
| C (lowest) | First non-metadata list column | Fallback |

Speaker mapping: `"1"` in speaker string → `"user"`, otherwise `"assistant"`.

---

**`parse_msc(dataset) -> dict[str, list[dict]]`**

Groups rows by `dialoug_id` (note: dataset typo), sorts sessions by `session_id`. Returns `{dialoug_id: [{session_id, turns}, ...]}`.

---

**`batch_encode(sessions: dict[str, list[dict]], chunk_size=1500) -> dict[str, torch.Tensor]`**

Same chunked encoding pattern. Session key format: `"{dlg_id}::{session_id}"`.

---

**Metrics computed:**

1. **Global pairwise cosine distance** — on a 2000-sample random subset of all state vectors. Non-collapse check (should be >> 0).
2. **Within-sequence consecutive cosine distance** — mean of `1 - dot(state_t, state_{t+1})` per dialogue. Compared between MSC (real) and synthetic trajectories.
3. **PCA** — 2 components on all MSC state vectors, used for path visualization.

**Outputs:** `fig_msc_validation.{png,pdf}` (two panels: PCA path plot + cosine distance histograms).

---

## 7. Configuration and Defaults

### Default paths

| Component | Default path |
|-----------|-------------|
| SQLite database | `~/.kokoro/kokoro.db` |
| ChromaDB memories | `~/.kokoro/memories/` |
| EmpatheticDialogues cache | `~/.cache/kokoro/empathetic_dialogues/` |
| TransitionModel checkpoint (default) | `<project_root>/checkpoints/transition_v1.pt` |
| TransitionModel checkpoint (10k) | `<project_root>/checkpoints/transition_v1_10k.pt` |
| Linear probe checkpoint | `<project_root>/checkpoints/valence_arousal_probe.pt` |

`<project_root>` is `Path(__file__).parent.parent` relative to any file in `kokoro/`.

---

### Thresholds and magic numbers

| Name | Value | Location | Meaning |
|------|-------|----------|---------|
| `_MIN_SESSIONS` | `3` | `decoder.py` | Minimum sessions before summary |
| `_MIN_TREND_ENTRIES` | `3` | `decoder.py` | Minimum history entries for slope |
| `_VALENCE_POS` | `+0.30` | `decoder.py` | Valence threshold for "positive" |
| `_VALENCE_NEG` | `-0.30` | `decoder.py` | Valence threshold for "negative" |
| `_AROUSAL_HIGH` | `+0.30` | `decoder.py` | Arousal threshold for "activated" |
| `_AROUSAL_LOW` | `-0.10` | `decoder.py` | Arousal threshold for "low-energy" |
| `_SLOPE_IMP` | `+0.02` | `decoder.py` | Slope threshold for "improving" |
| `_SLOPE_DEC` | `-0.02` | `decoder.py` | Slope threshold for "declining" |
| Probe clamp | `[-1.5, 1.5]` | `decoder.py` | Prevents extreme probe outputs |
| `_ARC_HISTORY_MAX` | `20` | `store.py` | Max arc_history entries per user |
| `_STATE_DTYPE` | `float32` | `store.py` | SQLite blob dtype |
| `_EMBED_DIM` | `384` | `retrieval.py` | Required embedding dimension |
| `CircumplexZone.tolerance` | `0.25` | `arc_templates.py` | Default zone acceptance window |
| `fallback_tolerance_boost` | `0.15` | `construct_trajectories.py` | Zone expansion on miss |
| `max_uses_per_conv` | `8` | `construct_trajectories.py` | Default reuse cap |
| EWMA alpha | `0.3` | `validate.py` | Persistence in EWMA baseline |
| `max_pairs_per_cell` | `2000` | `validate.py` | Pair sampling for separation metrics |
| `SAMPLES_PER_ARC` | `50` | `visualize_step3.py` | For arc separation PCA plot |
| `N_PLOT_SEQS` | `20` | `validate_msc.py` | Sequences shown in path plot |
| t-SNE perplexity | `30` | `visualize_step2.py` | |
| t-SNE iterations | `1000` | `visualize_step2.py` | |
| Balanced sample N | `500` | `visualize_step2.py` | Sessions for step-2 figures |
| Jitter σ | `0.045` | `visualize_step1.py` | Scatter jitter for circumplex plot |

---

### Training hyperparameters (defaults)

| Script | Hyperparameter | Default |
|--------|---------------|---------|
| `train.py` | epochs | 50 |
| `train.py` | lr | 1e-3 |
| `train.py` | weight_decay | 1e-4 |
| `train.py` | clip_grad_norm | 1.0 |
| `train.py` | LR schedule | CosineAnnealingLR, eta_min = lr/100 |
| `train_probe.py` | epochs | 100 |
| `train_probe.py` | lr | 1e-3 |
| `train_probe.py` | batch_size | 512 |
| `train_probe.py` | weight_decay | 0.0 |
| `train_probe.py` | val_split | 0.2 (trajectory-level) |

---

### Checkpoints shipped with the repo

| File | Trained on | Epochs |
|------|-----------|--------|
| `checkpoints/transition_v1.pt` | 300 trajectories | 50 |
| `checkpoints/transition_v1_10k.pt` | 10,000 trajectories | 100 |
| `checkpoints/valence_arousal_probe.pt` | Built from `transition_v1_10k.pt` | 100 |

`WorldMemory` defaults to `transition_v1.pt`. For best performance, pass `checkpoint_path="checkpoints/transition_v1_10k.pt"`.

---

## 8. Data Formats

### 8.1 `trajectories_10k.json` — example

```json
[
  {
    "trajectory_id": "traj_000042",
    "arc_name": "slow_recovery",
    "n_sessions": 9,
    "sessions": [
      {
        "session_index": 0,
        "week_offset": 0,
        "conv_id": "hit:5_conv:2",
        "emotion_label": "devastated",
        "valence": -0.80,
        "arousal": -0.30,
        "turns": [
          {"role": "user",      "content": "I just found out my dad has cancer."},
          {"role": "assistant", "content": "I'm so sorry to hear that..."},
          {"role": "user",      "content": "It feels completely unreal."}
        ]
      },
      {
        "session_index": 1,
        "week_offset": 2,
        "conv_id": "hit:12_conv:7",
        "emotion_label": "sad",
        "valence": -0.55,
        "arousal": -0.25,
        "turns": [...]
      }
    ]
  }
]
```

---

### 8.2 SQLite schema — `kokoro.db`

```sql
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS user_states (
    user_id       TEXT    PRIMARY KEY,
    state_vector  BLOB    NOT NULL,     -- float32 little-endian bytes
    state_dim     INTEGER NOT NULL,     -- 384 for default model
    session_count INTEGER NOT NULL DEFAULT 0,
    valence       REAL    NOT NULL DEFAULT 0.0,
    arousal       REAL    NOT NULL DEFAULT 0.0,
    arc_history   TEXT    NOT NULL DEFAULT '[]',  -- JSON: [[v,a], ...]
    created_at    TEXT    NOT NULL,     -- ISO-8601 UTC, e.g. "2025-01-15T10:23:45.123456Z"
    updated_at    TEXT    NOT NULL
);
```

`state_vector` reconstruction: `np.frombuffer(blob, dtype="float32").copy()`.

`arc_history` example: `"[[-0.6, 0.3], [-0.4, 0.1], [-0.1, 0.0]]"` — oldest first, max 20 entries.

---

### 8.3 ChromaDB collection schema

**Collection name:** `"kokoro"` (default) or user-configured.
**Distance metric:** `"hnsw:space": "cosine"`.

| Field | Type | Description |
|-------|------|-------------|
| id | `str` | `sha1(user_id + "\x00" + session_id)` |
| embedding | `list[float]` (384) | Session embedding from `SessionEncoder` |
| document | `str` | Session text summary (first 200 chars of user turns) |
| metadata.user_id | `str` | |
| metadata.session_id | `str` | e.g. `"session_3"` |
| metadata.valence | `float` | From linear probe |
| metadata.arousal | `float` | From linear probe |
| metadata.timestamp | `float` | Unix timestamp (`time.time()`) |

---

### 8.4 Checkpoint file formats

**TransitionModel checkpoint (`transition_v1.pt`, `transition_v1_10k.pt`):**

```python
{
    "epoch":                int,                       # epoch of best val loss
    "model_state_dict":     OrderedDict,               # TransitionModel weights
    "optimizer_state_dict": OrderedDict,               # AdamW state (for resuming)
    "val_loss":             float,                     # best val loss
    "train_loss":           float,                     # train loss at best epoch
    "model_config":         {"state_dim": 384, "hidden_dim": 512},
}
```

**Training history JSON (`transition_v1.history.json`):**

```json
{
    "train_loss": [0.8234, 0.7891, ...],
    "val_loss":   [0.8412, 0.7923, ...],
    "best_epoch": 47,
    "best_val_loss": 0.7156,
    "n_train": 240,
    "n_val": 60,
    "epochs": 50,
    "lr": 0.001
}
```

**Linear probe checkpoint (`valence_arousal_probe.pt`):**

```python
{
    "model_state_dict":  OrderedDict,      # Linear(384→2) weight + bias
    "probe_config":      {"state_dim": 384},
    "transition_ckpt":   str,              # Source checkpoint filename
    "val_r_valence":     float,            # Final Pearson r (valence)
    "val_r_arousal":     float,            # Final Pearson r (arousal)
    "val_mse":           float,            # Final val MSE
    "best_val_mse":      float,
    "epochs_trained":    int,
    "history": {
        "train_mse":     [...],
        "val_mse":       [...],
        "val_r_valence": [...],
        "val_r_arousal": [...],
    },
}
```

---

### 8.5 `WorldMemory.get_context()` output dict

```python
{
    "state_summary":     str | None,
    # Natural-language emotional context for LLM system prompt.
    # None when session_count < min_sessions (warm-up period).

    "relevant_memories": list[str],
    # Session text summaries retrieved by hybrid scoring.
    # Empty list when not ready or no stored sessions.

    "valence":           float,
    # Current valence from linear probe, clamped to [-1.5, 1.5],
    # rounded to 4 decimal places.

    "arousal":           float,
    # Current arousal from linear probe, clamped to [-1.5, 1.5],
    # rounded to 4 decimal places.

    "trend":             str,
    # "improving" | "declining" | "stable"
    # Derived from linear slope of arc_history valences.

    "session_count":     int,
    # Total sessions stored for this user.

    "ready":             bool,
    # False when session_count < min_sessions (default 3).
}
```

---

## 9. Dependencies

### Core runtime dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| `sentence-transformers` | `>=2.7.0` | `all-MiniLM-L6-v2` session encoder |
| `torch` | `>=2.2.0` | TransitionModel, linear probe, tensor ops |
| `numpy` | `>=1.26.0` | Embedding arithmetic, blob serialization |
| `datasets` | `>=2.19.0` | `load_dataset` for supplementary data |
| `chromadb` | `>=0.5.0` | Persistent vector store (MemoryStore) |
| `tqdm` | `>=4.66.0` | Progress bars during encoding |

### Optional training dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| `scikit-learn` | `>=1.4.0` | PCA, t-SNE, silhouette score |
| `scipy` | (unlisted) | `pointbiserialr`, `pearsonr` in validation |
| `matplotlib` | `>=3.8.0` | All figure generation |
| `transformers` | `>=4.40.0` | Tokenizer utilities (optional) |
| `accelerate` | `>=0.30.0` | Multi-GPU training (optional) |

### Optional dev dependencies

| Package | Min version |
|---------|-------------|
| `pytest` | `>=8.0` |
| `pytest-cov` | `>=5.0` |
| `black` | `>=24.0` |
| `ruff` | `>=0.4.0` |

### Standard library only (no extra deps)

`sqlite3`, `json`, `hashlib`, `csv`, `tarfile`, `urllib.request`, `pathlib`, `dataclasses`, `collections`, `time`, `logging`, `random`, `math`

---

### CPU vs GPU

| Component | Device |
|-----------|--------|
| `SessionEncoder` | Auto-detects CUDA; CPU-capable. The `all-MiniLM-L6-v2` model is small (22 MB) and CPU inference is ~100–200 ms/session. |
| `TransitionModel` | CPU in inference (all `WorldMemory` paths use `map_location="cpu"`). GPU used during training if available. |
| `ValenceArousalProbe` | CPU only (tiny linear layer). |
| `StateStore` (SQLite) | CPU / disk. |
| `MemoryStore` (ChromaDB) | CPU / disk. |
| `visualize_step3.py` | Forces `device = torch.device("cpu")` to avoid CUDA/matplotlib conflict. |

---

### Minimum hardware

- RAM: 2 GB for inference with cached encoder; 4 GB for encoding new sessions (sentence transformer needs ~500 MB).
- Storage: ~100 MB for checkpoints + `all-MiniLM-L6-v2` weights (~90 MB cached by HuggingFace).
- For training: 8 GB RAM recommended; GPU optional but 3–5× faster.

---

## 10. Call Graph

### 10.1 `WorldMemory.update(session_turns)`

```
WorldMemory.update()                        [memory.py]
├── SessionEncoder.encode(session_turns)    [encoder.py]
│   ├── encoder._prepare_turns(turns)
│   │   └── decode_parlai_artifacts(content)  [encoder.py]
│   └── encoder.model.encode(texts, ...)    [sentence-transformers]
├── StateStore.load(user_id)               [store.py]
│   └── store._connect() → sqlite3
├── TransitionModel.initial_state()         [transition.py]  (if no stored state)
├── torch.from_numpy(state)                 [torch]
├── TransitionModel.forward(state_t, emb_t) [transition.py]
│   ├── torch.cat([state, emb], dim=-1)
│   ├── self.net(x)  [Linear→LayerNorm→ReLU→Dropout→Linear]
│   └── F.normalize(x, dim=-1)
├── StateStore.get_info(user_id)            [store.py]
│   └── store._connect() → sqlite3
├── StateDecoder.decode(state, arc, count)  [decoder.py]
│   ├── decoder._run_probe(state)
│   │   ├── torch.from_numpy(state)
│   │   ├── probe.forward(x)  [Linear(384→2)]
│   │   └── np.clip(pred, -1.5, 1.5)
│   ├── _linear_slope(valences)            [decoder.py]
│   ├── _classify_trend(slope)             [decoder.py]
│   ├── _describe_valence(v)               [decoder.py]
│   ├── _describe_arousal(a)               [decoder.py]
│   └── _build_summary(...)                [decoder.py]
├── StateStore.save(user_id, state, v, a)  [store.py]
│   └── store._connect() → sqlite3 (UPSERT)
├── WorldMemory._session_summary(turns)    [memory.py]
├── MemoryStore.add_session(...)           [retrieval.py]
│   ├── _validate_embedding(emb, ...)
│   ├── _doc_id(user_id, session_id)
│   └── col.upsert(...)                    [chromadb]
└── WorldMemory.get_context("")            [memory.py]
    └── (see 10.2)
```

### 10.2 `WorldMemory.get_context(current_message)`

```
WorldMemory.get_context()                   [memory.py]
├── WorldMemory._get_current_state()        [memory.py]
│   └── StateStore.load(user_id)           [store.py]
├── StateStore.get_info(user_id)            [store.py]
│   └── store._connect() → sqlite3
├── StateDecoder.decode(...)                [decoder.py]
│   └── (same as above, abbreviated)
├── [if ready and message] SessionEncoder.encode_text(message)  [encoder.py]
│   └── encoder.model.encode([text], ...)  [sentence-transformers]
└── [if ready] MemoryStore.retrieve(...)   [retrieval.py]
    ├── col.get(where={"user_id": ...})    [chromadb]
    ├── np.array(embeddings, dtype=float32)
    ├── _cosine_sim_batch(query_emb, stored)  [retrieval.py]
    │   ├── _l2_normalize(query)
    │   └── stored / norms  @  query
    ├── _cosine_sim_batch(state_vec, stored)
    └── alpha * semantic + (1-alpha) * emotional
```

### 10.3 TransitionModel training loop (`training/train.py`)

```
__main__
├── json.load(trajectories)
├── random.shuffle(train_trajs)
├── SessionEncoder()                            [encoder.py]
├── precompute_embeddings(all_trajs, encoder)  [train.py]
│   ├── decode_parlai_artifacts(content)        [encoder.py]
│   └── encoder.model.encode(all_texts, ...)   [sentence-transformers]
├── TransitionModel()                           [transition.py]
└── train(model, train_trajs, val_trajs, ...)   [train.py]
    ├── torch.optim.AdamW(model.parameters(), lr, weight_decay)
    ├── CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    └── [each epoch]
        ├── random.shuffle(train_trajectories)
        ├── [each trajectory]
        │   ├── trajectory_loss(model, embs)    [train.py]
        │   │   ├── TransitionModel.initial_state()
        │   │   ├── [each step t]
        │   │   │   ├── model.forward(state, emb_t)
        │   │   │   ├── F.normalize(emb_{t+1}, dim=-1)
        │   │   │   └── 1 - (state * target).sum()
        │   │   └── mean(losses)
        │   ├── optimizer.zero_grad()
        │   ├── loss.backward()
        │   ├── clip_grad_norm_(model.parameters(), 1.0)
        │   └── optimizer.step()
        ├── scheduler.step()
        ├── evaluate(model, val_trajs, ...)    [train.py]
        └── [if best val] torch.save(checkpoint)
```

### 10.4 Probe training (`training/train_probe.py`)

```
__main__
├── torch.load(transition_checkpoint)
├── TransitionModel(**cfg).load_state_dict(...)
├── SessionEncoder()
├── precompute_embeddings(all_trajs, encoder)  [train_probe.py]
├── build_probe_dataset(train_trajs, emb_cache, model)  [train_probe.py]
│   └── [each trajectory]
│       ├── TransitionModel.initial_state()
│       └── [each session]
│           ├── model.forward(state, emb)
│           └── collect (state.numpy(), valence, arousal)
├── ValenceArousalProbe(state_dim=384)
└── train_probe(probe, X_train, y_train, X_val, y_val, ...)  [train_probe.py]
    ├── AdamW(probe.parameters(), lr=1e-3, weight_decay=0)
    ├── CosineAnnealingLR(optimizer, T_max=epochs)
    └── [each epoch]
        ├── [each mini-batch]
        │   ├── probe.forward(xb)
        │   ├── F.mse_loss(pred, yb)
        │   └── loss.backward() + optimizer.step()
        ├── scheduler.step()
        └── probe.forward(X_val) → pearson_r, val_mse
```

---

## 11. Known Limitations

### 11.1 Dimensional collapse (participation ratio 1.4/384)

The model's final state space is highly collapsed: participation ratio PR = 1.4 out of a possible 384, with PC1 explaining 82.6% of variance. This means the state space encodes emotional information along fewer than ~2 effective dimensions despite the 384-dimensional ambient space. The model compresses 11-arc emotional structure into a very low-dimensional manifold. Whether this is beneficial (efficient compression) or a limitation (loss of expressive capacity for fine-grained distinctions) depends on the downstream task. PRs below 10 indicate near-one-dimensional structure.

### 11.2 Cold start warm-up period

`get_context()` returns `ready=False`, `state_summary=None`, and no retrieved memories for the first `min_sessions-1` sessions (default: first 2 sessions). During this period the system provides no emotional context to the LLM. This is intentional (insufficient trajectory to form reliable trends) but may be frustrating for short-engagement users.

### 11.3 Synthetic training data limitations

The transition model is trained entirely on synthetically constructed trajectories from EmpatheticDialogues sessions arranged to match arc templates. Real users' emotional trajectories may not follow the 11 arc types. Specifically:
- Arc sessions are assigned by circumplex coordinate matching, not by actual emotional arc content within the session text.
- No real longitudinal user data was available during training.
- OOD validation on MSC shows 1.20× ratio (MSC within-seq distance / synthetic within-seq distance), indicating mild distributional shift but no catastrophic failure.

### 11.4 Session-level leakage in train/val split

The train/val split for both transition model training and probe training is at the **trajectory level** (all sessions from one trajectory go to one split). However, because the same EmpatheticDialogues source conversation (`conv_id`) can appear in up to 8 trajectories (default `max_uses_per_conv=8`), a conversation may appear in both training and validation trajectories. This constitutes session-level leakage: the model may have seen the turn text for a "validation" session during training under a different arc context. The trajectory-level split prevents the same arc from leaking, but not the same raw text.

### 11.5 MSC cold-start directional bias

The OOD validation runs the transition model starting from the zero cold-start vector on each MSC dialogue independently (no persistent state across dialogues). This means the first session in each MSC dialogue produces a state that's strongly influenced by the cold-start, regardless of prior context. For real deployment with persistent state, the cold-start effect only occurs once per user — but the OOD validation restarts it for every dialogue, potentially underestimating the model's true performance on continuing conversations.

### 11.6 Cosine objective without negatives

The training objective (`1 - cosine_similarity(z_{t+1}, e_{t+1})`) only provides positive signal: the state should point toward the next session. There is no contrastive negative signal: the state is not penalized for also pointing toward emotionally distant arcs. This may contribute to the dimensional collapse (the model learns to align with positive examples but the resulting directions may cluster). Adding hard negatives (state should be far from sessions of a different arc type) is an extension point.

### 11.7 No time-delta conditioning

The transition model does not know how many days elapsed between sessions. A user returning after 6 months should produce a different state update than one returning after 1 day. The `week_offset` field is computed and stored in trajectories but is not fed to the model. This is noted as an extension point.

---

## 12. Extension Points

### 12.1 Swap the encoder for a different model

**Where to change:** `kokoro/encoder.py`, `SessionEncoder.__init__`.

```python
class SessionEncoder:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # change this
    EMBEDDING_DIM = 384                                       # and this
```

All downstream code uses `EMBEDDING_DIM`. The TransitionModel's `state_dim` and the probe's `state_dim` must also be updated to match the new dimension, and both checkpoints must be retrained. The `_EMBED_DIM` constant in `retrieval.py` must also be updated.

### 12.2 Swap the MLP for an LSTM

The upgrade path is already defined in `kokoro/transition.py` as a commented-out `TransitionModelLSTM` class with identical `STATE_DIM = 384` and the same `forward` signature pattern.

The difference: LSTM `forward` returns `(new_state, hidden)` and takes an optional `hidden` argument. The `WorldMemory` class would need to store and pass the hidden state, which would require adding a hidden-state field to `StateStore` (stored as an additional blob column). The public `update()` and `get_context()` signatures would not change.

To activate:
1. Uncomment `TransitionModel = TransitionModelLSTM` in `transition.py`.
2. Add `hidden_state BLOB` column to `user_states` table in `store.py`.
3. Add `save_hidden()` / `load_hidden()` methods to `StateStore`.
4. Thread hidden state through `WorldMemory._get_current_state()` and `update()`.

### 12.3 Add time_delta as a third input

**Where to change:** `kokoro/transition.py`.

Current input is `cat([state (384), emb (384)]) = (768,)`. With time encoding:

```python
# In TransitionModel.__init__:
self.net = nn.Sequential(
    nn.Linear(state_dim * 2 + time_features, hidden_dim),  # 768+k → 512
    ...
)

# In forward:
def forward(self, state, session_emb, days_since_last: float = 0.0):
    time_enc = encode_time_delta(days_since_last)  # e.g. log(1+days) / 10
    x = torch.cat([state, session_emb, time_enc], dim=-1)
    ...
```

The `week_offset` field is already present in trajectory JSON. `WorldMemory.update()` would need a `days_since_last` parameter. `StateStore.get_info()` already returns `updated_at` which can be used to compute elapsed days.

### 12.4 Add contrastive training objective

**Where to change:** `training/train.py`, `trajectory_loss()`.

Current loss: `1 - cos_sim(z_{t+1}, normalize(e_{t+1}))`.

Proposed augmentation — add a negative term:

```python
# For each step t, sample a random session from a DIFFERENT arc type
negative_emb = sample_from_different_arc(traj, emb_cache, rng)
neg_sim = (state * F.normalize(negative_emb, dim=-1)).sum()
loss_t = (1 - pos_sim) + max(0, neg_sim - margin)   # margin triplet loss
```

This would require passing arc-type information alongside the embedding cache, or grouping trajectories by arc before the training loop.

### 12.5 Add real longitudinal data to training

**Where to change:** `data/prepare.py` (add new loader), `data/construct_trajectories.py` (or bypass arc templates entirely), `training/train.py` (real trajectories need no synthetic assembly).

Real longitudinal data requires:
- Multi-session conversations grouped by user ID.
- Ground-truth emotional labels per session (or dense annotation via a pre-trained affect detector).
- A train/val split that prevents the same user appearing in both splits (user-level leakage is more serious than session-level).

The simplest path: define a new `load_real_trajectories(source_path) -> list[dict]` function in `data/prepare.py` that returns the same JSON structure as `trajectories_10k.json`. All training scripts accept this format with no other changes.
