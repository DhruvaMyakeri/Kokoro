"""
experiments/build_contexts.py

Stage 1 of the two-process evaluation pipeline.

Runs entirely in the kokoro/PyTorch process — no Groq, no httpx, no SSL.
For every scenario (or a subset with --test), it:
  - Creates a WorldMemory instance in temp dirs
  - Feeds all sessions via update()
  - Retrieves context A (alpha=1.0, semantic-only)
  - Retrieves context B (alpha=0.6, hybrid + state summary)
  - Cleans up

Output: experiments/contexts.json

Usage:
    python experiments/build_contexts.py           # all 20 scenarios
    python experiments/build_contexts.py --test    # s01, s02, s03 only
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Patch ChromaDB before WorldMemory imports it.
# ---------------------------------------------------------------------------
import chromadb as _chromadb
_chromadb.PersistentClient = lambda path, **kw: _chromadb.EphemeralClient()

from experiments.scenarios import SCENARIOS
from experiments.scenarios_long import SCENARIOS_LONG
from kokoro import WorldMemory

CONTEXTS_PATH      = Path(__file__).parent / "contexts.json"
CONTEXTS_LONG_PATH = Path(__file__).parent / "contexts_long.json"


# ---------------------------------------------------------------------------
# In-memory session store — replaces ChromaDB during context building
# ---------------------------------------------------------------------------

class _InMemoryStore:
    """
    Pure-numpy session store: no native libs, no file I/O.

    Emotional axis uses VAD-coordinate L2 distance rather than state-to-state
    cosine similarity.  Cosine similarity in the transition model's state space
    is ineffective because all state vectors lie in a tight angular cone
    (~0.98–1.00 similarity regardless of emotional phase).  The decoded
    (valence, arousal) coordinates are far more discriminative: a burned-out
    session (v=-0.76) is clearly distant from a recovered session (v=+0.25)
    in 2-D VAD space.

    If decoder is provided, emotional_score = 1 - L2(va_current, va_stored) / max_dist.
    Otherwise falls back to state-to-state cosine (legacy behaviour).
    """

    def __init__(self, decoder=None) -> None:
        self._sessions: list[dict] = []
        self._decoder = decoder  # kokoro.decoder.StateDecoder instance, or None

    class _NoOpClient:
        def reset(self) -> None:
            pass
    _client = _NoOpClient()

    def add_session(
        self,
        user_id: str,
        session_id: str,
        session_text: str,
        session_embedding: np.ndarray,
        state_vector: np.ndarray,
        valence: float,
        arousal: float,
    ) -> None:
        self._sessions.append({
            "user_id":      user_id,
            "session_id":   session_id,
            "session_text": session_text,
            "embedding":    session_embedding.astype(np.float32).copy(),
            "state_vector": state_vector.astype(np.float32).copy(),
            "valence":      float(valence),
            "arousal":      float(arousal),
        })

    def retrieve(
        self,
        user_id: str,
        query_embedding: np.ndarray | None,
        state_vector: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.6,
        current_valence: float | None = None,
        current_arousal: float | None = None,
        recency_weight: float = 0.0,
        recency_tau: float = 5.0,
        adaptive: bool = False,
    ) -> list[dict]:
        user_sess = [s for s in self._sessions if s["user_id"] == user_id]
        if not user_sess:
            return []

        stored_embs   = np.array([s["embedding"]    for s in user_sess], dtype=np.float32)
        stored_states = np.array([s["state_vector"] for s in user_sess], dtype=np.float32)
        n = len(user_sess)

        def _cosine(q: np.ndarray, S: np.ndarray) -> np.ndarray:
            q = q.astype(np.float32)
            q_norm = q / (np.linalg.norm(q) + 1e-12)
            norms = np.linalg.norm(S, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            return (S / norms) @ q_norm

        # Semantic: query message embedding vs stored session embeddings (MiniLM),
        # rescaled from [-1, 1] to [0, 1] to be commensurable with the emotional
        # axis (mirrors kokoro/retrieval.py). None query → axis disabled.
        if query_embedding is not None:
            sem = (_cosine(query_embedding, stored_embs) + 1.0) / 2.0
        else:
            sem = np.zeros(n, dtype=np.float64)

        # Emotional axis: VAD-coordinate proximity is far more discriminative than
        # state-to-state cosine (which concentrates at ~0.98 regardless of phase).
        # Priority: decoder (most accurate) > passed-in va > cosine fallback.
        if self._decoder is not None:
            dec    = self._decoder.decode(state_vector, [], n)
            v, a   = dec["valence"], dec["arousal"]
        elif current_valence is not None and current_arousal is not None:
            v, a   = current_valence, current_arousal
        else:
            v, a   = None, None

        va_stored = np.array([[s["valence"], s["arousal"]] for s in user_sess],
                             dtype=np.float32)
        if v is not None:
            va_cur = np.array([v, a], dtype=np.float32)
            dists = np.linalg.norm(va_stored - va_cur, axis=1)
            emo   = 1.0 - dists / float(np.sqrt(8))
        else:
            emo = _cosine(state_vector, stored_states)

        # Recency axis: insertion order (newest = rank_age 0), mirrors production
        recency = np.exp(-np.arange(n - 1, -1, -1, dtype=np.float64) / max(recency_tau, 1e-6))

        effective_alpha = alpha
        if adaptive:
            from kokoro.retrieval import adaptive_alpha
            effective_alpha = adaptive_alpha(alpha, va_stored, (v, a))
        if query_embedding is None:
            effective_alpha = 0.0

        hybrid   = effective_alpha * sem + (1.0 - effective_alpha) * emo
        combined = (1.0 - recency_weight) * hybrid + recency_weight * recency

        k       = min(top_k, n)
        top_idx = np.argsort(-combined)[:k]

        return [
            {
                "session_text":    user_sess[i]["session_text"],
                "session_id":      user_sess[i]["session_id"],
                "semantic_score":  float(sem[i]),
                "emotional_score": float(emo[i]),
                "recency_score":   float(recency[i]),
                "combined_score":  float(combined[i]),
                "effective_alpha": float(effective_alpha),
                "valence":         user_sess[i]["valence"],
                "arousal":         user_sess[i]["arousal"],
            }
            for i in top_idx
        ]

    def delete_user(self, user_id: str) -> int:
        before = len(self._sessions)
        self._sessions = [s for s in self._sessions if s["user_id"] != user_id]
        return before - len(self._sessions)


# ---------------------------------------------------------------------------
# Per-scenario context builder
# ---------------------------------------------------------------------------

def build_context(scenario: dict) -> dict:
    sid         = scenario["scenario_id"]
    sessions    = scenario["sessions"]
    new_message = scenario["new_message"]

    db_tmp  = tempfile.TemporaryDirectory()
    mem_tmp = tempfile.TemporaryDirectory()

    try:
        memory = WorldMemory(
            user_id      = f"eval_{sid}",
            db_path      = Path(db_tmp.name) / "kokoro.db",
            persist_dir  = mem_tmp.name,
            min_sessions = 3,
            alpha        = 0.6,
            top_k        = 3,
        )
        # Pass the decoder so the emotional axis uses VAD-coordinate distance
        # rather than state-to-state cosine (which is too uniform for discrimination).
        memory._mem_store = _InMemoryStore(decoder=memory._decoder)

        for session in sessions:
            memory.update(session)

        ctx_a = memory.get_context(new_message, alpha=1.0)
        ctx_b = memory.get_context(new_message, alpha=0.6)

        # Condition C: recency-only baseline — the last top_k session summaries,
        # no state summary. This is the obvious cheap competitor to emotional
        # retrieval (a recent memory is usually emotionally current); without
        # beating it, the "world model" mechanism is not demonstrably needed.
        recency_memories = [
            s["session_text"]
            for s in memory._mem_store._sessions[-3:]
        ][::-1]  # newest first, matching retrieval-order presentation

        # Compute memory-set overlap to show retrieval divergence
        mems_a = set(ctx_a["relevant_memories"])
        mems_b = set(ctx_b["relevant_memories"])
        overlap = len(mems_a & mems_b)
        total   = max(len(mems_a), len(mems_b), 1)
        overlap_pct = round(overlap / total * 100, 1)

        return {
            "scenario_id":        scenario["scenario_id"],
            "arc_type":           scenario["arc_type"],
            "description":        scenario["description"],
            "new_message":        new_message,
            "expected_awareness": scenario.get("expected_awareness", ""),
            "context_a": {
                "relevant_memories": ctx_a["relevant_memories"],
                "state_summary":     ctx_a.get("state_summary"),
                "ready":             ctx_a.get("ready", False),
            },
            "context_b": {
                "relevant_memories": ctx_b["relevant_memories"],
                "state_summary":     ctx_b.get("state_summary"),
                "ready":             ctx_b.get("ready", False),
                "valence":           ctx_b.get("valence", 0.0),
                "arousal":           ctx_b.get("arousal", 0.0),
                "trend":             ctx_b.get("trend", "unknown"),
            },
            "context_c": {
                "relevant_memories": recency_memories,
                "state_summary":     None,
                "ready":             ctx_a.get("ready", False),
            },
            "retrieval_diff": {
                "overlap_pct":    overlap_pct,
                "memories_differ": overlap_pct < 100.0,
                "only_in_b":      sorted(mems_b - mems_a),
                "only_in_a":      sorted(mems_a - mems_b),
            },
        }

    finally:
        try:
            memory._mem_store._client.reset()
        except Exception:
            pass
        try:
            db_tmp.cleanup()
        except Exception:
            pass
        try:
            mem_tmp.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    long_mode = "--long" in sys.argv
    test_mode = "--test" in sys.argv and not long_mode

    if long_mode:
        scenarios     = SCENARIOS_LONG
        output_path   = CONTEXTS_LONG_PATH
        label         = f"all {len(scenarios)} long-history scenarios (--long)"
    elif test_mode:
        scenarios     = SCENARIOS[:3]
        output_path   = CONTEXTS_PATH
        label         = "first 3 (--test mode)"
    else:
        scenarios     = SCENARIOS
        output_path   = CONTEXTS_PATH
        label         = f"all {len(scenarios)} scenarios"

    print(f"\n{'='*60}")
    print(f"  build_contexts.py — {label}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    results = []
    for i, scenario in enumerate(scenarios, 1):
        sid      = scenario["scenario_id"]
        arc_type = scenario["arc_type"]
        n_sess   = len(scenario["sessions"])
        print(f"  {sid} [{arc_type}] ({i}/{len(scenarios)}, {n_sess} sessions) ... ",
              end="", flush=True)
        try:
            ctx = build_context(scenario)
            results.append(ctx)
            mem_a    = len(ctx["context_a"]["relevant_memories"])
            mem_b    = len(ctx["context_b"]["relevant_memories"])
            overlap  = ctx["retrieval_diff"]["overlap_pct"]
            diff_str = "SAME" if overlap == 100.0 else f"{100-overlap:.0f}% different"
            trend    = ctx["context_b"]["trend"]
            v        = ctx["context_b"]["valence"]
            a        = ctx["context_b"]["arousal"]
            print(f"done  [A:{mem_a} B:{mem_b} mems | {diff_str} | v={v:+.2f} a={a:+.2f} {trend}]")
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(results)} contexts to {output_path}")

    # Retrieval divergence summary
    if results:
        differ_count = sum(1 for r in results if r["retrieval_diff"]["memories_differ"])
        avg_overlap  = sum(r["retrieval_diff"]["overlap_pct"] for r in results) / len(results)
        print(f"\n  Retrieval divergence (A vs B):")
        print(f"    Scenarios where memories differ: {differ_count}/{len(results)}")
        print(f"    Average memory overlap:          {avg_overlap:.1f}%")

    if test_mode:
        print("\n  Run without --test to process all 20 scenarios.")
    if long_mode:
        print("\n  Run without --long to use standard 20 scenarios.")
    print()


if __name__ == "__main__":
    main()
