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
from kokoro import WorldMemory

CONTEXTS_PATH = Path(__file__).parent / "contexts.json"


# ---------------------------------------------------------------------------
# In-memory session store — replaces ChromaDB during context building
# ---------------------------------------------------------------------------

class _InMemoryStore:
    """Pure-numpy session store: no native libs, no file I/O."""

    def __init__(self) -> None:
        self._sessions: list[dict] = []

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
        valence: float,
        arousal: float,
    ) -> None:
        self._sessions.append({
            "user_id":      user_id,
            "session_id":   session_id,
            "session_text": session_text,
            "embedding":    session_embedding.astype(np.float32).copy(),
            "valence":      float(valence),
            "arousal":      float(arousal),
        })

    def retrieve(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        state_vector: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.6,
    ) -> list[dict]:
        user_sess = [s for s in self._sessions if s["user_id"] == user_id]
        if not user_sess:
            return []

        stored = np.array([s["embedding"] for s in user_sess], dtype=np.float32)

        def _cosine(q: np.ndarray, S: np.ndarray) -> np.ndarray:
            q = q.astype(np.float32)
            q_norm = q / (np.linalg.norm(q) + 1e-12)
            norms = np.linalg.norm(S, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            return (S / norms) @ q_norm

        sem      = _cosine(query_embedding, stored)
        emo      = _cosine(state_vector, stored)
        combined = alpha * sem + (1.0 - alpha) * emo

        k       = min(top_k, len(user_sess))
        top_idx = np.argsort(-combined)[:k]

        return [
            {
                "session_text":    user_sess[i]["session_text"],
                "session_id":      user_sess[i]["session_id"],
                "semantic_score":  float(sem[i]),
                "emotional_score": float(emo[i]),
                "combined_score":  float(combined[i]),
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
        memory._mem_store = _InMemoryStore()

        for session in sessions:
            memory.update(session)

        ctx_a = memory.get_context(new_message, alpha=1.0)
        ctx_b = memory.get_context(new_message, alpha=0.6)

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
    test_mode = "--test" in sys.argv
    scenarios = SCENARIOS[:3] if test_mode else SCENARIOS
    label     = f"first 3 (--test mode)" if test_mode else f"all {len(scenarios)}"

    print(f"\n{'='*60}")
    print(f"  build_contexts.py — {label} scenarios")
    print(f"  Output: {CONTEXTS_PATH}")
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
            mem_a = len(ctx["context_a"]["relevant_memories"])
            mem_b = len(ctx["context_b"]["relevant_memories"])
            summary_snippet = (ctx["context_b"]["state_summary"] or "")[:60]
            print(f"done  [A:{mem_a} mems | B:{mem_b} mems | {summary_snippet!r}]")
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    CONTEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTEXTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(results)} contexts to {CONTEXTS_PATH}")
    if test_mode:
        print("  Run without --test to process all 20 scenarios.")
    print()


if __name__ == "__main__":
    main()
