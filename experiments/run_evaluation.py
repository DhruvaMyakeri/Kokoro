"""
experiments/run_evaluation.py

Downstream evaluation experiment for Kokoro.

Compares two conditions for every scenario in scenarios.py:

  Condition A — alpha=1.0 (semantic-only retrieval, no state summary)
  Condition B — alpha=0.6 (hybrid retrieval + emotional state summary)

A Groq-hosted LLM generates companion responses for both conditions.
A separate judge call (blind to which is A or B) picks the better response.

Usage:
    export GROQ_API_KEY=<your_key>
    python experiments/run_evaluation.py

    # Or with .env file in project root:
    python experiments/run_evaluation.py

Results are saved to experiments/results.json.
"""

from __future__ import annotations

import os

# Force Groq/httpx/OpenSSL to initialize before PyTorch loads its own OpenSSL.
# On Windows, PyTorch ships a bundled OpenSSL that conflicts with the system one
# used by httpx — importing Groq first prevents the silent segfault.
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

from groq import Groq  # MUST be before any kokoro / sentence-transformers import

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or experiments/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env if present (python-dotenv is optional)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Patch ChromaDB before WorldMemory imports it.
# PersistentClient's native HNSW library segfaults in certain sandbox
# environments.  EphemeralClient is pure-Python and safe everywhere.
# We replace _mem_store immediately after WorldMemory.__init__ anyway, so
# no evaluation data ever touches ChromaDB — this patch just prevents the
# crash during __init__.
# ---------------------------------------------------------------------------
import chromadb as _chromadb
_chromadb.PersistentClient = lambda path, **kw: _chromadb.EphemeralClient()

from experiments.scenarios import SCENARIOS
from kokoro import WorldMemory


# ---------------------------------------------------------------------------
# In-memory session store — replaces ChromaDB for evaluation
# ---------------------------------------------------------------------------

class _InMemoryStore:
    """
    Pure-numpy session store used instead of ChromaDB during evaluation.

    Stores all sessions in a Python list and computes hybrid
    semantic + emotional retrieval with numpy cosine similarity.
    No file I/O, no native libraries, no cleanup needed.
    """

    def __init__(self) -> None:
        self._sessions: list[dict] = []

    # Minimal no-op _client so the finally-block cleanup doesn't crash
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
                "session_text":   user_sess[i]["session_text"],
                "session_id":     user_sess[i]["session_id"],
                "semantic_score": float(sem[i]),
                "emotional_score": float(emo[i]),
                "combined_score": float(combined[i]),
                "valence":        user_sess[i]["valence"],
                "arousal":        user_sess[i]["arousal"],
            }
            for i in top_idx
        ]

    def delete_user(self, user_id: str) -> int:
        before = len(self._sessions)
        self._sessions = [s for s in self._sessions if s["user_id"] != user_id]
        return before - len(self._sessions)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "llama-3.1-8b-instant"

BASE_SYSTEM_PROMPT = (
    "You are a warm, emotionally aware AI companion. "
    "You genuinely care about the person you are talking to. "
    "Keep responses concise (2-3 sentences)."
)

RESULTS_PATH = Path(__file__).parent / "results.json"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_memories(memories: list[str]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"- {m}" for m in memories)


def build_system_a(memories: list[str]) -> str:
    """Condition A: base prompt + semantic memories only, no state summary."""
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Relevant context from past conversations:\n"
        f"{_format_memories(memories)}"
    )


def build_system_b(state_summary: str | None, memories: list[str]) -> str:
    """Condition B: base prompt + emotional state summary + hybrid memories."""
    parts = [BASE_SYSTEM_PROMPT]
    if state_summary:
        parts.append(f"\n{state_summary}")
    parts.append(
        f"\nRelevant context from past conversations:\n"
        f"{_format_memories(memories)}"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Groq API helpers
# ---------------------------------------------------------------------------

def _chat(
    client: Groq,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """Single chat completion with simple retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user},
                ],
                temperature=temperature,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}] API error: {exc} — waiting {wait}s")
            time.sleep(wait)
    return ""   # unreachable


def generate_response(client: Groq, system: str, new_message: str) -> str:
    """Generate a companion response."""
    return _chat(client, system, new_message, temperature=0.7)


def run_judge(
    client: Groq,
    arc_type: str,
    description: str,
    new_message: str,
    response_a: str,
    response_b: str,
) -> tuple[str, str]:
    """
    Ask the judge to compare two responses.
    Returns (verdict, reasoning) where verdict ∈ {"A", "B", "TIE"}.

    The judge does not know which condition corresponds to A or B.
    """
    judge_system = (
        "You are evaluating two AI companion responses. "
        "Be objective and focus only on emotional awareness and appropriateness."
    )
    judge_user = (
        f"You are evaluating two AI companion responses.\n\n"
        f"User history summary: {arc_type} — {description}\n"
        f"User's new message: \"{new_message}\"\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Which response shows more emotional awareness and understanding of "
        f"the user's situation? Consider: does it acknowledge the user's "
        f"emotional history? Is it appropriate given what the user has been "
        f"going through?\n\n"
        f"Reply with exactly one of: A, B, or TIE\n"
        f"Then one sentence explaining why."
    )
    raw = _chat(client, judge_system, judge_user, temperature=0.0)
    return _parse_verdict(raw)


def _parse_verdict(text: str) -> tuple[str, str]:
    """
    Parse judge output into (verdict, reasoning).
    Verdict is the first non-empty token from {A, B, TIE}.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "TIE", text

    first = lines[0].upper()
    if first.startswith("A"):
        verdict = "A"
    elif first.startswith("B"):
        verdict = "B"
    elif "TIE" in first:
        verdict = "TIE"
    else:
        # Fall back: scan the full text
        upper = text.upper()
        if " B " in upper or upper.startswith("B"):
            verdict = "B"
        elif " A " in upper or upper.startswith("A"):
            verdict = "A"
        else:
            verdict = "TIE"

    reasoning = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    return verdict, reasoning


# ---------------------------------------------------------------------------
# Per-scenario runner
# ---------------------------------------------------------------------------

def run_scenario(client: Groq, scenario: dict) -> dict:
    """
    Build WorldMemory for the scenario, get both contexts, generate responses,
    run the judge, and return the full result dict.
    """
    sid         = scenario["scenario_id"]
    arc_type    = scenario["arc_type"]
    description = scenario["description"]
    sessions    = scenario["sessions"]
    new_message = scenario["new_message"]

    db_tmp  = tempfile.TemporaryDirectory()
    mem_tmp = tempfile.TemporaryDirectory()

    try:
        memory = WorldMemory(
            user_id       = f"eval_{sid}",
            db_path       = Path(db_tmp.name) / "kokoro.db",
            persist_dir   = mem_tmp.name,
            min_sessions  = 3,
            alpha         = 0.6,   # instance default (used by Condition B)
            top_k         = 3,
        )

        # Replace ChromaDB-backed store with the pure-numpy in-memory store.
        # update() calls add_session() on _mem_store, so sessions accumulate
        # here; get_context() calls _retrieve() which also uses _mem_store.
        memory._mem_store = _InMemoryStore()
        print("[dbg] store swapped", flush=True)

        # Feed all sessions into the state trajectory
        for i, session in enumerate(sessions):
            memory.update(session)
            print(f"[dbg] update {i+1}/{len(sessions)} done", flush=True)

        # Condition A: purely semantic retrieval, no state summary
        ctx_a = memory.get_context(new_message, alpha=1.0)
        print("[dbg] ctx_a done", flush=True)

        # Condition B: hybrid retrieval + state summary (instance default alpha)
        ctx_b = memory.get_context(new_message, alpha=0.6)
        print("[dbg] ctx_b done", flush=True)

        system_a = build_system_a(ctx_a["relevant_memories"])
        system_b = build_system_b(ctx_b["state_summary"], ctx_b["relevant_memories"])

        print("[dbg] calling generate_response A ...", flush=True)
        response_a = generate_response(client, system_a, new_message)
        print("[dbg] response_a done", flush=True)
        response_b = generate_response(client, system_b, new_message)
        print("[dbg] response_b done", flush=True)

        verdict, reasoning = run_judge(
            client, arc_type, description, new_message, response_a, response_b
        )

        return {
            "scenario_id":         sid,
            "arc_type":            arc_type,
            "description":         description,
            "new_message":         new_message,
            "condition_a_response": response_a,
            "condition_b_response": response_b,
            "judge_verdict":        verdict,
            "judge_reasoning":      reasoning,
            "context_a_memories":   ctx_a["relevant_memories"],
            "context_b_memories":   ctx_b["relevant_memories"],
            "context_b_summary":    ctx_b["state_summary"],
            "session_count":        ctx_b["session_count"],
            "valence":              ctx_b["valence"],
            "arousal":              ctx_b["arousal"],
            "trend":                ctx_b["trend"],
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
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("Groq_API")
    if not api_key:
        print("ERROR: GROQ_API_KEY (or Groq_API) not set. Export it or add it to .env")
        sys.exit(1)

    client = Groq(api_key=api_key)

    counts = {"A": 0, "B": 0, "TIE": 0}
    results: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Kokoro downstream evaluation — {len(SCENARIOS)} scenarios")
    print(f"  Model: {MODEL}")
    print(f"  Condition A: alpha=1.0 (semantic only, no state summary)")
    print(f"  Condition B: alpha=0.6 (hybrid + emotional state summary)")
    print(f"{'='*60}\n")

    for i, scenario in enumerate(SCENARIOS, 1):
        sid      = scenario["scenario_id"]
        arc_type = scenario["arc_type"]
        print(f"Running {sid} [{arc_type}] ({i}/{len(SCENARIOS)}) …", end=" ", flush=True)

        try:
            result = run_scenario(client, scenario)
        except Exception as exc:
            print(f"ERROR: {exc}")
            # Record failure and continue
            results.append({
                "scenario_id":  sid,
                "arc_type":     arc_type,
                "error":        str(exc),
                "judge_verdict": "ERROR",
            })
            continue

        verdict = result["judge_verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
        results.append(result)

        label = {
            "A":   "A wins (semantic)",
            "B":   "B wins (Kokoro)",
            "TIE": "TIE",
        }.get(verdict, verdict)

        total = counts["A"] + counts["B"] + counts["TIE"]
        print(
            f"{verdict} ({label}) | "
            f"Running: A={counts['A']} B={counts['B']} TIE={counts['TIE']}"
        )
        print(f"  Reasoning: {result['judge_reasoning'][:100]}")

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ------------------------------------------------------------------ #
    # Final summary                                                        #
    # ------------------------------------------------------------------ #
    total = counts["A"] + counts["B"] + counts["TIE"]
    kokoro_rate = (counts["B"] / total * 100) if total else 0.0

    print(f"\n{'='*60}")
    print(f"  FINAL: A wins={counts['A']}  B wins={counts['B']}  TIE={counts['TIE']}")
    print(f"  Kokoro (B) win rate: {counts['B']}/{total} = {kokoro_rate:.1f}%")

    # Arc-type breakdown
    arc_verdicts: dict[str, list[str]] = {}
    for r in results:
        arc = r.get("arc_type", "unknown")
        v   = r.get("judge_verdict", "ERROR")
        arc_verdicts.setdefault(arc, []).append(v)

    print(f"\n  Breakdown by arc type:")
    for arc, verdicts in sorted(arc_verdicts.items()):
        b_wins = verdicts.count("B")
        print(f"    {arc:<32} B={b_wins}/{len(verdicts)}  {verdicts}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
