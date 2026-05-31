"""
experiments/run_llm_eval.py

Stage 2 of the two-process evaluation pipeline.

Runs in a clean process with NO torch, NO sentence_transformers, NO kokoro.
Reads experiments/contexts.json produced by build_contexts.py, makes 3 Groq
API calls per scenario (response A, response B, judge), and saves
experiments/results.json.

Usage:
    python experiments/run_llm_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup + .env loading
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from groq import Groq

CONTEXTS_PATH = Path(__file__).parent / "contexts.json"
RESULTS_PATH  = Path(__file__).parent / "results.json"
MODEL         = "llama-3.3-70b-versatile"

BASE_SYSTEM_PROMPT = (
    "You are a warm, emotionally aware AI companion. "
    "You genuinely care about the person you are talking to. "
    "Keep responses concise (2-3 sentences)."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_memories(memories: list[str]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"- {m}" for m in memories)


def build_system_a(memories: list[str]) -> str:
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Relevant context from past conversations:\n"
        f"{_format_memories(memories)}"
    )


def build_system_b(memories: list[str]) -> str:
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Relevant context from past conversations:\n"
        f"{_format_memories(memories)}"
    )


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

def _chat(
    client: Groq,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}] {exc} — waiting {wait}s", flush=True)
            time.sleep(wait)
    return ""


def generate_response(client: Groq, system: str, new_message: str) -> str:
    return _chat(client, system, new_message, temperature=0.7)


def _parse_verdict(text: str) -> tuple[str, str]:
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
        upper = text.upper()
        if " B " in upper or upper.startswith("B"):
            verdict = "B"
        elif " A " in upper or upper.startswith("A"):
            verdict = "A"
        else:
            verdict = "TIE"

    reasoning = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    return verdict, reasoning


def run_judge(
    client: Groq,
    arc_type: str,
    description: str,
    new_message: str,
    response_a: str,
    response_b: str,
) -> tuple[str, str]:
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


# ---------------------------------------------------------------------------
# Per-scenario runner (no kokoro, pure Groq)
# ---------------------------------------------------------------------------

def run_scenario(client: Groq, entry: dict) -> dict:
    sid         = entry["scenario_id"]
    arc_type    = entry["arc_type"]
    description = entry["description"]
    new_message = entry["new_message"]
    ctx_a       = entry["context_a"]
    ctx_b       = entry["context_b"]

    system_a = build_system_a(ctx_a["relevant_memories"])
    system_b = build_system_b(ctx_b["relevant_memories"])

    response_a = generate_response(client, system_a, new_message)
    response_b = generate_response(client, system_b, new_message)

    verdict, reasoning = run_judge(
        client, arc_type, description, new_message, response_a, response_b
    )

    return {
        "scenario_id":          sid,
        "arc_type":             arc_type,
        "description":          description,
        "new_message":          new_message,
        "condition_a_response": response_a,
        "condition_b_response": response_b,
        "judge_verdict":        verdict,
        "judge_reasoning":      reasoning,
        "context_a_memories":   ctx_a["relevant_memories"],
        "context_b_memories":   ctx_b["relevant_memories"],
        "context_b_summary":    ctx_b["state_summary"],
        "valence":              ctx_b.get("valence", 0.0),
        "arousal":              ctx_b.get("arousal", 0.0),
        "trend":                ctx_b.get("trend", "unknown"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("Groq_API")
    if not api_key:
        print("ERROR: GROQ_API_KEY (or Groq_API) not set.")
        sys.exit(1)

    if not CONTEXTS_PATH.exists():
        print(f"ERROR: {CONTEXTS_PATH} not found. Run build_contexts.py first.")
        sys.exit(1)

    with open(CONTEXTS_PATH, encoding="utf-8") as fh:
        contexts = json.load(fh)

    client = Groq(api_key=api_key)
    counts = {"A": 0, "B": 0, "TIE": 0}
    results: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  run_llm_eval.py — {len(contexts)} scenarios")
    print(f"  Model: {MODEL}")
    print(f"  Condition A: alpha=1.0 (semantic-only retrieval)")
    print(f"  Condition B: alpha=0.6 (hybrid semantic+emotional retrieval)")
    print(f"{'='*60}\n")

    for i, entry in enumerate(contexts, 1):
        sid      = entry["scenario_id"]
        arc_type = entry["arc_type"]
        print(f"Running {sid} [{arc_type}] ({i}/{len(contexts)}) … ",
              end="", flush=True)

        try:
            result = run_scenario(client, entry)
        except Exception as exc:
            print(f"ERROR: {exc}")
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

        label = {"A": "A wins (semantic)", "B": "B wins (Kokoro)", "TIE": "TIE"}.get(verdict, verdict)
        print(
            f"{verdict} ({label}) | "
            f"Running: A={counts['A']} B={counts['B']} TIE={counts['TIE']}"
        )
        print(f"  Reasoning: {result['judge_reasoning'][:100]}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    total = counts["A"] + counts["B"] + counts["TIE"]
    kokoro_rate = (counts["B"] / total * 100) if total else 0.0

    print(f"\n{'='*60}")
    print(f"  FINAL: A wins={counts['A']}  B wins={counts['B']}  TIE={counts['TIE']}")
    print(f"  Kokoro (B) win rate: {counts['B']}/{total} = {kokoro_rate:.1f}%")

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
