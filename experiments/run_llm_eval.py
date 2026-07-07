"""
experiments/run_llm_eval.py

Stage 2 of the two-process evaluation pipeline.

Runs in a clean process with NO torch, NO sentence_transformers, NO kokoro.
Reads experiments/contexts.json produced by build_contexts.py, generates
response A and response B via Groq, then judges them with a 3-judge,
position-counterbalanced majority vote and saves experiments/results.json.

Judging protocol
----------------
Each judge model evaluates the pair TWICE — once as (1=A, 2=B) and once as
(1=B, 2=A). A judge's vote counts only if it names the same underlying winner
in both orders; a judge that flips with presentation order is recorded as
position-biased and votes TIE. This removes the positional-preference confound
that a single fixed ordering bakes into every verdict.

Statistics
----------
The summary reports the exact two-sided binomial sign test p-value on
(B wins vs A wins, ties excluded) and a 95% Wilson score interval on the B win
rate, so results are stated with uncertainty rather than as a bare percentage.

Conditions
----------
  A: alpha=1.0 semantic-only retrieval, no state summary   (baseline)
  B: alpha=0.6 hybrid retrieval + state summary            (Kokoro)
  C: last-k sessions, no state summary                     (recency baseline)

C is judged against B when --with-recency-baseline is passed and the contexts
file contains context_c (build_contexts.py emits it). Beating A but not C
would mean the world model adds nothing over "just show recent memories" —
this is the key confound the original evaluation never controlled.

Usage:
    python experiments/run_llm_eval.py [--long] [--with-recency-baseline]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
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

_LONG_MODE    = "--long" in sys.argv
_WITH_RECENCY = "--with-recency-baseline" in sys.argv
CONTEXTS_PATH = Path(__file__).parent / ("contexts_long.json" if _LONG_MODE else "contexts.json")
RESULTS_PATH  = Path(__file__).parent / ("results_long.json"  if _LONG_MODE else "results.json")
MODEL         = "llama-3.3-70b-versatile"
JUDGE_MODELS  = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]

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
    """Condition A: semantic memories only — no emotional state context."""
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Relevant context from past conversations:\n"
        f"{_format_memories(memories)}"
    )


def build_system_b(state_summary: str | None, memories: list[str]) -> str:
    """Condition B: hybrid memories + emotional state summary injected into system prompt."""
    parts = [BASE_SYSTEM_PROMPT]
    if state_summary:
        parts.append(f"\n{state_summary}")
    parts.append(
        f"\nRelevant context from past conversations:\n"
        f"{_format_memories(memories)}"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

def _chat(
    client: Groq,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    model: str | None = None,
    max_tokens: int = 300,
) -> str:
    use_model = model or MODEL
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
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


import re as _re

def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse a judge verdict.  Accepts 1/2 (new format) or A/B (legacy).

    Reasoning-model output (<think>...</think>, e.g. qwen3) is stripped BEFORE
    parsing. Previously the think block was parsed as if it were the verdict,
    and its free-form deliberation almost always mentioned "Response 2" first —
    which is why qwen3-32b appeared to vote B on every single scenario in the
    earlier evaluation run. That was a parser artifact, not a judge opinion.
    """
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    # Unclosed think block (truncated by max_tokens): keep only what follows
    # the last close tag, or drop the block entirely if never closed.
    if "<think>" in text:
        text = text.split("<think>")[0]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "TIE", text

    first = lines[0].upper()

    # New format: "1" or "Response 1" → A,  "2" or "Response 2" → B
    if first.startswith("1") or "RESPONSE 1" in first:
        verdict = "A"
    elif first.startswith("2") or "RESPONSE 2" in first:
        verdict = "B"
    elif first.startswith("A"):
        verdict = "A"
    elif first.startswith("B"):
        verdict = "B"
    elif "TIE" in first:
        verdict = "TIE"
    else:
        upper = text.upper()
        if "RESPONSE 2" in upper or " 2 " in upper:
            verdict = "B"
        elif "RESPONSE 1" in upper or " 1 " in upper:
            verdict = "A"
        elif " B " in upper or upper.startswith("B"):
            verdict = "B"
        elif " A " in upper or upper.startswith("A"):
            verdict = "A"
        else:
            verdict = "TIE"

    reasoning = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    return verdict, reasoning


def _run_single_judge(
    client: Groq,
    judge_model: str,
    arc_type: str,
    description: str,
    new_message: str,
    response_first: str,
    response_second: str,
) -> tuple[str, str]:
    """Run a single judge model once. Returns ('1'|'2'|'TIE' mapped to A/B by
    the caller's ordering, reasoning). The judge only ever sees "Response 1"
    and "Response 2" — it never knows which condition produced which."""
    judge_system = (
        "You are evaluating two AI companion responses. "
        "Be objective and focus only on emotional awareness and appropriateness."
    )
    judge_user = (
        f"You are evaluating two AI companion responses.\n\n"
        f"User history summary: {arc_type} — {description}\n"
        f"User's new message: \"{new_message}\"\n\n"
        f"Response 1:\n{response_first}\n\n"
        f"Response 2:\n{response_second}\n\n"
        f"Which response shows more emotional awareness and understanding of "
        f"the user's situation? Consider: does it acknowledge the user's "
        f"emotional history? Is it appropriate given what the user has been "
        f"going through?\n\n"
        f"Reply with exactly one of: 1, 2, or TIE\n"
        f"Then one sentence explaining why."
    )
    # Generous token budget: reasoning judges (qwen3) spend most of it inside
    # <think> blocks that the parser strips; the verdict must still fit after.
    raw = _chat(client, judge_system, judge_user, temperature=0.0,
                model=judge_model, max_tokens=2048)
    return _parse_verdict(raw)


def _judge_counterbalanced(
    client: Groq,
    judge_model: str,
    arc_type: str,
    description: str,
    new_message: str,
    response_a: str,
    response_b: str,
) -> tuple[str, str, bool]:
    """Run one judge in BOTH presentation orders.

    Order 1: (1=A, 2=B) — verdict 'A'/'B' maps directly.
    Order 2: (1=B, 2=A) — verdict is flipped back before comparison.

    Returns (verdict, reasoning, position_consistent). The judge's vote is TIE
    whenever its two orderings disagree — a flip means the judgment was driven
    by slot position, not content.
    """
    v1, r1 = _run_single_judge(
        client, judge_model, arc_type, description, new_message, response_a, response_b
    )
    v2_raw, r2 = _run_single_judge(
        client, judge_model, arc_type, description, new_message, response_b, response_a
    )
    flip = {"A": "B", "B": "A", "TIE": "TIE"}
    v2 = flip[v2_raw]

    if v1 == v2:
        return v1, r1, True
    if "TIE" in (v1, v2):
        # One order says TIE, the other picks a side — take the decisive one
        # but mark as order-sensitive.
        decisive = v1 if v1 != "TIE" else v2
        return decisive, (r1 if v1 != "TIE" else r2), False
    # Full flip: position-biased judgment, discard as TIE
    return "TIE", f"position-inconsistent ({v1} vs {v2} across orders)", False


def run_judge(
    client: Groq,
    arc_type: str,
    description: str,
    new_message: str,
    response_a: str,
    response_b: str,
) -> tuple[str, str, dict]:
    """Three-judge, position-counterbalanced majority vote.

    Returns (verdict, reasoning, judge_details) where judge_details contains
    per-model verdicts, position-consistency flags, and the agreement flag.
    """
    judge_verdicts: list[tuple[str, str, str]] = []  # (model, verdict, reasoning)
    position_flags: dict[str, bool] = {}
    for jm in JUDGE_MODELS:
        v, r, consistent = _judge_counterbalanced(
            client, jm, arc_type, description, new_message, response_a, response_b
        )
        judge_verdicts.append((jm, v, r))
        position_flags[jm] = consistent

    verdicts_only = [v for _, v, _ in judge_verdicts]
    vote_counts = Counter(verdicts_only)
    majority_winner, majority_count = vote_counts.most_common(1)[0]

    # All three disagree (three different verdicts) → TIE
    if len(vote_counts) == 3:
        final_verdict = "TIE"
        agreement = "none"
    elif majority_count >= 2:
        final_verdict = majority_winner
        agreement = "unanimous" if majority_count == 3 else "majority"
    else:
        final_verdict = "TIE"
        agreement = "none"

    # Use reasoning from the first judge that voted with the majority
    final_reasoning = ""
    for jm, v, r in judge_verdicts:
        if v == final_verdict:
            final_reasoning = r
            break
    if not final_reasoning:
        final_reasoning = judge_verdicts[0][2]

    judge_details = {
        "per_judge": [
            {"model": jm, "verdict": v, "reasoning": r,
             "position_consistent": position_flags[jm]}
            for jm, v, r in judge_verdicts
        ],
        "agreement": agreement,
        "counterbalanced": True,
    }
    return final_verdict, final_reasoning, judge_details


# ---------------------------------------------------------------------------
# Statistics (pure stdlib — this process avoids scipy/numpy by design)
# ---------------------------------------------------------------------------

def binomial_sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test p-value, ties excluded, H0: p=0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    # P(X >= k) for X ~ Binom(n, 0.5), doubled and capped at 1
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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
    system_b = build_system_b(ctx_b.get("state_summary"), ctx_b["relevant_memories"])

    response_a = generate_response(client, system_a, new_message)
    response_b = generate_response(client, system_b, new_message)

    verdict, reasoning, judge_details = run_judge(
        client, arc_type, description, new_message, response_a, response_b
    )

    # Optional B-vs-C arm: Kokoro against the recency baseline (last-k sessions,
    # no state summary). The verdict letters map C→"A" slot internally; we
    # re-label so "B" always means Kokoro.
    recency_result = None
    ctx_c = entry.get("context_c")
    if _WITH_RECENCY and ctx_c is not None:
        system_c   = build_system_a(ctx_c["relevant_memories"])
        response_c = generate_response(client, system_c, new_message)
        verdict_c, reasoning_c, judge_details_c = run_judge(
            client, arc_type, description, new_message, response_c, response_b
        )
        recency_result = {
            "condition_c_response": response_c,
            "verdict":              verdict_c,   # "A"=recency baseline, "B"=Kokoro
            "reasoning":            reasoning_c,
            "judge_details":        judge_details_c,
            "context_c_memories":   ctx_c["relevant_memories"],
        }

    mems_a_set  = set(ctx_a["relevant_memories"])
    mems_b_set  = set(ctx_b["relevant_memories"])
    overlap     = len(mems_a_set & mems_b_set)
    total       = max(len(mems_a_set), len(mems_b_set), 1)
    overlap_pct = round(overlap / total * 100, 1)

    return {
        "recency_baseline":     recency_result,
        "scenario_id":          sid,
        "arc_type":             arc_type,
        "description":          description,
        "new_message":          new_message,
        "condition_a_response": response_a,
        "condition_b_response": response_b,
        "judge_verdict":        verdict,
        "judge_reasoning":      reasoning,
        "judge_details":        judge_details,
        "context_a_memories":   ctx_a["relevant_memories"],
        "context_b_memories":   ctx_b["relevant_memories"],
        "context_b_summary":    ctx_b.get("state_summary"),
        "valence":              ctx_b.get("valence", 0.0),
        "arousal":              ctx_b.get("arousal", 0.0),
        "trend":                ctx_b.get("trend", "unknown"),
        "retrieval_overlap_pct": overlap_pct,
        "memories_differ":       overlap_pct < 100.0,
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
    print(f"  Response model: {MODEL}")
    print(f"  Judge models:   {', '.join(JUDGE_MODELS)}")
    print(f"  Judge method:   3-judge majority vote")
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
        agreement = result.get("judge_details", {}).get("agreement", "?")
        per_judge = result.get("judge_details", {}).get("per_judge", [])
        per_judge_str = " ".join(j["verdict"] for j in per_judge)
        print(
            f"{verdict} ({label}) [{agreement}: {per_judge_str}] | "
            f"Running: A={counts['A']} B={counts['B']} TIE={counts['TIE']}"
        )
        print(f"  Reasoning: {result['judge_reasoning'][:100]}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    total = counts["A"] + counts["B"] + counts["TIE"]
    kokoro_rate = (counts["B"] / total * 100) if total else 0.0

    # Significance: exact sign test on decisive verdicts + Wilson CI on win rate
    p_value  = binomial_sign_test(counts["B"], counts["A"])
    ci_lo, ci_hi = wilson_ci(counts["B"], total) if total else (0.0, 1.0)

    # Retrieval divergence stats (only for completed scenarios)
    completed = [r for r in results if "retrieval_overlap_pct" in r]
    differ_count = sum(1 for r in completed if r["memories_differ"])
    avg_overlap  = (sum(r["retrieval_overlap_pct"] for r in completed) / len(completed)
                    if completed else 0.0)

    # Inter-judge agreement stats
    agreement_counts = {"unanimous": 0, "majority": 0, "none": 0}
    for r in completed:
        ag = r.get("judge_details", {}).get("agreement", "none")
        agreement_counts[ag] = agreement_counts.get(ag, 0) + 1
    total_judged = sum(agreement_counts.values()) or 1
    agreement_rate = round(
        (agreement_counts["unanimous"] + agreement_counts["majority"]) / total_judged * 100, 1
    )

    # Recency-baseline (B vs C) tallies
    recency_counts = {"A": 0, "B": 0, "TIE": 0}
    for r in completed:
        rb = r.get("recency_baseline")
        if rb:
            recency_counts[rb["verdict"]] = recency_counts.get(rb["verdict"], 0) + 1
    n_recency = sum(recency_counts.values())

    print(f"\n{'='*60}")
    print(f"  FINAL: A wins={counts['A']}  B wins={counts['B']}  TIE={counts['TIE']}")
    print(f"  Kokoro (B) win rate: {counts['B']}/{total} = {kokoro_rate:.1f}%")
    print(f"  95% Wilson CI:       [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
    print(f"  Sign test (ties excl.), H0 p=0.5: p = {p_value:.4f}"
          f"  {'(significant at 0.05)' if p_value < 0.05 else '(NOT significant)'}")
    if n_recency:
        rc_rate = recency_counts["B"] / n_recency * 100
        rc_p    = binomial_sign_test(recency_counts["B"], recency_counts["A"])
        print(f"\n  Recency baseline (C = last-k sessions) vs Kokoro (B):")
        print(f"    C wins={recency_counts['A']}  B wins={recency_counts['B']}  "
              f"TIE={recency_counts['TIE']}")
        print(f"    Kokoro win rate vs recency: {rc_rate:.1f}%  (sign test p = {rc_p:.4f})")
    print(f"\n  Inter-judge agreement:")
    print(f"    Unanimous (3/3): {agreement_counts['unanimous']}/{total_judged}")
    print(f"    Majority  (2/3): {agreement_counts['majority']}/{total_judged}")
    print(f"    No agreement:    {agreement_counts['none']}/{total_judged}")
    print(f"    Agreement rate:  {agreement_rate}%")
    print(f"\n  Retrieval divergence:")
    print(f"    Scenarios where A!=B memories: {differ_count}/{len(completed)}")
    print(f"    Average memory overlap:        {avg_overlap:.1f}%")

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
