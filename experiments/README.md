# Kokoro Downstream Evaluation

This experiment tests whether Kokoro's emotional trajectory memory makes an AI companion meaningfully better at responding to users — not in theory, but as judged by a language model evaluating real responses.

## What it measures

For each of 20 multi-session conversation scenarios (covering arcs like gradual burnout, grief, slow recovery, chronic anxiety, and post-traumatic growth), two companion responses are generated from the same user message: **Condition A** uses only semantic retrieval (`alpha=1.0`, no state summary), while **Condition B** uses Kokoro's hybrid retrieval plus the emotional state summary (`alpha=0.6`). A separate judge LLM — blind to which condition is which — reads both responses and picks the one that shows more emotional awareness of the user's history. The metric is how often B (Kokoro) wins.

## How to run

```bash
# From the project root
export GROQ_API_KEY=your_key_here       # or add to .env
python experiments/run_evaluation.py
```

Requires `groq` (`pip install groq`) and a valid Groq API key. Results are printed live with a running tally and saved to `experiments/results.json` when complete. The full run takes roughly 5–10 minutes (one Groq call per scenario for each of: response A, response B, and the judge verdict).

## What the results mean

A **Kokoro win rate above 50%** indicates that the emotional trajectory context — the state summary and emotionally-weighted retrieval — is producing responses that a neutral evaluator considers more appropriate to the user's situation. The breakdown by arc type in the final output shows which emotional contexts benefit most from trajectory-aware memory. A high win rate on `gradual_decline` and `grief_arc` scenarios would suggest Kokoro is most valuable when the user's current message is ambiguous but their history carries significant emotional weight.
