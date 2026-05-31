"""
examples/basic_usage.py

Complete standalone example of Kokoro — no API key required.

Shows the full pattern:
  1. Create a WorldMemory for a user
  2. Feed conversation sessions as they happen
  3. Call get_context() before each LLM call
  4. Use the returned memories and state_summary to build a system prompt

Run:
    python examples/basic_usage.py
"""

import sys
import tempfile
from pathlib import Path

# Allow running from project root or examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro import WorldMemory

# ---------------------------------------------------------------------------
# Step 1: Create a WorldMemory instance for a user.
#
# db_path      — SQLite file that persists the emotional state trajectory.
# persist_dir  — ChromaDB directory for the session memory store.
# min_sessions — How many sessions must be logged before a state summary
#                is generated. Default is 3; we use 2 here so the example
#                shows a summary after fewer sessions.
# alpha        — Default retrieval blend: 0.6 = 60% semantic + 40% emotional.
# top_k        — How many past sessions to surface per get_context() call.
# ---------------------------------------------------------------------------

tmp = tempfile.mkdtemp()  # Use a temp dir so this example leaves no files

memory = WorldMemory(
    user_id     = "alice",
    db_path     = Path(tmp) / "kokoro.db",
    persist_dir = tmp,
    min_sessions = 2,   # Lower for demo purposes
    alpha        = 0.6,
    top_k        = 3,
)

print("=" * 60)
print("Kokoro basic usage example")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 2: Feed conversation sessions as they happen.
#
# Each session is a list of {"role": ..., "content": ...} turns — the same
# format as OpenAI / Anthropic chat messages.
#
# Call memory.update(session) at the END of each conversation session.
# Kokoro encodes the session text, updates the emotional state trajectory,
# and stores the session in the memory retrieval index.
# ---------------------------------------------------------------------------

SESSION_1 = [
    {"role": "user",      "content": "Work has been brutal lately. My manager keeps piling things on."},
    {"role": "assistant", "content": "That sounds exhausting. How long has it been like this?"},
    {"role": "user",      "content": "Maybe two months? I used to love this job. Now I dread Mondays."},
]

SESSION_2 = [
    {"role": "user",      "content": "Missed another deadline today. First time in two years. Felt awful."},
    {"role": "assistant", "content": "Missing a deadline when you care so much about your work is hard."},
    {"role": "user",      "content": "Yeah. My brain just feels scattered. Can't focus on anything."},
]

SESSION_3 = [
    {"role": "user",      "content": "Had a good-ish day. Finished the sprint at least."},
    {"role": "assistant", "content": "That's something. How are you feeling compared to last week?"},
    {"role": "user",      "content": "Still pretty empty. Not relieved. Just... done."},
]

for i, session in enumerate([SESSION_1, SESSION_2, SESSION_3], 1):
    memory.update(session)
    print(f"\nSession {i} logged.")

# ---------------------------------------------------------------------------
# Step 3: Call get_context() before sending the user's next message to the LLM.
#
# Returns a dict with:
#   relevant_memories  — list of past session excerpts ranked by relevance
#   state_summary      — plain-English description of the user's emotional arc
#                        (None until min_sessions sessions are logged)
#   ready              — True once enough sessions exist for a state summary
#   valence            — float, emotional valence of most recent state
#   arousal            — float, activation level of most recent state
#   trend              — "improving" | "declining" | "stable"
#   session_count      — number of sessions logged so far
# ---------------------------------------------------------------------------

new_message = "still here lol"

context = memory.get_context(new_message)

print("\n" + "=" * 60)
print(f"User's new message: {new_message!r}")
print("=" * 60)

print(f"\nSessions logged : {context['session_count']}")
print(f"Ready           : {context['ready']}")
print(f"Valence         : {context['valence']:+.3f}")
print(f"Arousal         : {context['arousal']:+.3f}")
print(f"Trend           : {context['trend']}")

print(f"\nState summary:")
print(f"  {context['state_summary']}")

print(f"\nRelevant memories ({len(context['relevant_memories'])}):")
for mem in context["relevant_memories"]:
    print(f"  - {mem[:100]}...")

# ---------------------------------------------------------------------------
# Step 4: Build a system prompt from the context and call your LLM.
#
# Kokoro does not make any LLM calls itself — it just gives you the context.
# You decide how to inject it into your system prompt.
# ---------------------------------------------------------------------------

def build_system_prompt(context: dict) -> str:
    parts = [
        "You are a warm, emotionally aware AI companion. "
        "You genuinely care about the person you are talking to."
    ]

    # Inject emotional state summary if available
    if context["state_summary"]:
        parts.append(f"\n{context['state_summary']}")

    # Inject relevant past memories
    if context["relevant_memories"]:
        memories_text = "\n".join(f"- {m}" for m in context["relevant_memories"])
        parts.append(f"\nContext from past conversations:\n{memories_text}")

    return "\n".join(parts)


system_prompt = build_system_prompt(context)

print("\n" + "=" * 60)
print("System prompt that would be sent to your LLM:")
print("=" * 60)
print(system_prompt)

# Mock LLM response to show the full pattern without needing an API key
mock_response = (
    "Hey. 'Still here' — I'll take that. "
    "You've been carrying a lot these past months. How are you actually doing?"
)

print("\n" + "=" * 60)
print("Mock LLM response:")
print("=" * 60)
print(mock_response)
print()
