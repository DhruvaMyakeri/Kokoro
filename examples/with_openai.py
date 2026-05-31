"""
examples/with_openai.py

Kokoro + OpenAI (or any OpenAI-compatible API: Groq, Together, Anthropic proxy).

Shows the exact integration pattern:
  1. Kokoro tracks the user's emotional trajectory across sessions
  2. Before each LLM call, get_context() returns memories and a state summary
  3. Both are injected into the system prompt
  4. The LLM responds with awareness of the user's history

Run:
    export OPENAI_API_KEY=sk-...          # or GROQ_API_KEY for Groq
    python examples/with_openai.py

To use Groq instead of OpenAI, set base_url and model below.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro import WorldMemory

# ---------------------------------------------------------------------------
# Config — swap base_url / model to use Groq, Together, etc.
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
BASE_URL       = None          # None = OpenAI default; "https://api.groq.com/openai/v1" for Groq
MODEL          = "gpt-4o-mini" # or "llama-3.3-70b-versatile" for Groq

if not OPENAI_API_KEY:
    print("ERROR: Set OPENAI_API_KEY (or GROQ_API_KEY) to run this example.")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

client_kwargs = {"api_key": OPENAI_API_KEY}
if BASE_URL:
    client_kwargs["base_url"] = BASE_URL
llm = OpenAI(**client_kwargs)

# ---------------------------------------------------------------------------
# Step 1: Set up Kokoro memory for the user
# ---------------------------------------------------------------------------

tmp = tempfile.mkdtemp()

memory = WorldMemory(
    user_id     = "alice",
    db_path     = Path(tmp) / "kokoro.db",
    persist_dir = tmp,
    min_sessions = 3,
    alpha        = 0.6,   # 60% semantic + 40% emotional retrieval
    top_k        = 3,
)

# ---------------------------------------------------------------------------
# Step 2: Simulate three past sessions being logged over time.
#
# In a real app these would be stored at the END of each conversation.
# ---------------------------------------------------------------------------

PAST_SESSIONS = [
    [
        {"role": "user",      "content": "Work has been brutal. Manager keeps piling things on."},
        {"role": "assistant", "content": "That sounds exhausting. How long has this been going on?"},
        {"role": "user",      "content": "Two months? I used to love this job. Now I dread Mondays."},
    ],
    [
        {"role": "user",      "content": "Missed a deadline today. First time in two years."},
        {"role": "assistant", "content": "That's hard when you care so much about your work."},
        {"role": "user",      "content": "Yeah. Brain feels scattered. Can't focus on anything."},
    ],
    [
        {"role": "user",      "content": "Finished the sprint but feel completely empty."},
        {"role": "assistant", "content": "How are you feeling compared to last week?"},
        {"role": "user",      "content": "Not relieved. Just done. I don't know how much longer I can do this."},
    ],
]

print("Logging past sessions...")
for i, session in enumerate(PAST_SESSIONS, 1):
    memory.update(session)
    print(f"  Session {i} logged.")

# ---------------------------------------------------------------------------
# Step 3: User sends a new message. Get context BEFORE calling the LLM.
# ---------------------------------------------------------------------------

new_message = "still here lol"

# Kokoro retrieves the most emotionally and semantically relevant past memories
# and generates a state summary based on the trajectory.
context = memory.get_context(new_message)

print(f"\nUser: {new_message!r}")
print(f"State summary: {context['state_summary']}")
print(f"Memories retrieved: {len(context['relevant_memories'])}")

# ---------------------------------------------------------------------------
# Step 4: Build system prompt — this is where Kokoro plugs into your LLM call.
# ---------------------------------------------------------------------------

def build_system_prompt(ctx: dict) -> str:
    """
    Construct a system prompt from Kokoro context.

    The two key pieces are:
      ctx["state_summary"]      — plain-English emotional arc description
      ctx["relevant_memories"]  — ranked list of relevant past session excerpts
    """
    lines = [
        "You are a warm, emotionally aware AI companion. "
        "You genuinely care about the person you are talking to. "
        "Keep responses concise (2-3 sentences).",
    ]

    # Emotional state context — tells the LLM how the user has been feeling
    if ctx["state_summary"]:
        lines.append(f"\n{ctx['state_summary']}")

    # Episodic memory — specific moments from past conversations
    if ctx["relevant_memories"]:
        mems = "\n".join(f"- {m}" for m in ctx["relevant_memories"])
        lines.append(f"\nContext from past conversations:\n{mems}")

    return "\n".join(lines)


system_prompt = build_system_prompt(context)

# ---------------------------------------------------------------------------
# Step 5: Call the LLM with the Kokoro-enriched system prompt.
# ---------------------------------------------------------------------------

response = llm.chat.completions.create(
    model    = MODEL,
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": new_message},
    ],
    max_tokens  = 150,
    temperature = 0.7,
)

assistant_reply = response.choices[0].message.content.strip()

print(f"\nAssistant: {assistant_reply}")
print()
