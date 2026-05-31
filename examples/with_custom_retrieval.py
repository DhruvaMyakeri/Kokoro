"""
examples/with_custom_retrieval.py

Kokoro with a pluggable retrieval_fn — bring your own vector database.

Use this pattern when:
  - You already have a vector DB (Pinecone, Weaviate, pgvector, Qdrant, etc.)
  - You want Kokoro to handle emotional state tracking ONLY
  - Kokoro's MemoryStore (ChromaDB) is replaced by your existing RAG system

How it works:
  - You pass retrieval_fn to WorldMemory's constructor
  - Kokoro calls retrieval_fn(user_id, query_emb, state_vec, top_k) instead
    of its built-in ChromaDB store
  - Your function receives the query embedding AND the current emotional
    state vector, so you can apply emotional reranking in your own store
  - Kokoro still handles: session encoding, state transitions, state summaries

Run:
    python examples/with_custom_retrieval.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro import WorldMemory

# ---------------------------------------------------------------------------
# Step 1: Build a minimal stand-in for your existing vector database.
#
# In a real app this would be Pinecone, pgvector, Qdrant, etc.
# This in-memory stub has the same interface your retrieval_fn must satisfy.
# ---------------------------------------------------------------------------

class MyVectorDB:
    """Stand-in for an existing vector database."""

    def __init__(self) -> None:
        self._docs: list[dict] = []

    def upsert(self, doc_id: str, embedding: np.ndarray, metadata: dict) -> None:
        """Store a document with its embedding and metadata."""
        self._docs.append({
            "id":        doc_id,
            "embedding": embedding.astype(np.float32).copy(),
            "metadata":  metadata,
        })

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Return top_k docs by cosine similarity."""
        if not self._docs:
            return []
        stored = np.array([d["embedding"] for d in self._docs], dtype=np.float32)
        q = query_embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        norms = np.linalg.norm(stored, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        scores = (stored / norms) @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {**self._docs[i]["metadata"], "score": float(scores[i])}
            for i in top_idx
        ]


my_db = MyVectorDB()

# ---------------------------------------------------------------------------
# Step 2: Define retrieval_fn.
#
# Signature: (user_id, query_embedding, state_vector, top_k) -> list[dict]
#
# - user_id        str           the user whose memories to retrieve
# - query_embedding np.ndarray  384-dim embedding of the current message
# - state_vector    np.ndarray  current emotional state vector from Kokoro
# - top_k           int         how many results to return
#
# Return a list of dicts. Each dict must have at least:
#   {"session_text": str}
# All other keys are passed through to get_context()["relevant_memories"].
#
# You can use state_vector to apply emotional reranking in your own store.
# Here we blend semantic similarity and emotional similarity, same as Kokoro's
# built-in store — but your logic can be anything.
# ---------------------------------------------------------------------------

def my_retrieval_fn(
    user_id: str,
    query_embedding: np.ndarray,
    state_vector: np.ndarray,
    top_k: int,
) -> list[dict]:
    """
    Custom retrieval using MyVectorDB.

    Blends semantic similarity (query vs stored embedding) with emotional
    similarity (current state vector vs stored embedding) at alpha=0.6.
    Your implementation can use any logic — full-text BM25, hybrid search,
    metadata filters, etc.
    """
    # Pull candidates from our vector DB — fetch 2× and rerank
    candidates = my_db.query(query_embedding, top_k=min(top_k * 2, len(my_db._docs)))

    if not candidates:
        return []

    # Rerank using emotional state similarity
    alpha = 0.6
    reranked = []
    for c in candidates:
        stored_emb = np.array(c.get("_embedding", query_embedding), dtype=np.float32)
        sem_score = c["score"]

        # Emotional similarity: cosine(state_vector, stored_embedding)
        sv = state_vector.astype(np.float32)
        sv_norm = sv / (np.linalg.norm(sv) + 1e-12)
        emb_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-12)
        emo_score = float(np.dot(sv_norm, emb_norm))

        combined = alpha * sem_score + (1 - alpha) * emo_score
        reranked.append({**c, "combined_score": combined})

    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Step 3: Create WorldMemory with retrieval_fn.
#
# Kokoro will call my_retrieval_fn instead of its built-in MemoryStore.
# State tracking (StateStore, StateDecoder, TransitionModel) still runs
# normally — only the retrieval step is replaced.
# ---------------------------------------------------------------------------

tmp = tempfile.mkdtemp()

memory = WorldMemory(
    user_id      = "alice",
    db_path      = Path(tmp) / "kokoro.db",
    persist_dir  = tmp,           # ChromaDB dir (created but not used for retrieval)
    min_sessions = 3,
    alpha        = 0.6,
    top_k        = 2,
    retrieval_fn = my_retrieval_fn,  # <-- plug in your vector DB here
)

# ---------------------------------------------------------------------------
# Step 4: Also hook into the session storage side.
#
# When memory.update(session) is called, Kokoro stores the session embedding
# in its built-in MemoryStore. Since we replaced retrieval, we also need to
# populate our own DB.
#
# Pattern: wrap update() to also upsert into your DB.
# ---------------------------------------------------------------------------

from kokoro import SessionEncoder

encoder = SessionEncoder()  # same encoder Kokoro uses internally

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

print("Logging sessions into Kokoro (state tracking) and MyVectorDB (retrieval)...")
for i, session in enumerate(PAST_SESSIONS, 1):
    # Let Kokoro update its state trajectory
    memory.update(session)

    # Also upsert into our own DB so retrieval_fn can find it
    session_text = " ".join(t["content"] for t in session)
    session_emb  = encoder.encode(session)  # 384-dim ndarray
    my_db.upsert(
        doc_id    = f"alice_session_{i}",
        embedding = session_emb,
        metadata  = {
            "session_text": session_text[:200],
            "_embedding":   session_emb.tolist(),  # stored for emotional reranking
            "user_id":      "alice",
            "session_num":  i,
        },
    )
    print(f"  Session {i} logged.")

# ---------------------------------------------------------------------------
# Step 5: get_context() — Kokoro calls your retrieval_fn internally.
# ---------------------------------------------------------------------------

new_message = "still here lol"
context = memory.get_context(new_message)

print(f"\nUser: {new_message!r}")
print(f"\nState summary : {context['state_summary']}")
print(f"Valence       : {context['valence']:+.3f}")
print(f"Trend         : {context['trend']}")
print(f"\nRetrieved via my_retrieval_fn ({len(context['relevant_memories'])} results):")
for mem in context["relevant_memories"]:
    print(f"  - {mem[:100]}")

# ---------------------------------------------------------------------------
# Step 6: Build system prompt — identical to any other Kokoro integration.
# ---------------------------------------------------------------------------

system_prompt_parts = [
    "You are a warm, emotionally aware AI companion.",
]
if context["state_summary"]:
    system_prompt_parts.append(context["state_summary"])
if context["relevant_memories"]:
    mems = "\n".join(f"- {m}" for m in context["relevant_memories"])
    system_prompt_parts.append(f"Context from past conversations:\n{mems}")

system_prompt = "\n\n".join(system_prompt_parts)

print(f"\nSystem prompt that would be sent to the LLM:")
print("-" * 50)
print(system_prompt)
print()
