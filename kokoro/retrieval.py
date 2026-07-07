# -*- coding: utf-8 -*-
"""
kokoro/retrieval.py — Hybrid emotionally-aware RAG memory store.

Retrieves sessions that are emotionally relevant to the current user state,
not just topically relevant to the current message. This is the second novel
contribution of Kokoro.

Retrieval score
---------------
  base  = α × semantic_score + (1-α) × emotional_score
  score = (1-γ) × base + γ × recency_score           (γ = recency_weight, default 0)

  semantic_score:  cosine_sim(current_message_embedding, stored_session_embedding),
                   rescaled from [-1, 1] to [0, 1] so it is commensurable with the
                   emotional axis — topical relevance via MiniLM on both sides.

  emotional_score: 1 - L2((v, a)_current, (v, a)_stored) / sqrt(8), in [1-sqrt(8)/sqrt(8), 1]
                   ≈ [0, 1] — proximity in the decoded valence/arousal circumplex.
                   State-to-state cosine is available as a fallback but is known to
                   be non-discriminative (all states lie in a tight angular cone).

  recency_score:   exp(-rank_age / tau) where rank_age is how many sessions ago the
                   memory was stored (0 = most recent). Controlled by recency_weight;
                   0.0 (default) preserves the original two-axis behaviour.

Adaptive alpha
--------------
  adaptive_alpha(base_alpha, va_stored, va_current) shifts weight back toward the
  semantic axis when the stored sessions are emotionally homogeneous (the emotional
  axis then carries no ranking information and only adds noise), and keeps or
  strengthens the emotional axis when the user's history spans distinct emotional
  phases — exactly the regime where phase-matched retrieval matters.

Public API
----------
MemoryStore(collection_name="kokoro", persist_dir=None)
store.add_session(user_id, session_id, session_text, session_embedding, state_vector, valence, arousal)
store.retrieve(user_id, query_embedding, state_vector, top_k=5, alpha=0.6,
               current_valence=None, current_arousal=None, recency_weight=0.0)
store.get_user_sessions(user_id)
store.delete_user(user_id)
adaptive_alpha(base_alpha, va_stored, va_current)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional ChromaDB import — fail clearly if not installed
# ---------------------------------------------------------------------------
try:
    import chromadb
except ImportError as _e:
    raise ImportError(
        "chromadb is required for kokoro.retrieval. "
        "Install it with:  pip install chromadb"
    ) from _e

_EMBED_DIM = 384
_DEFAULT_PERSIST_DIR = Path.home() / ".kokoro" / "memories"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of *v*. Raises ValueError if norm is zero."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


def _cosine_sim_batch(
    query: np.ndarray,
    stored: np.ndarray,
) -> np.ndarray:
    """Cosine similarities between one query vector and a batch of stored vectors.

    Parameters
    ----------
    query:  shape (D,)
    stored: shape (N, D)

    Returns
    -------
    scores: shape (N,), values in [-1, 1]
    """
    q = _l2_normalize(query.astype(np.float32))
    norms = np.linalg.norm(stored, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    s_normed = stored.astype(np.float32) / norms
    return (s_normed @ q).astype(np.float64)


def _doc_id(user_id: str, session_id: str) -> str:
    """Stable, unique document ID derived from user and session."""
    return hashlib.sha1(f"{user_id}\x00{session_id}".encode()).hexdigest()


def adaptive_alpha(
    base_alpha: float,
    va_stored: np.ndarray,
    va_current: tuple[float, float] | None = None,
    low_dispersion: float = 0.15,
    high_dispersion: float = 0.45,
) -> float:
    """Adjust the semantic/emotional blend based on how informative the emotional
    axis actually is for this user's history.

    Dispersion = mean L2 distance of stored (valence, arousal) points from their
    centroid. When all stored sessions sit in one emotional phase (dispersion below
    ``low_dispersion``), the emotional axis cannot re-rank anything meaningful and
    only injects decoder noise — alpha is pushed to 1.0 (pure semantic). When the
    history spans clearly distinct phases (dispersion above ``high_dispersion``),
    the caller's base_alpha is kept as-is. In between, alpha is linearly
    interpolated between base_alpha and 1.0.

    Parameters
    ----------
    base_alpha:  The configured alpha (e.g. 0.6).
    va_stored:   (N, 2) array of stored [valence, arousal] pairs.
    va_current:  Unused hook for future query-conditioned policies; kept in the
                 signature so callers can pass it without a version check.

    Returns
    -------
    Effective alpha in [base_alpha, 1.0].
    """
    va = np.asarray(va_stored, dtype=np.float64)
    if va.ndim != 2 or va.shape[0] < 2:
        return 1.0  # 0 or 1 stored sessions: emotional axis has nothing to rank
    centroid = va.mean(axis=0)
    dispersion = float(np.linalg.norm(va - centroid, axis=1).mean())
    if dispersion <= low_dispersion:
        return 1.0
    if dispersion >= high_dispersion:
        return float(base_alpha)
    # Linear interpolation between pure-semantic and the configured blend
    frac = (dispersion - low_dispersion) / (high_dispersion - low_dispersion)
    return float(1.0 - frac * (1.0 - base_alpha))


def _validate_embedding(arr: np.ndarray, name: str) -> np.ndarray:
    """Check shape and dtype; return float32 1-D array."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array of shape ({_EMBED_DIM},), "
            f"got shape {arr.shape}"
        )
    if arr.shape[0] != _EMBED_DIM:
        raise ValueError(
            f"{name} must have {_EMBED_DIM} dimensions, got {arr.shape[0]}"
        )
    return arr


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Hybrid emotionally-aware RAG memory store backed by ChromaDB.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection. Defaults to "kokoro".
    persist_dir:
        Directory for persistent storage. Defaults to ~/.kokoro/memories/.
        Pass a custom path (e.g. a tempdir) for testing.
    """

    def __init__(
        self,
        collection_name: str = "kokoro",
        persist_dir: str | Path | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir) if persist_dir else _DEFAULT_PERSIST_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Monotonic tiebreaker: time.time() has ~15ms resolution on Windows, so
        # sessions added in a tight loop can collide, corrupting recency ordering.
        self._insert_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_session(
        self,
        user_id: str,
        session_id: str,
        session_text: str,
        session_embedding: np.ndarray,
        state_vector: np.ndarray | None = None,
        valence: float = 0.0,
        arousal: float = 0.0,
    ) -> None:
        """Store a session summary for later retrieval.

        Parameters
        ----------
        user_id:           Unique identifier for the user.
        session_id:        Unique identifier for this session.
        session_text:      Human-readable summary of the session.
        session_embedding: 384-dim MiniLM encoder output — used for semantic axis.
        state_vector:      Optional 384-dim transition model output at this session.
                           Stored so the legacy state-to-state cosine fallback can
                           work; the preferred emotional axis uses (valence, arousal).
                           Omit it to use the store as a plain semantic RAG index.
        valence:           Predicted valence in [-1, 1] (default 0.0 = neutral).
        arousal:           Predicted arousal in [-1, 1] (default 0.0 = neutral).
        """
        emb = _validate_embedding(session_embedding, "session_embedding")
        valence = float(valence)
        arousal = float(arousal)

        metadata: dict[str, Any] = {
            "user_id":      user_id,
            "session_id":   session_id,
            "valence":      valence,
            "arousal":      arousal,
            "timestamp":    time.time() + self._insert_counter * 1e-6,
        }
        self._insert_counter += 1
        if state_vector is not None:
            sv = _validate_embedding(state_vector, "state_vector")
            metadata["state_vector"] = json.dumps(sv.tolist())

        doc_id = _doc_id(user_id, session_id)
        self._col.upsert(
            ids=[doc_id],
            embeddings=[emb.tolist()],
            documents=[session_text],
            metadatas=[metadata],
        )

    def retrieve(
        self,
        user_id: str,
        query_embedding: np.ndarray | None,
        state_vector: np.ndarray | None = None,
        top_k: int = 5,
        alpha: float = 0.6,
        current_valence: float | None = None,
        current_arousal: float | None = None,
        recency_weight: float = 0.0,
        recency_tau: float = 5.0,
        adaptive: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k sessions by hybrid semantic + emotional (+ recency) score.

        Parameters
        ----------
        user_id:          Only sessions belonging to this user are considered.
        query_embedding:  384-dim embedding of the current query message, or None.
                          When None (no message available), the semantic axis is
                          disabled and ranking uses the emotional and recency axes
                          only — this replaces the old hack of feeding the state
                          vector in as if it were a MiniLM embedding.
        state_vector:     384-dim emotional state vector from the transition model.
                          Only needed for the legacy state-cosine fallback.
        top_k:            Maximum number of results to return.
        alpha:            Weight for semantic score; (1-alpha) weights emotional.
                          Must be in [0, 1]. Semantic cosine is rescaled to [0, 1]
                          before blending so the two axes are commensurable.
        current_valence:  Decoded valence of the current state in [-1, 1].
        current_arousal:  Decoded arousal of the current state in [-1, 1].
                          When both are provided, the emotional axis uses
                          L2 distance in (valence, arousal) space, which is far
                          more discriminative than state-to-state cosine similarity
                          (cosine concentrates at ~0.98 regardless of emotional phase).
                          Falls back to cosine if not provided.
        recency_weight:   γ in [0, 1]. Final score = (1-γ)·hybrid + γ·recency where
                          recency = exp(-rank_age / recency_tau) and rank_age is how
                          many sessions ago the memory was stored (0 = newest).
                          Default 0.0 (off) preserves the original behaviour.
        recency_tau:      Decay constant for the recency term, in sessions.
        adaptive:         If True, alpha is adjusted per-query via adaptive_alpha():
                          emotionally homogeneous histories fall back to pure
                          semantic ranking; phase-shifted histories keep the blend.

        Returns
        -------
        List of result dicts, sorted by combined score descending. Each dict:
            session_text    : str
            session_id      : str
            semantic_score  : float   (rescaled to [0, 1])
            emotional_score : float
            recency_score   : float
            combined_score  : float
            effective_alpha : float   (post-adaptation alpha actually used)
            valence         : float
            arousal         : float
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(f"recency_weight must be in [0, 1], got {recency_weight}")

        result = self._col.get(
            where={"user_id": user_id},
            include=["embeddings", "documents", "metadatas"],
        )

        ids = result["ids"]
        if not ids:
            return []

        stored_embs = np.array(result["embeddings"], dtype=np.float32)  # (N, D)
        documents   = result["documents"]
        metadatas   = result["metadatas"]
        n = len(ids)

        # Semantic axis: current message embedding vs stored session embeddings
        # (MiniLM), rescaled from cosine [-1, 1] to [0, 1] so alpha blends two
        # commensurable quantities. Disabled (0.0) when no query is available.
        if query_embedding is not None:
            q_emb = _validate_embedding(query_embedding, "query_embedding")
            semantic_scores = (_cosine_sim_batch(q_emb, stored_embs) + 1.0) / 2.0
        else:
            semantic_scores = np.zeros(n, dtype=np.float64)

        va_stored = np.array(
            [[float(m["valence"]), float(m["arousal"])] for m in metadatas],
            dtype=np.float64,
        )

        # Emotional axis.
        # Preferred: L2 distance in (valence, arousal) space.
        #   State-to-state cosine similarity clusters at ~0.98 for all sessions
        #   regardless of emotional phase (all state vectors lie in a tight angular
        #   cone).  VAD-coordinate distance is far more discriminative: a burned-out
        #   session (v=-0.76) is clearly distant from a recovered state (v=+0.25).
        # Fallback: state-to-state cosine (used when VAD coords not provided).
        if current_valence is not None and current_arousal is not None:
            va_cur           = np.array([current_valence, current_arousal], dtype=np.float64)
            dists            = np.linalg.norm(va_stored - va_cur, axis=1)
            max_dist         = float(np.sqrt(8))   # max L2 in [-1, 1]^2
            emotional_scores = 1.0 - dists / max_dist
        elif state_vector is not None:
            s_vec = _validate_embedding(state_vector, "state_vector")
            stored_states = []
            for i, meta in enumerate(metadatas):
                if "state_vector" in meta:
                    stored_states.append(np.array(json.loads(meta["state_vector"]), dtype=np.float32))
                else:
                    stored_states.append(stored_embs[i])
            stored_states_arr = np.stack(stored_states)
            emotional_scores  = _cosine_sim_batch(s_vec, stored_states_arr)
        else:
            emotional_scores = np.zeros(n, dtype=np.float64)

        # Recency axis: exponential decay over storage order (newest = 1.0)
        timestamps = np.array([float(m["timestamp"]) for m in metadatas], dtype=np.float64)
        order      = np.argsort(np.argsort(-timestamps))    # rank_age: 0 = newest
        recency_scores = np.exp(-order / max(recency_tau, 1e-6))

        # Adaptive alpha: fall back to semantic-only when the emotional axis is
        # uninformative for this user's history.
        effective_alpha = alpha
        if adaptive:
            effective_alpha = adaptive_alpha(
                alpha, va_stored, (current_valence, current_arousal)
            )
        if query_embedding is None:
            effective_alpha = 0.0   # no semantic signal to weight

        hybrid          = effective_alpha * semantic_scores + (1.0 - effective_alpha) * emotional_scores
        combined_scores = (1.0 - recency_weight) * hybrid + recency_weight * recency_scores

        k = min(top_k, len(ids))
        top_idx = np.argsort(-combined_scores)[:k]

        return [
            {
                "session_text":    documents[i],
                "session_id":      metadatas[i]["session_id"],
                "semantic_score":  float(semantic_scores[i]),
                "emotional_score": float(emotional_scores[i]),
                "recency_score":   float(recency_scores[i]),
                "combined_score":  float(combined_scores[i]),
                "effective_alpha": float(effective_alpha),
                "valence":         float(metadatas[i]["valence"]),
                "arousal":         float(metadatas[i]["arousal"]),
            }
            for i in top_idx
        ]

    def get_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Return all stored sessions for *user_id*, ordered by timestamp."""
        result = self._col.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
        )
        sessions = [
            {
                "session_id":   meta["session_id"],
                "session_text": doc,
                "valence":      float(meta["valence"]),
                "arousal":      float(meta["arousal"]),
                "timestamp":    float(meta["timestamp"]),
            }
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        sessions.sort(key=lambda x: x["timestamp"])
        return sessions

    def delete_user(self, user_id: str) -> int:
        """Delete all sessions for *user_id*. Returns the number deleted."""
        result = self._col.get(
            where={"user_id": user_id},
            include=[],
        )
        ids = result["ids"]
        if not ids:
            return 0
        self._col.delete(ids=ids)
        return len(ids)

    def __repr__(self) -> str:
        return (
            f"MemoryStore(collection={self._collection_name!r}, "
            f"persist_dir={str(self._persist_dir)!r})"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    def print_results(results: list[dict], header: str) -> None:
        print(f"\n  {header}")
        for r in results[:3]:
            print(
                f"    #{results.index(r)+1}  [{r['session_id']}]  "
                f"sem={r['semantic_score']:+.3f}  "
                f"emo={r['emotional_score']:+.3f}  "
                f"comb={r['combined_score']:+.3f}  "
                f"v={r['valence']:+.2f}  "
                f'"{r["session_text"][:55]}"'
            )

    print("\n=== MemoryStore self-test ===\n")
    rng = np.random.default_rng(42)

    # Use a temp directory; explicitly reset the ChromaDB client before cleanup
    # to release file locks on Windows (avoids PermissionError on rmtree).
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_obj.name
    try:
        store = MemoryStore(collection_name="test_kokoro", persist_dir=tmpdir)
        print(f"  Store: {store}\n")

        # ------------------------------------------------------------------ #
        # Build two embedding clusters.                                       #
        #                                                                     #
        # Cluster A ("work / stress"):                                        #
        #   - Semantic query will prefer these (query ~ base_A).              #
        #   - Negative valence: -0.60 to -0.40.                               #
        #                                                                     #
        # Cluster B ("family / joy"):                                         #
        #   - Emotional (state) query will prefer these (state ~ base_B).     #
        #   - Positive valence: +0.55 to +0.75.                               #
        #                                                                     #
        # Each cluster has a unit base direction; sessions within a cluster   #
        # are that direction plus small Gaussian noise, then re-normalized.   #
        # ------------------------------------------------------------------ #

        D = _EMBED_DIM

        # Two orthogonal unit directions
        base_A = rng.standard_normal(D).astype(np.float32)
        base_A /= np.linalg.norm(base_A)

        base_B_raw = rng.standard_normal(D).astype(np.float32)
        base_B_raw -= base_B_raw.dot(base_A) * base_A   # Gram-Schmidt
        base_B = base_B_raw / np.linalg.norm(base_B_raw)

        cluster_A_texts = [
            "Feeling overwhelmed by deadlines at work, struggling to keep up.",
            "Stressful meeting with my manager today, feel undervalued.",
            "Work project is falling apart; anxiety is through the roof.",
            "Exhausted after another brutal week of late nights at the office.",
            "Conflict with a colleague has been draining all my energy.",
        ]
        cluster_B_texts = [
            "Had a wonderful afternoon with my kids in the park today.",
            "Went hiking with friends, felt truly alive and grateful.",
            "My partner surprised me with dinner — feeling so loved.",
            "Reconnected with old friends today, feeling full of joy.",
            "Celebrated a small win with loved ones, so much to be thankful for.",
        ]

        def make_cluster_emb(base: np.ndarray, noise_scale: float = 0.15) -> np.ndarray:
            v = base + rng.standard_normal(D).astype(np.float32) * noise_scale
            return v / np.linalg.norm(v)

        # Store Cluster A (work / stress)
        # State vectors for cluster A are near base_A (same direction as session embeddings
        # in this test — in production they come from the transition model, not MiniLM)
        for i, text in enumerate(cluster_A_texts):
            valence = -0.60 + i * 0.05   # -0.60 .. -0.40
            store.add_session(
                user_id="alice",
                session_id=f"work_{i}",
                session_text=text,
                session_embedding=make_cluster_emb(base_A),
                state_vector=make_cluster_emb(base_A),
                valence=valence,
                arousal=0.5,
            )

        # Store Cluster B (family / joy)
        for i, text in enumerate(cluster_B_texts):
            valence = 0.55 + i * 0.05    # +0.55 .. +0.75
            store.add_session(
                user_id="alice",
                session_id=f"joy_{i}",
                session_text=text,
                session_embedding=make_cluster_emb(base_B),
                state_vector=make_cluster_emb(base_B),
                valence=valence,
                arousal=0.3,
            )

        sessions = store.get_user_sessions("alice")
        print("--- Basic storage ---")
        check("10 sessions stored", len(sessions) == 10, f"got {len(sessions)}")
        check("sessions are dicts", all(isinstance(s, dict) for s in sessions))
        check("session_id present",  all("session_id"   in s for s in sessions))
        check("session_text present", all("session_text" in s for s in sessions))
        check("valence present",      all("valence"      in s for s in sessions))
        check("arousal present",      all("arousal"      in s for s in sessions))

        # ------------------------------------------------------------------ #
        # Semantic retrieval (alpha=1.0): query near base_A.                 #
        # Should return work_* sessions.                                      #
        # ------------------------------------------------------------------ #
        print("\n--- Semantic retrieval (alpha=1.0) ---")
        query_sem = make_cluster_emb(base_A, noise_scale=0.05)
        state_neutral = rng.standard_normal(D).astype(np.float32)
        state_neutral /= np.linalg.norm(state_neutral)

        res_sem = store.retrieve(
            "alice", query_sem, state_neutral, top_k=5, alpha=1.0
        )
        print_results(res_sem, "Top 3 (purely semantic, alpha=1.0):")
        check("semantic returns 5 results",
              len(res_sem) == 5, f"got {len(res_sem)}")
        check("top semantic result is cluster A (work_*)",
              res_sem[0]["session_id"].startswith("work_"),
              f"got {res_sem[0]['session_id']}")
        check("semantic: combined == semantic (alpha=1.0)",
              all(abs(r["combined_score"] - r["semantic_score"]) < 1e-5
                  for r in res_sem))

        sem_top5_ids = [r["session_id"] for r in res_sem]

        # ------------------------------------------------------------------ #
        # Emotional retrieval (alpha=0.0): state near base_B.                #
        # Should return joy_* sessions.                                       #
        # ------------------------------------------------------------------ #
        print("\n--- Emotional retrieval (alpha=0.0) ---")
        query_neutral = rng.standard_normal(D).astype(np.float32)
        query_neutral /= np.linalg.norm(query_neutral)
        state_emo = make_cluster_emb(base_B, noise_scale=0.05)

        res_emo = store.retrieve(
            "alice", query_neutral, state_emo, top_k=5, alpha=0.0
        )
        print_results(res_emo, "Top 3 (purely emotional, alpha=0.0):")
        check("emotional returns 5 results",
              len(res_emo) == 5, f"got {len(res_emo)}")
        check("top emotional result is cluster B (joy_*)",
              res_emo[0]["session_id"].startswith("joy_"),
              f"got {res_emo[0]['session_id']}")
        check("emotional: combined == emotional (alpha=0.0)",
              all(abs(r["combined_score"] - r["emotional_score"]) < 1e-5
                  for r in res_emo))

        emo_top5_ids = [r["session_id"] for r in res_emo]

        # ------------------------------------------------------------------ #
        # Rankings must differ — proves emotional axis is meaningful.        #
        # ------------------------------------------------------------------ #
        print("\n--- Rankings differ (alpha=1.0 vs alpha=0.0) ---")
        print(f"  Semantic top-5:  {sem_top5_ids}")
        print(f"  Emotional top-5: {emo_top5_ids}")
        check("top-1 differs between semantic and emotional",
              sem_top5_ids[0] != emo_top5_ids[0],
              f"both returned {sem_top5_ids[0]}")
        check("top-5 lists differ",
              sem_top5_ids != emo_top5_ids,
              "lists are identical — emotional retrieval is not working")

        # ------------------------------------------------------------------ #
        # Hybrid retrieval (alpha=0.6)                                        #
        # ------------------------------------------------------------------ #
        print("\n--- Hybrid retrieval (alpha=0.6) ---")
        res_hyb = store.retrieve(
            "alice", query_sem, state_emo, top_k=5, alpha=0.6
        )
        print_results(res_hyb, "Top 3 (hybrid, alpha=0.6):")
        check("hybrid returns 5 results",
              len(res_hyb) == 5, f"got {len(res_hyb)}")
        check("hybrid combined = 0.6*sem + 0.4*emo",
              all(
                  abs(r["combined_score"]
                      - (0.6 * r["semantic_score"] + 0.4 * r["emotional_score"])) < 1e-5
                  for r in res_hyb
              ))

        # ------------------------------------------------------------------ #
        # User isolation                                                       #
        # ------------------------------------------------------------------ #
        print("\n--- User isolation ---")
        store.add_session(
            user_id="bob",
            session_id="bob_0",
            session_text="Bob's only session.",
            session_embedding=make_cluster_emb(base_A),
            state_vector=make_cluster_emb(base_A),
            valence=0.1,
            arousal=0.0,
        )

        alice_q = rng.standard_normal(D).astype(np.float32)
        alice_q /= np.linalg.norm(alice_q)
        alice_s = rng.standard_normal(D).astype(np.float32)
        alice_s /= np.linalg.norm(alice_s)

        res_alice = store.retrieve("alice", alice_q, alice_s, top_k=10)
        res_bob   = store.retrieve("bob",   alice_q, alice_s, top_k=10)

        check("alice retrieval returns only alice sessions",
              all(not r["session_id"].startswith("bob_") for r in res_alice),
              f"found bob session in alice results")
        check("bob retrieval returns only bob sessions",
              all(r["session_id"].startswith("bob_") for r in res_bob),
              f"found alice session in bob results")
        check("alice count unchanged at 10",
              len(res_alice) == 10, f"got {len(res_alice)}")
        check("bob count is 1",
              len(res_bob) == 1, f"got {len(res_bob)}")

        # ------------------------------------------------------------------ #
        # Result structure                                                    #
        # ------------------------------------------------------------------ #
        print("\n--- Result structure ---")
        required_keys = {
            "session_text", "session_id",
            "semantic_score", "emotional_score", "combined_score",
            "valence", "arousal",
        }
        sample = res_sem[0]
        check("all required keys present",
              required_keys.issubset(sample.keys()),
              f"missing: {required_keys - sample.keys()}")
        check("session_text is str",      isinstance(sample["session_text"],   str))
        check("semantic_score is float",  isinstance(sample["semantic_score"], float))
        check("emotional_score is float", isinstance(sample["emotional_score"], float))
        check("combined_score is float",  isinstance(sample["combined_score"], float))
        check("valence is float",         isinstance(sample["valence"],        float))
        check("arousal is float",         isinstance(sample["arousal"],        float))

        # ------------------------------------------------------------------ #
        # Empty user                                                          #
        # ------------------------------------------------------------------ #
        print("\n--- Empty user ---")
        q_tmp = rng.standard_normal(D).astype(np.float32)
        q_tmp /= np.linalg.norm(q_tmp)
        s_tmp = rng.standard_normal(D).astype(np.float32)
        s_tmp /= np.linalg.norm(s_tmp)

        res_empty = store.retrieve("nobody", q_tmp, s_tmp)
        check("empty user returns []", res_empty == [], f"got {res_empty}")

        sessions_empty = store.get_user_sessions("nobody")
        check("get_user_sessions([]) for unknown user",
              sessions_empty == [], f"got {sessions_empty}")

        # ------------------------------------------------------------------ #
        # delete_user                                                         #
        # ------------------------------------------------------------------ #
        print("\n--- delete_user ---")
        n_deleted = store.delete_user("bob")
        check("delete_user returns count (1)", n_deleted == 1, f"got {n_deleted}")

        res_bob_after = store.retrieve("bob", alice_q, alice_s)
        check("bob sessions gone after delete", res_bob_after == [])

        n_deleted_again = store.delete_user("bob")
        check("delete_user returns 0 for unknown user",
              n_deleted_again == 0, f"got {n_deleted_again}")

        res_alice_after = store.retrieve("alice", alice_q, alice_s, top_k=10)
        check("alice sessions intact after bob deleted",
              len(res_alice_after) == 10, f"got {len(res_alice_after)}")

        # ------------------------------------------------------------------ #
        # Input validation                                                    #
        # ------------------------------------------------------------------ #
        print("\n--- Input validation ---")
        try:
            store.retrieve("alice",
                           np.zeros((2, D), dtype=np.float32),
                           np.zeros(D, dtype=np.float32))
            check("2-D query_embedding raises ValueError", False, "no error")
        except ValueError:
            check("2-D query_embedding raises ValueError", True)

        try:
            store.retrieve("alice",
                           np.zeros(D, dtype=np.float32),
                           np.zeros((2, D), dtype=np.float32))
            check("2-D state_vector raises ValueError", False, "no error")
        except ValueError:
            check("2-D state_vector raises ValueError", True)

        try:
            store.retrieve("alice",
                           np.zeros(D, dtype=np.float32),
                           np.zeros(D, dtype=np.float32),
                           alpha=1.5)
            check("alpha > 1.0 raises ValueError", False, "no error")
        except ValueError:
            check("alpha > 1.0 raises ValueError", True)

    finally:
        # Reset the ChromaDB client to release file locks before cleanup
        # (required on Windows; PersistentClient holds open file handles).
        try:
            store._client.reset()
        except Exception:
            pass
        del store
        try:
            tmpdir_obj.cleanup()
        except Exception:
            pass  # Windows may still hold locks; non-fatal for the test run

    # ---------------------------------------------------------------------- #
    # Summary                                                                 #
    # ---------------------------------------------------------------------- #
    total = passed + failed
    print(f"\n{'='*44}")
    print(f"  {passed}/{total} tests passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print("  -- all clear")
    print(f"{'='*44}\n")
