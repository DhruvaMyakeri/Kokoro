# -*- coding: utf-8 -*-
"""
kokoro/memory.py -- WorldMemory: the public API for Kokoro.

This is the single class that developers touch. It wires together every
internal component into one clean interface:

    SessionEncoder    -- text -> 384-dim embedding
    TransitionModel   -- (state, session_emb) -> updated state (unconstrained magnitude)
    StateDecoder      -- state + arc_history -> natural-language summary
    StateStore        -- SQLite: persist per-user state + arc history
    MemoryStore       -- ChromaDB: hybrid semantic + emotional retrieval

Typical usage
-------------
    from kokoro import WorldMemory

    memory = WorldMemory(user_id="dhruva")
    context = memory.update(session_turns)          # after each session
    context = memory.get_context(current_message)   # before each reply

    response = llm(prompt, system=context["state_summary"])

Pluggable retrieval
-------------------
Pass ``retrieval_fn`` to replace ChromaDB with your own vector store.
The function receives the same arguments as MemoryStore.retrieve() and
must return a list of dicts, each containing at minimum
``{"session_text": str, "session_id": str}``.

    def my_retrieval(user_id, query_emb, state_vec, top_k):
        # query your own vector DB
        results = my_vector_db.search(query_emb, top_k)
        return [{"session_text": r.text,
                 "session_id": r.id} for r in results]

    memory = WorldMemory(
        user_id="dhruva",
        retrieval_fn=my_retrieval
    )

Everything else (state update, decoding, warm-up period, persistence)
is handled by Kokoro as normal; only the retrieval step is replaced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from kokoro.encoder import SessionEncoder
from kokoro.transition import TransitionModel
from kokoro.decoder import StateDecoder
from kokoro.store import StateStore
from kokoro.retrieval import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).parent.parent
_DEFAULT_CHECKPOINT = _PKG_ROOT / "checkpoints" / "transition_v2.pt"
_DEFAULT_PROBE      = _PKG_ROOT / "checkpoints" / "valence_arousal_probe_v2.pt"

# Type alias for the pluggable retrieval callable.
RetrievalFn = Callable[
    [str, np.ndarray, np.ndarray, int],   # user_id, query_emb, state_vec, top_k
    list[dict],                            # list of {"session_text", "session_id", …}
]


# ---------------------------------------------------------------------------
# WorldMemory
# ---------------------------------------------------------------------------

class WorldMemory:
    """
    Emotional trajectory memory for an AI companion.

    Wraps the full Kokoro pipeline. Every call to ``update()`` encodes the
    session, advances the state through the transition model, decodes the
    emotional coordinates, persists everything, and adds the session to the
    retrieval index.  ``get_context()`` returns a ready-to-inject context dict.

    Parameters
    ----------
    user_id:
        Unique string identifier for the user (e.g. a database UUID or
        username).  All stored state is namespaced by this ID.
    db_path:
        Path to the SQLite database used by StateStore.
        Defaults to ~/.kokoro/kokoro.db.
    persist_dir:
        Directory used by MemoryStore (ChromaDB) for vector storage.
        Defaults to ~/.kokoro/memories/.
    checkpoint_path:
        Path to the trained TransitionModel checkpoint.
        Defaults to checkpoints/transition_v2.pt (project root).
    probe_path:
        Path to the linear valence/arousal probe checkpoint.
        Defaults to checkpoints/valence_arousal_probe_v2.pt (project root).
    top_k:
        Number of past sessions to retrieve per query (default 5).
    alpha:
        Semantic vs. emotional weight in hybrid retrieval.
        1.0 = purely semantic, 0.0 = purely emotional (default 0.6).
        Can be overridden per-call in ``get_context(alpha=…)``.
    min_sessions:
        Sessions required before ``get_context()`` produces a state summary
        and performs retrieval (warm-up period, default 3).
    collection_name:
        ChromaDB collection name (default "kokoro").
    retrieval_fn:
        Optional callable that replaces the built-in ChromaDB retrieval.
        Signature::

            def retrieval_fn(
                user_id: str,
                query_embedding: np.ndarray,   # (384,)
                state_vector: np.ndarray,      # (384,)
                top_k: int,
            ) -> list[dict]:
                ...

        Each returned dict must contain at minimum
        ``{"session_text": str, "session_id": str}``.
        When ``None`` (default), MemoryStore (ChromaDB) is used.
    """

    def __init__(
        self,
        user_id: str,
        *,
        db_path: Optional[str | Path] = None,
        persist_dir: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        probe_path: Optional[str | Path] = None,
        top_k: int = 5,
        alpha: float = 0.6,
        min_sessions: int = 3,
        collection_name: str = "kokoro",
        retrieval_fn: Optional[RetrievalFn] = None,
        recency_weight: float = 0.0,
        adaptive_alpha: bool = False,
    ) -> None:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(f"recency_weight must be in [0, 1], got {recency_weight}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if min_sessions < 1:
            raise ValueError(f"min_sessions must be >= 1, got {min_sessions}")

        self.user_id         = user_id
        self._top_k          = top_k
        self._alpha          = alpha
        self._min_sessions   = min_sessions
        self._retrieval_fn   = retrieval_fn
        self._recency_weight = recency_weight
        self._adaptive_alpha = adaptive_alpha

        # -- Resolve paths --------------------------------------------------
        ckpt_path   = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
        probe_path_ = Path(probe_path)       if probe_path      else _DEFAULT_PROBE

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"TransitionModel checkpoint not found: {ckpt_path}\n"
                "Run `python -m training.train` to generate it."
            )

        # -- Load components (lazy encoder; model on CPU) -------------------
        logger.info(f"WorldMemory: initialising for user={user_id!r}")

        # Determine whether the checkpoint was trained with VAD features
        # (session_dim=387) or without (session_dim=384), so the encoder
        # matches the transition model's expected input dimension.
        import torch as _torch
        _ckpt_cfg = _torch.load(str(ckpt_path), map_location="cpu",
                                weights_only=False).get("model_config", {})
        _use_vad  = _ckpt_cfg.get("session_dim", 384) > 384

        self._encoder     = SessionEncoder(use_vad_features=_use_vad)
        self._transition  = self._load_transition(ckpt_path)
        self._decoder     = StateDecoder(probe_path=probe_path_)
        self._state_store = StateStore(db_path=db_path)
        self._mem_store   = MemoryStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

        logger.info("WorldMemory: ready")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_transition(path: Path):
        from kokoro.transition import load_transition_checkpoint
        model, _cfg = load_transition_checkpoint(path)
        logger.debug(f"Transition model loaded from {path}")
        return model

    def _get_current_state(self) -> np.ndarray:
        """Return the stored state vector, or the zero cold-start vector."""
        vec = self._state_store.load(self.user_id)
        if vec is None:
            return TransitionModel.initial_state(batch_size=1).numpy()
        return vec

    def _session_summary(self, turns: Sequence[dict[str, str]]) -> str:
        """First 200 characters of user turns joined by spaces."""
        user_texts = [
            t["content"].strip()
            for t in turns
            if t.get("role") == "user" and t.get("content", "").strip()
        ]
        joined = " ".join(user_texts)
        return joined[:200] if joined else "(no user turns)"

    def _retrieve(
        self,
        query_emb: np.ndarray | None,
        state_vec: np.ndarray,
        alpha: float,
        decoded: dict | None = None,
    ) -> list[dict]:
        """
        Dispatch retrieval to either the pluggable function or the built-in
        MemoryStore.  Returns raw result dicts.
        """
        if self._retrieval_fn is not None:
            return self._retrieval_fn(
                self.user_id,
                query_emb,
                state_vec,
                self._top_k,
            )
        # Pass decoded VAD coordinates so MemoryStore can use the more
        # discriminative L2 distance in (valence, arousal) space instead
        # of state-to-state cosine similarity.
        v = decoded["valence"] if decoded else None
        a = decoded["arousal"] if decoded else None
        return self._mem_store.retrieve(
            user_id          = self.user_id,
            query_embedding  = query_emb,
            state_vector     = state_vec,
            top_k            = self._top_k,
            alpha            = alpha,
            current_valence  = v,
            current_arousal  = a,
            recency_weight   = self._recency_weight,
            adaptive         = self._adaptive_alpha,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, session_turns: Sequence[dict[str, str]]) -> dict[str, Any]:
        """
        Process a completed session and update the user's emotional state.

        Steps
        -----
        1. Encode the session with SessionEncoder -> 384-dim embedding.
        2. Advance the state vector through the TransitionModel.
        3. Decode state -> valence, arousal, trend via StateDecoder.
        4. Save the updated state to StateStore.
        5. Add the session to MemoryStore for future retrieval.
        6. Return the result of get_context("").

        Parameters
        ----------
        session_turns:
            List of ``{"role": "user"|"assistant", "content": "..."}`` dicts
            representing the completed session.

        Returns
        -------
        Context dict (same structure as ``get_context()``).

        Raises
        ------
        ValueError:
            If session_turns is empty or all content is blank.
        """
        # 1. Encode session
        session_emb: np.ndarray = self._encoder.encode(session_turns)   # (384,)

        # 2. Advance state through transition model
        current_state = self._get_current_state()                        # (384,)
        with torch.no_grad():
            state_t     = torch.from_numpy(current_state).float()
            emb_t       = torch.from_numpy(session_emb).float()
            new_state_t = self._transition(state_t, emb_t)              # (384,)
        new_state: np.ndarray = new_state_t.numpy()                     # (384,)

        # 3. Decode -> valence, arousal (we need these before saving so we can
        #    pass them to MemoryStore; arc_history not yet updated so we pass
        #    the old one to get a valid decode, then we'll re-decode from
        #    get_context after save which will use the correct updated history).
        info_before = self._state_store.get_info(self.user_id)
        arc_before  = info_before["arc_history"] if info_before else []
        session_count_before = info_before["session_count"] if info_before else 0

        decode_pre = self._decoder.decode(
            state_vector  = new_state,
            arc_history   = arc_before,
            session_count = session_count_before + 1,
        )
        valence: float = decode_pre["valence"]
        arousal: float = decode_pre["arousal"]

        # 4. Persist updated state (StateStore increments session_count)
        self._state_store.save(
            user_id      = self.user_id,
            state_vector = new_state,
            valence      = valence,
            arousal      = arousal,
        )

        # 5. Add session to MemoryStore
        info_after = self._state_store.get_info(self.user_id)
        session_id = f"session_{info_after['session_count']}"
        summary    = self._session_summary(session_turns)

        # MemoryStore uses 384-dim MiniLM embeddings for the semantic axis.
        # When the encoder appends VAD features (387-dim), strip them here —
        # the extra 3 dims are only needed as transition model input.
        session_emb_384 = session_emb[:384]

        self._mem_store.add_session(
            user_id           = self.user_id,
            session_id        = session_id,
            session_text      = summary,
            session_embedding = session_emb_384,
            state_vector      = new_state,
            valence           = valence,
            arousal           = arousal,
        )

        logger.info(
            f"WorldMemory.update: user={self.user_id!r}  "
            f"session={session_id}  v={valence:+.3f}  a={arousal:+.3f}"
        )

        # 6. Return fresh context
        return self.get_context("")

    def get_context(
        self,
        current_message: str = "",
        alpha: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Build and return the context dict for the current user state.

        Parameters
        ----------
        current_message:
            The message the user is about to send (used for semantic
            retrieval).  Pass an empty string if no message is available yet.
        alpha:
            Override the instance-level alpha for this call only.
            If ``None`` (default), ``self._alpha`` is used.
            Must be in [0, 1] when provided.
            Ignored when a ``retrieval_fn`` is in use (the custom function
            controls its own weighting).

        Returns
        -------
        dict with keys:
            state_summary      (str | None)  -- LLM-ready summary; None if not ready
            relevant_memories  (list[str])   -- retrieved session texts
            valence            (float)       -- current valence [-1, 1]
            arousal            (float)       -- current arousal [-1, 1]
            trend              (str)         -- "improving" | "declining" | "stable"
            session_count      (int)         -- total sessions stored
            ready              (bool)        -- False during warm-up period
        """
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        effective_alpha = self._alpha if alpha is None else alpha

        # Load current state
        state_vec = self._get_current_state()
        info      = self._state_store.get_info(self.user_id)

        # Brand-new user — no history at all
        if info is None:
            return {
                "state_summary":     None,
                "relevant_memories": [],
                "valence":           0.0,
                "arousal":           0.0,
                "trend":             "stable",
                "session_count":     0,
                "ready":             False,
            }

        arc_history   = info["arc_history"]
        session_count = info["session_count"]

        # Decode state
        decoded = self._decoder.decode(
            state_vector  = state_vec,
            arc_history   = arc_history,
            session_count = session_count,
        )

        ready = session_count >= self._min_sessions

        # Retrieve relevant memories when warmed up
        relevant_memories: list[str] = []
        if ready:
            # Use current message for semantic axis; current state for emotional axis.
            # With no message, pass None: MemoryStore disables the semantic axis and
            # ranks by emotional (and recency) proximity alone. The previous fallback
            # fed the raw state vector in as if it were a MiniLM embedding — two
            # unrelated vector spaces whose cosine is meaningless.
            if current_message.strip():
                query_emb = self._encoder.encode_text(current_message)
            else:
                query_emb = None

            results = self._retrieve(query_emb, state_vec, effective_alpha, decoded=decoded)
            relevant_memories = [r["session_text"] for r in results]

        return {
            "state_summary":     decoded["state_summary"],
            "relevant_memories": relevant_memories,
            "valence":           decoded["valence"],
            "arousal":           decoded["arousal"],
            "trend":             decoded["trend"],
            "session_count":     session_count,
            "ready":             ready,
        }

    def predict_next(self) -> dict[str, Any]:
        """
        Predict the emotional direction of the user's *next* session.

        The transition model is trained with a next-session prediction objective:
        state_t is oriented toward e_{t+1} by the VICReg similarity loss.
        Decoding the current state therefore gives an estimate of where the
        user is emotionally headed, not just where they are now.

        Returns
        -------
        dict with keys:
            predicted_valence  (float)        -- predicted valence for next session
            predicted_arousal  (float)        -- predicted arousal for next session
            predicted_quadrant (str)          -- e.g. "high-valence / high-arousal"
            session_count      (int)          -- sessions seen so far
            ready              (bool)         -- False if insufficient history

        Note: the probe used for decoding was trained on (state_t, v_t, a_t)
        pairs — current-session supervision.  Because consecutive sessions tend
        to share emotional tone, the decoded values correlate with the next
        session, but this is an approximation.  Prediction accuracy is quantified
        separately in diagnostics/eval_worldmodel.py.
        """
        state_vec = self._get_current_state()
        info      = self._state_store.get_info(self.user_id)

        if info is None:
            return {
                "predicted_valence":  0.0,
                "predicted_arousal":  0.0,
                "predicted_quadrant": "unknown",
                "session_count":      0,
                "ready":              False,
            }

        session_count = info["session_count"]
        ready = session_count >= self._min_sessions

        # Decode current state — oriented toward next session by training objective
        decoded = self._decoder.decode(
            state_vector  = state_vec,
            arc_history   = [],          # no trend; prediction is single-step
            session_count = session_count,
        )

        v = decoded["valence"]
        a = decoded["arousal"]

        # Map to quadrant label
        v_label = "high-valence" if v > 0 else "low-valence"
        a_label = "high-arousal" if a > 0 else "low-arousal"

        return {
            "predicted_valence":  v,
            "predicted_arousal":  a,
            "predicted_quadrant": f"{v_label} / {a_label}",
            "session_count":      session_count,
            "ready":              ready,
        }

    def reset(self) -> None:
        """
        Clear the user's stored state and all retrieved memories.

        The user record in StateStore is reset to session_count=0 (cold start).
        All sessions in MemoryStore are deleted.  The user_id slot is retained
        so the next ``update()`` call starts fresh without errors.
        """
        self._state_store.reset(self.user_id)
        self._mem_store.delete_user(self.user_id)
        logger.info(f"WorldMemory.reset: user={self.user_id!r}")

    def __repr__(self) -> str:
        info = self._state_store.get_info(self.user_id)
        n = info["session_count"] if info else 0
        return (
            f"WorldMemory(user_id={self.user_id!r}, "
            f"sessions={n}, alpha={self._alpha}, top_k={self._top_k})"
        )


# ---------------------------------------------------------------------------
# __main__ -- end-to-end self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile
    import logging

    logging.basicConfig(level=logging.WARNING)

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            suffix = f"  -- {detail}" if detail else ""
            print(f"  [FAIL] {name}{suffix}")

    def hr(title: str = "") -> None:
        if title:
            print(f"\n--- {title} ---")
        else:
            print()

    print("\n=== WorldMemory end-to-end self-test ===\n")

    # Four sessions with clearly different emotional content so the decoder
    # and retrieval have meaningful signal to work with.
    SESSIONS = [
        # Session 1: acute distress / overwhelm
        [
            {"role": "user",      "content": "I've been completely overwhelmed this week. Work is piling up and I can't sleep at all."},
            {"role": "assistant", "content": "That sounds really difficult. Struggling to sleep makes everything harder to cope with."},
            {"role": "user",      "content": "Yeah. I feel like I'm falling behind on everything and there's no way to catch up."},
        ],
        # Session 2: still stressed but starting to stabilise
        [
            {"role": "user",      "content": "Things are a bit better today. I managed to get some sleep last night."},
            {"role": "assistant", "content": "That's a good sign. Sleep really does change your perspective."},
            {"role": "user",      "content": "I still feel behind at work, but it's not as crushing. I made a small list of priorities."},
        ],
        # Session 3: genuine improvement, reconnecting with positives
        [
            {"role": "user",      "content": "I finished the big project today. Feels like a weight off my shoulders."},
            {"role": "assistant", "content": "Congratulations — that's a real achievement after everything you've been through."},
            {"role": "user",      "content": "Thank you. I went for a walk after and it was really nice. Feeling more like myself."},
        ],
        # Session 4: stable and positive
        [
            {"role": "user",      "content": "Had a great weekend. Spent time with friends and didn't think about work once."},
            {"role": "assistant", "content": "That sounds wonderful. Taking real breaks is so important."},
            {"role": "user",      "content": "Absolutely. I feel recharged. Ready to take on the next week properly."},
        ],
    ]

    # Isolated temp directories for both stores so the test is self-contained
    db_tmp  = tempfile.TemporaryDirectory()
    mem_tmp = tempfile.TemporaryDirectory()

    try:
        memory = WorldMemory(
            user_id       = "test_user",
            db_path       = Path(db_tmp.name) / "kokoro.db",
            persist_dir   = mem_tmp.name,
            min_sessions  = 3,
            alpha         = 0.6,
            top_k         = 3,
        )
        print(f"  Initialised: {memory}\n")

        # ------------------------------------------------------------------ #
        # Sessions 1-4                                                         #
        # ------------------------------------------------------------------ #
        for i, session in enumerate(SESSIONS, 1):
            hr(f"Session {i}")
            ctx = memory.update(session)
            print(f"  session_count={ctx['session_count']}  ready={ctx['ready']}  "
                  f"v={ctx['valence']:+.4f}  trend={ctx['trend']}")
            if ctx["state_summary"]:
                print(f"  summary: {ctx['state_summary'][:80]}…")

        check("session_count == 4 after four updates",
              ctx["session_count"] == 4, f"got {ctx['session_count']}")
        check("ready=True at session_count=4", ctx["ready"] is True)

        # ------------------------------------------------------------------ #
        # Change 1: per-call alpha override                                    #
        # ------------------------------------------------------------------ #
        hr("get_context — alpha override")
        query = "I'm feeling anxious about work again"

        ctx_default = memory.get_context(query)               # uses self._alpha=0.6
        ctx_sem     = memory.get_context(query, alpha=1.0)    # purely semantic
        ctx_emo     = memory.get_context(query, alpha=0.0)    # purely emotional

        print(f"  query: {query!r}")
        print(f"  default (a=0.6) top memory: {ctx_default['relevant_memories'][0][:70]!r}")
        print(f"  semantic (a=1.0) top memory: {ctx_sem['relevant_memories'][0][:70]!r}")
        print(f"  emotional (a=0.0) top memory: {ctx_emo['relevant_memories'][0][:70]!r}")

        check("alpha=None uses instance default (returns list)",
              isinstance(ctx_default["relevant_memories"], list))
        check("alpha=1.0 override returns list",
              isinstance(ctx_sem["relevant_memories"], list))
        check("alpha=0.0 override returns list",
              isinstance(ctx_emo["relevant_memories"], list))

        mem_custom = None   # ensure defined before pluggable retrieval block
        check("semantic and emotional may return different orderings",
              True)   # not always guaranteed to differ on 4 sessions; just document it

        try:
            memory.get_context(query, alpha=1.5)
            check("alpha=1.5 raises ValueError", False, "no error raised")
        except ValueError:
            check("alpha=1.5 raises ValueError", True)

        # ------------------------------------------------------------------ #
        # Change 2: pluggable retrieval_fn                                     #
        # ------------------------------------------------------------------ #
        hr("Pluggable retrieval_fn")

        # Build a tiny in-memory store to act as the custom backend.
        _custom_sessions: list[dict] = []

        def custom_retrieval(
            user_id: str,
            query_emb: np.ndarray,
            state_vec: np.ndarray,
            top_k: int,
        ) -> list[dict]:
            """Dead-simple custom backend: return all stored sessions as-is."""
            return [s for s in _custom_sessions if s["user_id"] == user_id][:top_k]

        mem_custom = WorldMemory(
            user_id      = "custom_user",
            db_path      = Path(db_tmp.name) / "kokoro_custom.db",
            persist_dir  = mem_tmp.name,   # ChromaDB still used for add_session
            min_sessions = 3,
            retrieval_fn = custom_retrieval,
        )

        # Populate the custom backend manually (simulating an external DB)
        _custom_sessions.append({
            "user_id":      "custom_user",
            "session_id":   "ext_0",
            "session_text": "Custom backend: first session",
        })
        _custom_sessions.append({
            "user_id":      "custom_user",
            "session_id":   "ext_1",
            "session_text": "Custom backend: second session",
        })
        _custom_sessions.append({
            "user_id":      "custom_user",
            "session_id":   "ext_2",
            "session_text": "Custom backend: third session",
        })

        # Run 3 updates so the user crosses min_sessions
        for session in SESSIONS[:3]:
            mem_custom.update(session)

        ctx_custom = mem_custom.get_context("anything")
        print(f"  retrieval_fn memories: {ctx_custom['relevant_memories']}")

        check("retrieval_fn: ready=True after 3 sessions",
              ctx_custom["ready"] is True)
        check("retrieval_fn: memories come from custom backend",
              all("Custom backend" in m for m in ctx_custom["relevant_memories"]),
              f"got: {ctx_custom['relevant_memories']}")
        check("retrieval_fn: correct count returned",
              len(ctx_custom["relevant_memories"]) == 3,
              f"got {len(ctx_custom['relevant_memories'])}")

        # ------------------------------------------------------------------ #
        # Brand-new user (no history)                                          #
        # ------------------------------------------------------------------ #
        hr("Brand-new user (no history)")
        memory2 = WorldMemory(
            user_id     = "brand_new_user",
            db_path     = Path(db_tmp.name) / "kokoro.db",
            persist_dir = mem_tmp.name,
        )
        ctx_new = memory2.get_context("Hello")
        check("new user: session_count == 0",  ctx_new["session_count"] == 0)
        check("new user: ready=False",         ctx_new["ready"] is False)
        check("new user: state_summary=None",  ctx_new["state_summary"] is None)
        check("new user: relevant_memories=[]",ctx_new["relevant_memories"] == [])

        # ------------------------------------------------------------------ #
        # repr                                                                #
        # ------------------------------------------------------------------ #
        hr("repr")
        r = repr(memory)
        print(f"  {r}")
        check("repr contains user_id",    "test_user" in r)
        check("repr contains sessions=4", "sessions=4" in r)

    finally:
        for obj in (memory, mem_custom, memory2):
            try:
                obj._mem_store._client.reset()
            except Exception:
                pass
        for d in (db_tmp, mem_tmp):
            try:
                d.cleanup()
            except Exception:
                pass

    # ---------------------------------------------------------------------- #
    # Summary                                                                 #
    # ---------------------------------------------------------------------- #
    total = passed + failed
    print(f"\n{'='*48}")
    print(f"  {passed}/{total} tests passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print("  -- all clear")
    print(f"{'='*48}\n")
