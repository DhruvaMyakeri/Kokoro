"""
Tests for kokoro/memory.py (WorldMemory — full pipeline)

Covers:
  - update() returns a context dict
  - get_context() returns correct keys
  - ready=False for first 2 sessions
  - ready=True after 3rd session
  - brand-new user returns all nulls / zeros
  - state_summary is non-empty string after 3 sessions
  - full end-to-end pipeline
"""

from pathlib import Path

import pytest

from kokoro.memory import WorldMemory

# ---------------------------------------------------------------------------
# Sample sessions
# ---------------------------------------------------------------------------

SESSIONS = [
    # Session 1: acute distress
    [
        {"role": "user",      "content": "I've been completely overwhelmed this week. Work is piling up and I can't sleep."},
        {"role": "assistant", "content": "That sounds really difficult. Struggling to sleep makes everything harder."},
        {"role": "user",      "content": "Yeah. I feel like I'm falling behind on everything."},
    ],
    # Session 2: stabilising
    [
        {"role": "user",      "content": "Things are a bit better today. I managed to get some sleep last night."},
        {"role": "assistant", "content": "That's a good sign. Sleep really does change your perspective."},
        {"role": "user",      "content": "I still feel behind at work, but it's not as crushing."},
    ],
    # Session 3: genuine improvement
    [
        {"role": "user",      "content": "I finished the big project today. Feels like a weight off my shoulders."},
        {"role": "assistant", "content": "Congratulations — that's a real achievement."},
        {"role": "user",      "content": "Thank you. I went for a walk after. Feeling more like myself."},
    ],
]

_CONTEXT_KEYS = {
    "state_summary", "relevant_memories",
    "valence", "arousal", "trend", "session_count", "ready",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wm(tmp_path):
    """
    Fresh WorldMemory for each test. Releases ChromaDB locks on teardown.
    """
    db_path     = tmp_path / "kokoro.db"
    persist_dir = tmp_path / "memories"
    m = WorldMemory(
        user_id        = "test_user",
        db_path        = db_path,
        persist_dir    = str(persist_dir),
        min_sessions   = 3,
        alpha          = 0.6,
        top_k          = 3,
    )
    yield m
    try:
        m._mem_store._client.reset()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# update() return dict
# ---------------------------------------------------------------------------

def test_update_returns_dict(wm):
    ctx = wm.update(SESSIONS[0])
    assert isinstance(ctx, dict)


def test_update_returns_all_required_keys(wm):
    ctx = wm.update(SESSIONS[0])
    assert _CONTEXT_KEYS.issubset(ctx.keys())


# ---------------------------------------------------------------------------
# get_context() keys
# ---------------------------------------------------------------------------

def test_get_context_returns_all_required_keys(wm):
    wm.update(SESSIONS[0])
    ctx = wm.get_context("")
    assert _CONTEXT_KEYS.issubset(ctx.keys())


def test_get_context_relevant_memories_is_list(wm):
    wm.update(SESSIONS[0])
    ctx = wm.get_context("")
    assert isinstance(ctx["relevant_memories"], list)


# ---------------------------------------------------------------------------
# ready flag
# ---------------------------------------------------------------------------

def test_ready_false_after_first_session(wm):
    ctx = wm.update(SESSIONS[0])
    assert ctx["ready"] is False


def test_ready_false_after_second_session(wm):
    wm.update(SESSIONS[0])
    ctx = wm.update(SESSIONS[1])
    assert ctx["ready"] is False


def test_ready_true_after_third_session(wm):
    wm.update(SESSIONS[0])
    wm.update(SESSIONS[1])
    ctx = wm.update(SESSIONS[2])
    assert ctx["ready"] is True


# ---------------------------------------------------------------------------
# state_summary
# ---------------------------------------------------------------------------

def test_state_summary_none_before_ready(wm):
    ctx = wm.update(SESSIONS[0])
    assert ctx["state_summary"] is None


def test_state_summary_non_empty_after_3_sessions(wm):
    wm.update(SESSIONS[0])
    wm.update(SESSIONS[1])
    ctx = wm.update(SESSIONS[2])
    assert ctx["state_summary"] is not None
    assert isinstance(ctx["state_summary"], str)
    assert len(ctx["state_summary"]) > 10


# ---------------------------------------------------------------------------
# session_count
# ---------------------------------------------------------------------------

def test_session_count_increments_with_each_update(wm):
    ctx1 = wm.update(SESSIONS[0])
    assert ctx1["session_count"] == 1

    ctx2 = wm.update(SESSIONS[1])
    assert ctx2["session_count"] == 2

    ctx3 = wm.update(SESSIONS[2])
    assert ctx3["session_count"] == 3


# ---------------------------------------------------------------------------
# Brand-new user (no history)
# ---------------------------------------------------------------------------

def test_brand_new_user_returns_all_nulls(tmp_path):
    db_path     = tmp_path / "new.db"
    persist_dir = tmp_path / "new_memories"
    m = WorldMemory(
        user_id     = "brand_new",
        db_path     = db_path,
        persist_dir = str(persist_dir),
    )
    try:
        ctx = m.get_context("Hello")
        assert ctx["state_summary"] is None
        assert ctx["relevant_memories"] == []
        assert ctx["session_count"] == 0
        assert ctx["ready"] is False
        assert ctx["valence"] == 0.0
        assert ctx["arousal"] == 0.0
        assert ctx["trend"] == "stable"
    finally:
        try:
            m._mem_store._client.reset()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_end_to_end(wm):
    """
    Run 3 updates then call get_context() with a real message.
    All fields should be populated and ready=True.
    """
    for session in SESSIONS:
        wm.update(session)

    ctx = wm.get_context("How am I doing emotionally?")

    assert ctx["ready"] is True
    assert isinstance(ctx["state_summary"], str) and len(ctx["state_summary"]) > 0
    assert isinstance(ctx["relevant_memories"], list)
    assert isinstance(ctx["valence"], float)
    assert isinstance(ctx["arousal"], float)
    assert ctx["trend"] in ("improving", "declining", "stable")
    assert ctx["session_count"] == 3


def test_rag_retrieves_semantically_relevant_memories(tmp_path):
    """
    Integration test: semantic retrieval surfaces content relevant to the query.

    Six sessions are stored in two clear thematic clusters:
      Sessions 1-3: work stress
      Sessions 4-6: family joy

    A work-flavoured query should retrieve work sessions; a family-flavoured
    query should retrieve family sessions.  The two result sets must differ,
    proving that retrieval is query-sensitive rather than returning the same
    memories regardless of context.
    """
    WORK_SESSIONS = [
        [{"role": "user", "content": "Had a terrible day at work, my manager criticized everything I did"}],
        [{"role": "user", "content": "Another stressful meeting, deadlines are impossible"}],
        [{"role": "user", "content": "Can't sleep because of work anxiety, overwhelmed"}],
    ]
    FAMILY_SESSIONS = [
        [{"role": "user", "content": "Wonderful evening with family, felt so loved"}],
        [{"role": "user", "content": "Kids did great at school, so proud of them"}],
        [{"role": "user", "content": "Amazing weekend trip with loved ones, grateful"}],
    ]

    db_path     = tmp_path / "rag_test.db"
    persist_dir = tmp_path / "rag_memories"
    m = WorldMemory(
        user_id      = "rag_user",
        db_path      = db_path,
        persist_dir  = str(persist_dir),
        min_sessions = 3,
        top_k        = 3,
        alpha        = 0.9,   # lean semantic so query wording drives retrieval
    )
    try:
        for session in WORK_SESSIONS + FAMILY_SESSIONS:
            m.update(session)

        assert m._state_store.get_info("rag_user")["session_count"] == 6

        ctx_work   = m.get_context("I'm stressed about work deadlines again")
        ctx_family = m.get_context("Spending time with family this weekend")

        # --- Query A: work ---
        assert ctx_work["relevant_memories"], "Expected non-empty memories for work query"
        work_keywords = {"work", "manager", "deadline", "stress", "meeting", "anxiety"}
        work_hits = [
            mem for mem in ctx_work["relevant_memories"]
            if any(kw in mem.lower() for kw in work_keywords)
        ]
        assert work_hits, (
            f"No work-related memory retrieved for work query.\n"
            f"Got: {ctx_work['relevant_memories']}"
        )

        # --- Query B: family ---
        assert ctx_family["relevant_memories"], "Expected non-empty memories for family query"
        family_keywords = {"family", "kids", "weekend", "loved", "grateful", "proud"}
        family_hits = [
            mem for mem in ctx_family["relevant_memories"]
            if any(kw in mem.lower() for kw in family_keywords)
        ]
        assert family_hits, (
            f"No family-related memory retrieved for family query.\n"
            f"Got: {ctx_family['relevant_memories']}"
        )

        # --- Retrieval is query-sensitive ---
        assert ctx_work["relevant_memories"] != ctx_family["relevant_memories"], (
            "Work query and family query returned identical memories — "
            "retrieval is not responding to query content."
        )

    finally:
        try:
            m._mem_store._client.reset()
        except Exception:
            pass
