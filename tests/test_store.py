"""
Tests for kokoro/store.py (StateStore)

Covers:
  - save/load round trip is lossless
  - session_count increments correctly
  - arc_history capped at 20
  - delete removes user
  - reset zeros state and clears history
  - two users do not contaminate each other
  - load returns None for unknown user
"""

import numpy as np
import pytest

from kokoro.store import StateStore

STATE_DIM = 384


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Fresh StateStore backed by a temp SQLite file."""
    return StateStore(tmp_path / "test.db")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _vec(rng, dim=STATE_DIM) -> np.ndarray:
    return rng.standard_normal(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip_is_lossless(store, rng):
    vec = _vec(rng)
    store.save("alice", vec, valence=0.5, arousal=0.1)
    loaded = store.load("alice")

    assert loaded is not None
    assert loaded.shape == (STATE_DIM,)
    assert loaded.dtype == np.float32
    assert np.allclose(vec, loaded, atol=1e-7), \
        f"Max diff: {np.abs(vec - loaded).max():.2e}"


# ---------------------------------------------------------------------------
# session_count
# ---------------------------------------------------------------------------

def test_session_count_starts_at_1_after_first_save(store, rng):
    store.save("alice", _vec(rng), valence=0.1, arousal=0.0)
    assert store.get_info("alice")["session_count"] == 1


def test_session_count_increments_on_each_save(store, rng):
    for i in range(5):
        store.save("alice", _vec(rng), valence=float(i) * 0.1, arousal=0.0)
    assert store.get_info("alice")["session_count"] == 5


# ---------------------------------------------------------------------------
# arc_history
# ---------------------------------------------------------------------------

def test_arc_history_capped_at_20(store, rng):
    for i in range(25):
        store.save("alice", _vec(rng), valence=float(i) * 0.01, arousal=0.0)
    info = store.get_info("alice")
    assert len(info["arc_history"]) == 20
    assert info["session_count"] == 25


def test_arc_history_entries_are_two_element_lists(store, rng):
    for i in range(5):
        store.save("alice", _vec(rng), valence=float(i) * 0.1, arousal=float(i) * 0.05)
    info = store.get_info("alice")
    for entry in info["arc_history"]:
        assert isinstance(entry, list)
        assert len(entry) == 2


def test_arc_history_oldest_first(store, rng):
    """
    First save → valence=0.9.  After 5 saves the first entry should be [0.9, …].
    """
    store.save("alice", _vec(rng), valence=0.9, arousal=0.1)
    for i in range(4):
        store.save("alice", _vec(rng), valence=float(i) * 0.1, arousal=0.0)
    info = store.get_info("alice")
    assert abs(info["arc_history"][0][0] - 0.9) < 1e-5


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_removes_user(store, rng):
    store.save("bob", _vec(rng), valence=0.1, arousal=0.0)
    result = store.delete("bob")
    assert result is True
    assert store.load("bob") is None
    assert store.get_info("bob") is None


def test_delete_returns_false_for_unknown_user(store):
    assert store.delete("nonexistent") is False


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_zeros_state_vector(store, rng):
    for _ in range(3):
        store.save("alice", _vec(rng), valence=0.5, arousal=0.2)
    store.reset("alice")
    loaded = store.load("alice")
    assert np.all(loaded == 0.0)


def test_reset_clears_session_count(store, rng):
    for _ in range(3):
        store.save("alice", _vec(rng), valence=0.5, arousal=0.2)
    store.reset("alice")
    assert store.get_info("alice")["session_count"] == 0


def test_reset_clears_arc_history(store, rng):
    for _ in range(3):
        store.save("alice", _vec(rng), valence=0.5, arousal=0.2)
    store.reset("alice")
    assert store.get_info("alice")["arc_history"] == []


def test_reset_zeros_valence_and_arousal(store, rng):
    store.save("alice", _vec(rng), valence=0.7, arousal=0.3)
    store.reset("alice")
    info = store.get_info("alice")
    assert info["valence"] == 0.0
    assert info["arousal"] == 0.0


def test_reset_user_still_exists_in_list(store, rng):
    store.save("alice", _vec(rng), valence=0.1, arousal=0.0)
    store.reset("alice")
    assert "alice" in store.list_users()


def test_reset_returns_false_for_unknown_user(store):
    assert store.reset("nobody") is False


# ---------------------------------------------------------------------------
# User isolation
# ---------------------------------------------------------------------------

def test_two_users_do_not_contaminate_each_other(store, rng):
    vec1 = _vec(rng)
    vec2 = _vec(rng)
    store.save("user1", vec1, valence=0.9,  arousal=0.5)
    store.save("user2", vec2, valence=-0.7, arousal=0.1)

    loaded1 = store.load("user1")
    loaded2 = store.load("user2")

    assert np.allclose(vec1, loaded1, atol=1e-7)
    assert np.allclose(vec2, loaded2, atol=1e-7)
    assert not np.allclose(loaded1, loaded2)


def test_two_users_independent_session_counts(store, rng):
    for _ in range(3):
        store.save("alice", _vec(rng), valence=0.1, arousal=0.0)
    for _ in range(7):
        store.save("bob",   _vec(rng), valence=0.2, arousal=0.0)

    assert store.get_info("alice")["session_count"] == 3
    assert store.get_info("bob")["session_count"] == 7


# ---------------------------------------------------------------------------
# Unknown user
# ---------------------------------------------------------------------------

def test_load_returns_none_for_unknown_user(store):
    assert store.load("unknown") is None


def test_get_info_returns_none_for_unknown_user(store):
    assert store.get_info("unknown") is None


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------

def test_list_users_contains_saved_users(store, rng):
    store.save("alice", _vec(rng), valence=0.1, arousal=0.0)
    store.save("bob",   _vec(rng), valence=0.2, arousal=0.0)
    users = store.list_users()
    assert "alice" in users
    assert "bob" in users


def test_list_users_empty_for_fresh_store(store):
    assert store.list_users() == []
