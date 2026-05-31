"""
Tests for kokoro/retrieval.py (MemoryStore)

Covers:
  - add_session + retrieve round trip
  - alpha=1.0 → combined_score equals semantic_score
  - alpha=0.0 → combined_score equals emotional_score
  - semantic and emotional rankings differ (orthogonal clusters)
  - user isolation
  - empty user returns []
"""

import numpy as np
import pytest

from kokoro.retrieval import MemoryStore

DIM = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, noise: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + rng.standard_normal(DIM).astype(np.float32) * noise
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Fresh MemoryStore backed by a temp ChromaDB directory."""
    s = MemoryStore(collection_name="test_kokoro", persist_dir=str(tmp_path))
    yield s
    # Release ChromaDB file locks before temp-dir cleanup (Windows)
    try:
        s._client.reset()
    except Exception:
        pass


@pytest.fixture
def orthogonal_bases():
    """
    Two orthogonal unit directions (base_A, base_B) for building clusters.
    base_A: semantic cluster (work / negative).
    base_B: emotional cluster (family / positive).
    """
    rng = np.random.default_rng(42)
    base_A = rng.standard_normal(DIM).astype(np.float32)
    base_A /= np.linalg.norm(base_A)

    base_B_raw = rng.standard_normal(DIM).astype(np.float32)
    base_B_raw -= base_B_raw.dot(base_A) * base_A   # Gram-Schmidt
    base_B = base_B_raw / np.linalg.norm(base_B_raw)

    return base_A, base_B


# ---------------------------------------------------------------------------
# Add + retrieve round trip
# ---------------------------------------------------------------------------

def test_add_and_retrieve_finds_added_session(store):
    emb = _rand_unit(0)
    store.add_session("alice", "s0", "My first session", emb, valence=0.5, arousal=0.1)
    results = store.retrieve("alice", emb, emb, top_k=1)
    assert len(results) == 1
    assert results[0]["session_text"] == "My first session"
    assert results[0]["session_id"] == "s0"


def test_retrieve_returns_required_keys(store):
    emb = _rand_unit(0)
    store.add_session("alice", "s0", "Session text", emb, valence=0.5, arousal=0.1)
    result = store.retrieve("alice", emb, emb, top_k=1)[0]
    required = {
        "session_text", "session_id",
        "semantic_score", "emotional_score", "combined_score",
        "valence", "arousal",
    }
    assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Empty user
# ---------------------------------------------------------------------------

def test_empty_user_returns_empty_list(store):
    results = store.retrieve("nobody", _rand_unit(0), _rand_unit(1))
    assert results == []


# ---------------------------------------------------------------------------
# alpha=1.0 → purely semantic
# ---------------------------------------------------------------------------

def test_alpha_1_combined_equals_semantic(store, orthogonal_bases):
    base_A, base_B = orthogonal_bases
    rng = np.random.default_rng(10)

    for i in range(4):
        v = base_A + rng.standard_normal(DIM).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        store.add_session("user", f"A_{i}", f"Cluster A {i}", v, valence=-0.5, arousal=0.0)
    for i in range(4):
        v = base_B + rng.standard_normal(DIM).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        store.add_session("user", f"B_{i}", f"Cluster B {i}", v, valence=0.5, arousal=0.0)

    query = _near(base_A, noise=0.05, seed=99)
    state = _near(base_B, noise=0.05, seed=100)  # state in opposite cluster

    res = store.retrieve("user", query, state, top_k=5, alpha=1.0)
    assert len(res) == 5
    for r in res:
        assert abs(r["combined_score"] - r["semantic_score"]) < 1e-5, \
            f"alpha=1.0: combined != semantic: {r['combined_score']:.6f} vs {r['semantic_score']:.6f}"


# ---------------------------------------------------------------------------
# alpha=0.0 → purely emotional
# ---------------------------------------------------------------------------

def test_alpha_0_combined_equals_emotional(store, orthogonal_bases):
    base_A, base_B = orthogonal_bases
    rng = np.random.default_rng(20)

    for i in range(4):
        v = base_A + rng.standard_normal(DIM).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        store.add_session("user", f"A_{i}", f"Cluster A {i}", v, valence=-0.5, arousal=0.0)
    for i in range(4):
        v = base_B + rng.standard_normal(DIM).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        store.add_session("user", f"B_{i}", f"Cluster B {i}", v, valence=0.5, arousal=0.0)

    query = _near(base_A, noise=0.05, seed=101)
    state = _near(base_B, noise=0.05, seed=102)

    res = store.retrieve("user", query, state, top_k=5, alpha=0.0)
    assert len(res) == 5
    for r in res:
        assert abs(r["combined_score"] - r["emotional_score"]) < 1e-5, \
            f"alpha=0.0: combined != emotional: {r['combined_score']:.6f} vs {r['emotional_score']:.6f}"


# ---------------------------------------------------------------------------
# Rankings differ between alpha=1.0 and alpha=0.0
# ---------------------------------------------------------------------------

def test_semantic_and_emotional_rankings_differ(store, orthogonal_bases):
    """
    With two orthogonal clusters, a query near A (alpha=1.0) should rank
    cluster-A sessions first, while a state near B (alpha=0.0) should rank
    cluster-B sessions first — the two top-5 lists must differ.
    """
    base_A, base_B = orthogonal_bases
    rng = np.random.default_rng(30)

    for i in range(5):
        v = base_A + rng.standard_normal(DIM).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        store.add_session("omega", f"A_{i}", f"A session {i}", v, valence=-0.5, arousal=0.0)
    for i in range(5):
        v = base_B + rng.standard_normal(DIM).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        store.add_session("omega", f"B_{i}", f"B session {i}", v, valence=0.5, arousal=0.0)

    query_near_A = _near(base_A, noise=0.05, seed=103)
    state_near_B = _near(base_B, noise=0.05, seed=104)

    res_sem = store.retrieve("omega", query_near_A, state_near_B, top_k=5, alpha=1.0)
    res_emo = store.retrieve("omega", query_near_A, state_near_B, top_k=5, alpha=0.0)

    ids_sem = [r["session_id"] for r in res_sem]
    ids_emo = [r["session_id"] for r in res_emo]

    assert ids_sem != ids_emo, \
        f"Semantic and emotional rankings are identical: {ids_sem}. " \
        "Emotional retrieval is not differentiating by state."


# ---------------------------------------------------------------------------
# User isolation
# ---------------------------------------------------------------------------

def test_user_isolation(store):
    emb_a = _rand_unit(10)
    emb_b = _rand_unit(11)
    store.add_session("alice", "a0", "Alice session", emb_a, valence=0.1, arousal=0.0)
    store.add_session("bob",   "b0", "Bob session",   emb_b, valence=0.2, arousal=0.0)

    query = _rand_unit(0)
    state = _rand_unit(1)

    res_alice = store.retrieve("alice", query, state, top_k=10)
    res_bob   = store.retrieve("bob",   query, state, top_k=10)

    alice_ids = [r["session_id"] for r in res_alice]
    bob_ids   = [r["session_id"] for r in res_bob]

    assert "a0" in alice_ids
    assert "b0" not in alice_ids
    assert "b0" in bob_ids
    assert "a0" not in bob_ids
