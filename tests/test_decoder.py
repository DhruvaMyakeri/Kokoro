"""
Tests for kokoro/decoder.py (StateDecoder)

Covers:
  - ready=False for session_count < 3
  - ready=True for session_count >= 3
  - state_summary is None when not ready
  - state_summary is non-empty string when ready
  - declining arc_history → trend="declining"
  - improving arc_history → trend="improving"
  - stable arc_history → trend="stable"
"""

import numpy as np
import pytest

from kokoro.decoder import StateDecoder

STATE_DIM = 384


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decoder():
    """Load the probe once for the whole module."""
    return StateDecoder()  # uses default checkpoint path


def _unit_vec(seed: int = 0, dim: int = STATE_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------

def test_is_ready_false_below_threshold(decoder):
    assert decoder.is_ready(1) is False
    assert decoder.is_ready(2) is False


def test_is_ready_true_at_threshold(decoder):
    assert decoder.is_ready(3) is True


def test_is_ready_true_above_threshold(decoder):
    assert decoder.is_ready(10) is True


# ---------------------------------------------------------------------------
# ready flag in decode()
# ---------------------------------------------------------------------------

def test_decode_ready_false_for_session_count_1(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=1)
    assert ctx["ready"] is False


def test_decode_ready_false_for_session_count_2(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=2)
    assert ctx["ready"] is False


def test_decode_ready_true_for_session_count_3(decoder):
    history = [[0.1, 0.0], [0.15, 0.0], [0.2, 0.0]]
    ctx = decoder.decode(_unit_vec(), arc_history=history, session_count=3)
    assert ctx["ready"] is True


def test_decode_ready_true_for_session_count_10(decoder):
    history = [[0.1, 0.0]] * 10
    ctx = decoder.decode(_unit_vec(), arc_history=history, session_count=10)
    assert ctx["ready"] is True


# ---------------------------------------------------------------------------
# state_summary
# ---------------------------------------------------------------------------

def test_state_summary_is_none_when_not_ready(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=2)
    assert ctx["state_summary"] is None


def test_state_summary_is_non_empty_string_when_ready(decoder):
    history = [[0.2, 0.0], [0.25, 0.0], [0.3, 0.0]]
    ctx = decoder.decode(_unit_vec(seed=1), arc_history=history, session_count=3)
    assert ctx["state_summary"] is not None
    assert isinstance(ctx["state_summary"], str)
    assert len(ctx["state_summary"]) > 10


# ---------------------------------------------------------------------------
# trend classification
# ---------------------------------------------------------------------------

def test_declining_arc_history_produces_declining_trend(decoder):
    """
    Valence drops from 0.6 to -0.12 over 10 steps (slope ≈ -0.08),
    well below the _SLOPE_DEC = -0.02 threshold.
    """
    n = 10
    history = [[0.6 - i * 0.08, 0.0] for i in range(n)]
    ctx = decoder.decode(_unit_vec(seed=2), arc_history=history, session_count=n)
    assert ctx["trend"] == "declining", \
        f"Expected 'declining', got '{ctx['trend']}' (strength={ctx['trend_strength']:.5f})"


def test_improving_arc_history_produces_improving_trend(decoder):
    """
    Valence rises from -0.5 to +0.22 over 10 steps (slope ≈ +0.08),
    well above the _SLOPE_IMP = +0.02 threshold.
    """
    n = 10
    history = [[-0.5 + i * 0.08, 0.0] for i in range(n)]
    ctx = decoder.decode(_unit_vec(seed=3), arc_history=history, session_count=n)
    assert ctx["trend"] == "improving", \
        f"Expected 'improving', got '{ctx['trend']}' (strength={ctx['trend_strength']:.5f})"


def test_stable_arc_history_produces_stable_trend(decoder):
    """Flat arc history (slope == 0) should be classified as stable."""
    history = [[0.3, 0.0]] * 10
    ctx = decoder.decode(_unit_vec(seed=4), arc_history=history, session_count=10)
    assert ctx["trend"] == "stable", \
        f"Expected 'stable', got '{ctx['trend']}'"


# ---------------------------------------------------------------------------
# Return dict structure
# ---------------------------------------------------------------------------

def test_decode_returns_all_required_keys(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=1)
    required = {
        "state_summary", "valence", "arousal",
        "trend", "trend_strength", "mean_valence", "mean_arousal",
        "session_count", "ready",
    }
    assert required.issubset(ctx.keys())


def test_decode_valence_and_arousal_are_floats(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=1)
    assert isinstance(ctx["valence"], float)
    assert isinstance(ctx["arousal"], float)


def test_decode_trend_is_string(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=1)
    assert isinstance(ctx["trend"], str)
    assert ctx["trend"] in ("improving", "declining", "stable")


def test_decode_session_count_echoed(decoder):
    ctx = decoder.decode(_unit_vec(), arc_history=[], session_count=7)
    assert ctx["session_count"] == 7


def test_decode_raises_on_non_1d_input(decoder):
    bad = np.zeros((2, STATE_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        decoder.decode(bad, arc_history=[], session_count=5)
