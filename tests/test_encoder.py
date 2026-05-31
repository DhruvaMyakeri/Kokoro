"""
Tests for kokoro/encoder.py

Covers:
  - ParlAI artifact decoding
  - Weighted pooling (user turns 2x assistant)
  - Output shape (384,)
  - Determinism
  - ValueError on empty turns
"""

import numpy as np
import pytest

from kokoro.encoder import SessionEncoder, decode_parlai_artifacts


# ---------------------------------------------------------------------------
# ParlAI artifact decoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("hello _comma_ world",         "hello, world"),
    ("done _period_",               "done."),
    ("what _question_",             "what?"),
    ("wow _exclamation_",           "wow!"),
    ("word _semicolon_ next",       "word; next"),
    ("key _colon_ value",           "key: value"),
    ("it_apostrophe_s fine",         "it's fine"),
    ("hello _comma_ world _period_","hello, world."),
    ("no artifacts here",           "no artifacts here"),
    ("line1 _newline_ line2",       "line1 line2"),
])
def test_parlai_replacements(raw, expected):
    assert decode_parlai_artifacts(raw) == expected


def test_parlai_case_insensitive():
    assert decode_parlai_artifacts("hello _COMMA_ world") == "hello, world"


def test_parlai_strips_result():
    # Leading/trailing whitespace should be stripped
    result = decode_parlai_artifacts("  hello world  ")
    assert result == "hello world"


def test_parlai_double_spaces_collapsed():
    # _newline_ leaves a space; two spaces should collapse to one
    result = decode_parlai_artifacts("a _newline_ b")
    assert "  " not in result


# ---------------------------------------------------------------------------
# SessionEncoder  (module-scoped so the model is loaded only once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder():
    return SessionEncoder()


def test_output_shape(encoder):
    turns = [{"role": "user", "content": "I feel sad today."}]
    emb = encoder.encode(turns)
    assert emb.shape == (384,)


def test_output_dtype(encoder):
    turns = [{"role": "user", "content": "I feel sad today."}]
    emb = encoder.encode(turns)
    assert emb.dtype == np.float32


def test_deterministic(encoder):
    turns = [
        {"role": "user",      "content": "I feel really anxious."},
        {"role": "assistant", "content": "I understand that feeling."},
    ]
    emb1 = encoder.encode(turns)
    emb2 = encoder.encode(turns)
    assert np.allclose(emb1, emb2, atol=1e-6)


def test_empty_turns_raises_value_error(encoder):
    with pytest.raises(ValueError):
        encoder.encode([])


def test_all_blank_content_raises_value_error(encoder):
    with pytest.raises(ValueError):
        encoder.encode([{"role": "user", "content": ""}])


def test_blank_turns_after_whitespace_strip_raises(encoder):
    with pytest.raises(ValueError):
        encoder.encode([{"role": "user", "content": "   "}])


def test_weighted_pooling_user_weight_higher_than_assistant(encoder):
    """
    Same text but distress lives in the user turn vs the assistant turn.
    Because user_weight=2.0 > 1.0 (assistant), swapping which role carries
    the distress should shift the pooled embedding, giving cosine_sim < 1.0.
    """
    neutral  = "The weather is fine today."
    distress = "I haven't slept in days and everything feels pointless."

    turns_user_distress = [
        {"role": "user",      "content": distress},
        {"role": "assistant", "content": neutral},
    ]
    turns_asst_distress = [
        {"role": "user",      "content": neutral},
        {"role": "assistant", "content": distress},
    ]

    emb_ud = encoder.encode(turns_user_distress)
    emb_ad = encoder.encode(turns_asst_distress)

    cosine_sim = float(
        np.dot(emb_ud, emb_ad) / (np.linalg.norm(emb_ud) * np.linalg.norm(emb_ad))
    )
    assert cosine_sim < 1.0, (
        f"Embeddings should differ when distress is in different roles, "
        f"but cosine_sim={cosine_sim:.6f}"
    )


def test_parlai_artifacts_cleaned_before_encoding(encoder):
    """
    ParlAI-encoded text and its cleaned equivalent should be near-identical
    after encoding, confirming decode_parlai_artifacts is applied inside encode().
    """
    parlai_turns = [{"role": "user", "content": "I feel really sad _comma_ nothing helps _period_"}]
    clean_turns  = [{"role": "user", "content": "I feel really sad, nothing helps."}]

    emb_parlai = encoder.encode(parlai_turns)
    emb_clean  = encoder.encode(clean_turns)

    cosine_sim = float(
        np.dot(emb_parlai, emb_clean)
        / (np.linalg.norm(emb_parlai) * np.linalg.norm(emb_clean))
    )
    assert cosine_sim > 0.99, (
        f"ParlAI and cleaned text should produce near-identical embeddings, "
        f"but cosine_sim={cosine_sim:.6f}"
    )


def test_multi_turn_session_encodes_without_error(encoder):
    turns = [
        {"role": "user",      "content": "I've been feeling really down lately."},
        {"role": "assistant", "content": "I'm sorry to hear that. What's been on your mind?"},
        {"role": "user",      "content": "Work stress mostly. My boss is really difficult."},
        {"role": "assistant", "content": "That sounds exhausting. Have you talked to anyone?"},
    ]
    emb = encoder.encode(turns)
    assert emb.shape == (384,)
    assert np.isfinite(emb).all()
