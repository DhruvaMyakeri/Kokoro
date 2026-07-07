"""
Tests for kokoro/transition.py

Covers:
  - Output shape (384,)
  - output magnitude unconstrained (L2 norm removed for VICReg)
  - Output changes with different inputs
  - initial_state returns (384,) zeros
  - 5-session sequence produces different states at each step
"""

import pytest
import torch

from kokoro.transition import TransitionModel

STATE_DIM = TransitionModel.STATE_DIM  # 384


# ---------------------------------------------------------------------------
# Module-scoped model (loaded once for all tests in this file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    torch.manual_seed(42)
    m = TransitionModel()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# initial_state
# ---------------------------------------------------------------------------

def test_initial_state_shape():
    state = TransitionModel.initial_state()
    assert state.shape == (STATE_DIM,)


def test_initial_state_is_zeros():
    state = TransitionModel.initial_state()
    assert torch.all(state == 0.0)


def test_initial_state_batch():
    batch = TransitionModel.initial_state(batch_size=4)
    assert batch.shape == (4, STATE_DIM)
    assert torch.all(batch == 0.0)


# ---------------------------------------------------------------------------
# forward — single vector
# ---------------------------------------------------------------------------

def test_output_shape(model):
    state       = TransitionModel.initial_state()
    session_emb = torch.randn(STATE_DIM)
    with torch.no_grad():
        out = model(state, session_emb)
    assert out.shape == (STATE_DIM,)


def test_output_is_not_l2_normalized(model):
    """L2 normalization was removed so VICReg's variance term can operate
    (unit-sphere outputs cap per-dim std at ~0.051 vs the required gamma=1.0).
    The output must simply be finite and non-zero; magnitude is unconstrained."""
    state       = TransitionModel.initial_state()
    session_emb = torch.randn(STATE_DIM)
    with torch.no_grad():
        out = model(state, session_emb)
    norm = out.norm().item()
    assert norm > 0.0 and torch.isfinite(out).all(), f"Bad output norm: {norm}"


def test_output_changes_with_different_session_embeddings(model):
    state = TransitionModel.initial_state()
    torch.manual_seed(0)
    emb1 = torch.randn(STATE_DIM)
    emb2 = torch.randn(STATE_DIM)
    with torch.no_grad():
        out1 = model(state, emb1)
        out2 = model(state, emb2)
    assert not torch.allclose(out1, out2), \
        "Different session embeddings should produce different state outputs"


# ---------------------------------------------------------------------------
# forward — batch
# ---------------------------------------------------------------------------

def test_batch_output_shape(model):
    batch_state = TransitionModel.initial_state(batch_size=4)
    batch_embs  = torch.randn(4, STATE_DIM)
    with torch.no_grad():
        out = model(batch_state, batch_embs)
    assert out.shape == (4, STATE_DIM)


def test_batch_outputs_finite_and_nonzero(model):
    batch_state = TransitionModel.initial_state(batch_size=4)
    batch_embs  = torch.randn(4, STATE_DIM)
    with torch.no_grad():
        out = model(batch_state, batch_embs)
    norms = out.norm(dim=-1)
    assert (norms > 0).all() and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5-session sequence
# ---------------------------------------------------------------------------

def test_five_session_sequence_produces_different_states(model):
    """
    Each step in a 5-session sequence should produce a state different from
    both the initial state and the previous step's state.
    """
    torch.manual_seed(7)
    state  = TransitionModel.initial_state()
    states = [state.clone()]

    for _ in range(5):
        session_emb = torch.randn(STATE_DIM)
        with torch.no_grad():
            state = model(state, session_emb)
        states.append(state.clone())

    # Every consecutive pair of states must differ
    for i in range(len(states) - 1):
        assert not torch.allclose(states[i], states[i + 1]), \
            f"States at step {i} and {i+1} are identical — model is not updating"


def test_five_session_states_all_finite(model):
    """All output states in a sequence must stay finite with non-zero norm
    (magnitude is unconstrained — VICReg controls the output distribution)."""
    torch.manual_seed(13)
    state = TransitionModel.initial_state()

    for _ in range(5):
        session_emb = torch.randn(STATE_DIM)
        with torch.no_grad():
            state = model(state, session_emb)
        norm = state.norm().item()
        assert norm > 0.0 and torch.isfinite(state).all(), f"Bad state norm: {norm}"
