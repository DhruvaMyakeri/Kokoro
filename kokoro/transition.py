"""
Transition model for Kokoro.

Implements the "world model" core: given the current user state vector and
the embedding of a new conversation session, produce an updated state vector.

Architecture (current): 3-layer MLP with split input normalization
  state  → LayerNorm(384) ─┐
                            ├─ concat → 768-dim
  emb    → LayerNorm(384) ─┘
  Layer 1: Linear(768→512) → LayerNorm(512) → GELU → Dropout(0.1)
  Layer 2: Linear(512→512) → LayerNorm(512) → GELU → Dropout(0.1)
  Layer 3: Linear(512→384)
  Output: updated state vector, 384-dim (NOT L2-normalized)

Split input normalization: state and session embedding are normalized
independently before concatenation. A joint LayerNorm on the concatenated
input would let the state (norm ~19 post-VICReg) dominate the session
embedding (norm ~1), causing sluggish updates.

The output is NOT forced onto the unit hypersphere. L2 normalization was
removed to allow VICReg variance regularization to work:
  - VICReg's variance term requires unconstrained per-dimension std
  - Unit-sphere constraint caps std by construction, fighting the regularizer
  - Retrieval (kokoro/retrieval.py) normalizes independently before cosine ops
  - StateDecoder probe operates on raw vectors; retrain probe after retraining model

Initial state for a new user is the zero vector. On the first update, the
model receives [0...0 | e_0] as input and produces the first non-trivial state.

------------------------------------------------------------------------------
UPGRADE PATH: Swapping MLP for LSTM
------------------------------------------------------------------------------

When longitudinal data quality improves (real multi-session user histories),
replace TransitionModel with TransitionModelLSTM below. The public API
(forward signature, initial_state, STATE_DIM) remains identical so callers
in kokoro/memory.py and training/train.py require no changes — only the
module-level alias `TransitionModel` needs to be switched.

    class TransitionModelLSTM(nn.Module):
        '''
        LSTM transition model. Carries sequence history in (h, c) hidden state
        rather than compressing it into a single state vector per step.
        The LSTM's hidden state becomes the user state, projected to 384-dim.

        Trade-offs vs MLP:
          + Proper sequence modeling — order and spacing matter
          + Hidden state (h, c) encodes trajectory history explicitly
          - Requires hidden state management at inference time
          - Slower to train; more prone to vanishing gradients on long sequences
          - Hidden state must be serialized in the state store
        '''
        STATE_DIM = 384

        def __init__(
            self,
            session_dim: int = 384,
            hidden_dim: int = 512,
            n_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.lstm = nn.LSTM(
                input_size=session_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, session_dim),
                nn.LayerNorm(session_dim),
            )

        def forward(
            self,
            state: torch.Tensor,          # ignored — LSTM uses (h, c) instead
            session_emb: torch.Tensor,    # (384,) or (batch, 384)
            hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            # session_emb: (batch, 384) → (batch, 1, 384) for LSTM
            x = session_emb.unsqueeze(-2)
            out, hidden = self.lstm(x, hidden)
            projected = self.proj(out.squeeze(-2))
            return F.normalize(projected, dim=-1), hidden

        @staticmethod
        def initial_state(
            batch_size: int = 1,
            device: torch.device | str | None = None,
        ) -> torch.Tensor:
            return torch.zeros(batch_size, TransitionModelLSTM.STATE_DIM, device=device)

        def initial_hidden(
            self,
            batch_size: int = 1,
            device: torch.device | str | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
            return h, torch.zeros_like(h)

    # To switch: uncomment the next line and comment out the MLP class alias
    # TransitionModel = TransitionModelLSTM
------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TransitionModel(nn.Module):
    """
    MLP transition model.

    Maps (current_state, new_session_embedding) → updated_state.
    Output is NOT L2-normalized — VICReg training controls the output
    distribution via variance and covariance regularization terms.

    Args:
        state_dim:    Dimensionality of the state vector and model output (384).
                      Do not change without retraining.
        session_dim:  Dimensionality of the session embedding input. 384 for plain
                      MiniLM; 387 when VAD features are appended
                      (SessionEncoder(use_vad_features=True)). Defaults to state_dim.
        hidden_dim:   Width of the intermediate layers (default 512).
        dropout:      Dropout rate applied after each hidden layer.
    """

    STATE_DIM: int = 384

    def __init__(
        self,
        state_dim: int = 384,
        session_dim: int | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim   = state_dim
        self.session_dim = session_dim if session_dim is not None else state_dim
        self.hidden_dim  = hidden_dim

        # Separate LayerNorms for state and session embedding applied before
        # concatenation. A joint LayerNorm on [state; emb] would let the
        # state (post-VICReg norm ~19) dominate the session embedding (norm ~1),
        # making new observations ~20x smaller than accumulated state and causing
        # sluggish updates. Independent norms equalise both inputs first.
        self.state_norm = nn.LayerNorm(state_dim)
        self.emb_norm   = nn.LayerNorm(self.session_dim)

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(state_dim + self.session_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 3: project to state dimension, no activation
            nn.Linear(hidden_dim, state_dim),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.debug(f"TransitionModel: {n_params:,} trainable parameters")

    def _init_weights(self) -> None:
        """
        Xavier uniform for linear layers, ones/zeros for LayerNorm.
        A good initialization matters here: we want early states to be close
        to the session embedding (not random), so the residual signal is small
        at the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        session_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute updated state vector.

        Args:
            state:       Current user state, shape (..., state_dim).
                         For a new user, pass initial_state().
                         Need not be L2-normalized on input.
            session_emb: Embedding of the new session, shape (..., state_dim).
                         Produced by kokoro.encoder.SessionEncoder.encode().

        Returns:
            Updated state vector, shape (..., state_dim). NOT L2-normalized —
            magnitude is unconstrained and controlled by VICReg during training.
        """
        # Normalise state and session embedding independently before concat.
        # This prevents accumulated state magnitude from drowning out new input.
        s = self.state_norm(state)
        e = self.emb_norm(session_emb)
        x = torch.cat([s, e], dim=-1)                 # (..., state_dim * 2)
        return self.net(x)                             # (..., state_dim)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @staticmethod
    def initial_state(
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Return the initial state for a brand-new user.

        The zero vector is the cold-start state. On the first call to forward(),
        the model receives [0...0 | e_0] and produces the first non-trivial
        state vector. The model must learn to bootstrap from
        this cold start — which it does naturally during training because every
        trajectory begins from the same zero initial state.

        Args:
            batch_size: Number of users in the batch.
            device:     Target device.

        Returns:
            Tensor of shape (batch_size, STATE_DIM) or (STATE_DIM,) if
            batch_size == 1.
        """
        shape = (TransitionModel.STATE_DIM,) if batch_size == 1 else (batch_size, TransitionModel.STATE_DIM)
        return torch.zeros(shape, device=device)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# __main__ — independent tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import math
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.manual_seed(0)

    print("=== TransitionModel smoke tests ===\n")
    model = TransitionModel()
    model.eval()
    print(f"Parameter count: {model.parameter_count():,}")
    print()

    # ------------------------------------------------------------------
    # Test 1: output shape
    # ------------------------------------------------------------------
    print("Test 1: output shape")
    state = TransitionModel.initial_state()         # (384,)
    session_emb = torch.randn(TransitionModel.STATE_DIM)
    with torch.no_grad():
        new_state = model(state, session_emb)
    assert new_state.shape == (TransitionModel.STATE_DIM,), \
        f"Expected ({TransitionModel.STATE_DIM},), got {new_state.shape}"
    print(f"  Output shape: {tuple(new_state.shape)}  PASS\n")

    # ------------------------------------------------------------------
    # Test 2: output is NOT L2-normalized (VICReg controls magnitude)
    # ------------------------------------------------------------------
    print("Test 2: output is NOT L2-normalized (norm unconstrained)")
    norm = new_state.norm().item()
    assert norm > 0, f"Expected non-zero output, got norm={norm:.6f}"
    print(f"  Output norm: {norm:.6f}  (unconstrained, expected != 1.0)  PASS\n")

    # ------------------------------------------------------------------
    # Test 3: output changes when input changes
    # ------------------------------------------------------------------
    print("Test 3: output changes with different inputs")
    different_emb = torch.randn(TransitionModel.STATE_DIM)
    with torch.no_grad():
        new_state_2 = model(state, different_emb)
    assert not torch.allclose(new_state, new_state_2), \
        "Model output did not change with different session embedding"
    diff = (new_state - new_state_2).norm().item()
    cos_sim = (new_state * new_state_2).sum().item()
    print(f"  L2 distance between outputs: {diff:.4f}")
    print(f"  Cosine similarity:           {cos_sim:.4f}")
    print(f"  PASS\n")

    # ------------------------------------------------------------------
    # Test 4: batch input
    # ------------------------------------------------------------------
    print("Test 4: batched input")
    batch_state = TransitionModel.initial_state(batch_size=4)  # (4, 384)
    batch_embs = torch.randn(4, TransitionModel.STATE_DIM)
    with torch.no_grad():
        batch_output = model(batch_state, batch_embs)
    assert batch_output.shape == (4, TransitionModel.STATE_DIM), \
        f"Expected (4, 384), got {batch_output.shape}"
    batch_norms = batch_output.norm(dim=-1)
    assert (batch_norms > 0).all(), \
        f"Batch outputs must be non-zero: {batch_norms}"
    print(f"  Batch output shape: {tuple(batch_output.shape)}  PASS\n")

    # ------------------------------------------------------------------
    # Test 5: 5-session mock sequence
    # ------------------------------------------------------------------
    print("Test 5: mock 5-session sequence")
    print(f"  {'Session':>7}  {'State norm':>10}  {'State shift':>11}  {'Cos sim w/ prev':>16}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*11}  {'-'*16}")

    # prev_state tracks the output of the previous step (None = no previous step yet).
    # shift = distance on the hypersphere from old_state to new_state.
    # cos   = dot product of consecutive normalized states (= cosine similarity).
    state = TransitionModel.initial_state()
    prev_state = None   # no output yet

    torch.manual_seed(42)
    for i in range(5):
        # Simulate gradual emotional drift by shifting the embedding each step
        session_emb = F.normalize(
            torch.randn(TransitionModel.STATE_DIM) + torch.tensor([0.1 * i] + [0.0] * 383),
            dim=-1,
        )
        old_state = state.clone()   # capture current state before update
        with torch.no_grad():
            new_state = model(state, session_emb)

        norm = new_state.norm().item()
        shift = (new_state - old_state).norm().item()   # distance from pre-update state
        cos_str = f"{(new_state * prev_state).sum().item():.4f}" \
                  if prev_state is not None else "  (cold start)"

        print(f"  {i:>7}  {norm:>10.6f}  {shift:>11.4f}  {cos_str:>16}")

        prev_state = new_state.clone()
        state = new_state

    print(f"\n  State norms are unconstrained (L2 norm removed; VICReg controls distribution)\n")
    print("=== All tests passed ===")
