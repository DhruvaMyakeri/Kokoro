"""
Session encoder for Kokoro.

Converts a conversation session (list of turns) into a single 384-dimensional
embedding that represents the emotional and semantic content of that session.

Design decisions:
  - Model: all-MiniLM-L6-v2 (22 MB, CPU-capable, 384-dim output)
  - Unit of encoding: the full session, not individual messages
  - Pooling: weighted mean over per-turn embeddings
      user turns   weight 2.0 — user words carry primary emotional signal
      assistant turns  weight 1.0 — context, not primary affect source
  - ParlAI artifact decoding: EmpatheticDialogues (and other ParlAI-sourced
    datasets) encode punctuation as _comma_, _period_, etc. These are decoded
    to real punctuation before encoding so the sentence transformer sees
    natural text. This is encoder responsibility, not data pipeline
    responsibility — the data pipeline stores text as-is; the encoder is
    the layer that must produce clean embeddings regardless of source.
  - The encoder is stateless — it holds only the loaded model.
    It has no knowledge of users, sessions, or trajectories.
"""

from __future__ import annotations

import re
import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ParlAI artifact decoding
# ---------------------------------------------------------------------------

# Each pattern absorbs optional whitespace *before* the token so that
# "word _comma_ next" → "word, next"  (not "word , next")
# "word_comma_ next"  → "word, next"  (ParlAI often concatenates directly)
# The space *after* the token is preserved, so no trailing-space cleanup
# is needed for mid-sentence tokens. A final multi-space collapse handles
# any residual double spaces (e.g., from _newline_ replacements).
_PARLAI_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\s*_comma_",       re.IGNORECASE), ","),
    (re.compile(r"\s*_period_",      re.IGNORECASE), "."),
    (re.compile(r"\s*_exclamation_", re.IGNORECASE), "!"),
    (re.compile(r"\s*_question_",    re.IGNORECASE), "?"),
    (re.compile(r"\s*_semicolon_",   re.IGNORECASE), ";"),
    (re.compile(r"\s*_colon_",       re.IGNORECASE), ":"),
    (re.compile(r"_apostrophe_",     re.IGNORECASE), "'"),
    (re.compile(r"\s*_newline_\s*",  re.IGNORECASE), " "),
    (re.compile(r"\s*_tab_\s*",      re.IGNORECASE), " "),
    # Collapse any residual double spaces
    (re.compile(r" {2,}"),                           " "),
]


def decode_parlai_artifacts(text: str) -> str:
    """
    Replace ParlAI punctuation tokens with their actual characters.

    Examples:
        "hello _comma_ world"  -> "hello, world"
        "really _period_"      -> "really."
        "what _question_"      -> "what?"
    """
    for pattern, replacement in _PARLAI_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text.strip()


# ---------------------------------------------------------------------------
# SessionEncoder
# ---------------------------------------------------------------------------

class SessionEncoder:
    """
    Encodes a conversation session into a 384-dim embedding.

    Usage:
        encoder = SessionEncoder()
        embedding = encoder.encode(session_turns)
        # embedding.shape == (384,), dtype float32

    Args:
        model_name: sentence-transformers model identifier.
                    Defaults to all-MiniLM-L6-v2.
        user_weight: Relative weight for user turns during mean pooling.
                     Assistant turn weight is always 1.0.
        device:     "cpu", "cuda", or None (auto-detect).
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        user_weight: float = 2.0,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._user_weight = user_weight
        self._device = device
        self._model = None  # lazy load

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading sentence transformer: {self._model_name}")
        kwargs: dict = {}
        if self._device is not None:
            kwargs["device"] = self._device
        self._model = SentenceTransformer(self._model_name, **kwargs)
        logger.info(
            f"  Model loaded. Embedding dim: {self.EMBEDDING_DIM}, "
            f"device: {self._model.device}"
        )

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        turns: Sequence[dict[str, str]],
    ) -> np.ndarray:
        """
        Encode a session (list of turns) into a single 384-dim embedding.

        Args:
            turns: List of {"role": "user"|"assistant", "content": "..."} dicts.
                   At least one turn must have non-empty content.

        Returns:
            np.ndarray of shape (384,), dtype float32.

        Raises:
            ValueError: if turns is empty or all content is blank after cleaning.
        """
        if not turns:
            raise ValueError("turns must be non-empty")

        texts, weights = self._prepare_turns(turns)

        if not texts:
            raise ValueError(
                "All turns had empty content after cleaning — cannot encode."
            )

        # Encode all turns in a single batch (more efficient than one by one)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )  # shape: (n_turns, 384)

        # Weighted mean pooling
        weights_arr = np.array(weights, dtype=np.float32)  # (n_turns,)
        weights_arr /= weights_arr.sum()                    # normalise to sum=1
        pooled = (embeddings * weights_arr[:, np.newaxis]).sum(axis=0)  # (384,)

        return pooled.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single cleaned string. Used by the transition model for
        encoding the current message during inference.

        Returns np.ndarray of shape (384,), dtype float32.
        """
        if not text.strip():
            raise ValueError("text must be non-empty")
        result = self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return result[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_turns(
        self,
        turns: Sequence[dict[str, str]],
    ) -> tuple[list[str], list[float]]:
        """
        Clean and weight turns, dropping empty ones.

        Returns:
            texts:   list of cleaned turn strings
            weights: parallel list of per-turn weights
        """
        texts: list[str] = []
        weights: list[float] = []

        for turn in turns:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")

            cleaned = decode_parlai_artifacts(content)
            if not cleaned:
                continue

            weight = self._user_weight if role == "user" else 1.0
            texts.append(cleaned)
            weights.append(weight)

        return texts, weights


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_default_encoder: SessionEncoder | None = None


def get_encoder() -> SessionEncoder:
    """Return the module-level default encoder, loading it on first call."""
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = SessionEncoder()
    return _default_encoder


def encode_session(turns: Sequence[dict[str, str]]) -> np.ndarray:
    """Convenience wrapper: encode a session using the default encoder."""
    return get_encoder().encode(turns)


# ---------------------------------------------------------------------------
# __main__ — independent smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    encoder = SessionEncoder()

    # --- 1. ParlAI artifact decoding ---
    print("\n=== ParlAI artifact decoding ===")
    test_cases = [
        ("hello _comma_ world _period_",    "hello, world."),
        ("really _exclamation_",            "really!"),
        ("what _question_",                 "what?"),
        ("no artifacts here",               "no artifacts here"),
        ("_comma_ leading",                 ", leading"),
        ("Mixed _comma_ and normal, text.", "Mixed, and normal, text."),
    ]
    all_pass = True
    for raw, expected in test_cases:
        got = decode_parlai_artifacts(raw)
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {raw!r:45s} -> {got!r}")
    if not all_pass:
        print("  Some decoding tests FAILED — check _PARLAI_REPLACEMENTS")
        sys.exit(1)

    # --- 2. Encode a minimal session ---
    print("\n=== Encoding smoke test ===")
    sample_turns = [
        {"role": "user",      "content": "I've been feeling really down lately _comma_ can't sleep."},
        {"role": "assistant", "content": "I'm sorry to hear that _period_ What's been on your mind _question_"},
        {"role": "user",      "content": "Work stress mostly _period_ My boss is really difficult."},
        {"role": "assistant", "content": "That sounds exhausting _period_ Have you talked to anyone about it _question_"},
    ]

    t0 = time.perf_counter()
    emb = encoder.encode(sample_turns)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Shape:    {emb.shape}")
    print(f"  Dtype:    {emb.dtype}")
    print(f"  Norm:     {np.linalg.norm(emb):.4f}")
    print(f"  Min/Max:  {emb.min():.4f} / {emb.max():.4f}")
    print(f"  Latency:  {elapsed:.1f} ms")

    assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"
    assert emb.dtype == np.float32

    # --- 3. User-weight sensitivity check ---
    print("\n=== Weight sensitivity check ===")
    user_only = [{"role": "user",      "content": "I feel completely hopeless and exhausted."}]
    asst_only = [{"role": "assistant", "content": "I feel completely hopeless and exhausted."}]

    emb_user = encoder.encode(user_only)
    emb_asst = encoder.encode(asst_only)
    # Same text, so embeddings should be identical regardless of weight
    assert np.allclose(emb_user, emb_asst, atol=1e-5), \
        "Single-turn embeddings should be identical regardless of role weight"
    print("  Single-turn equality (role weight irrelevant for 1 turn): PASS")

    # Two-turn session: user vs assistant carrying the emotional content
    neutral  = "The weather is nice today."
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
    print(f"  Cosine sim (distress in user vs assistant role): {cosine_sim:.4f}")
    print(f"  (< 1.0 confirms weighting shifts the embedding — expected)")
    assert cosine_sim < 1.0, "Embeddings should differ when distress is in different roles"
    print("  Weight sensitivity: PASS")

    # --- 4. Edge cases ---
    print("\n=== Edge cases ===")
    try:
        encoder.encode([])
        print("  Empty turns: FAIL (should have raised ValueError)")
    except ValueError:
        print("  Empty turns raises ValueError: PASS")

    try:
        encoder.encode([{"role": "user", "content": ""}])
        print("  All-blank turns: FAIL (should have raised ValueError)")
    except ValueError:
        print("  All-blank turns raises ValueError: PASS")

    # --- 5. Real trajectory session ---
    print("\n=== Real EmpatheticDialogues session (with ParlAI artifacts) ===")
    parlai_turns = [
        {"role": "assistant", "content": "I was driving through flood waters and my car stalled out _period_"},
        {"role": "assistant", "content": "Holy cow _period_ I hate when that happens _period_ It happens a lot here since I live near the ocean _period_ How scary _exclamation_"},
        {"role": "user",      "content": "Yes _comma_ very scary _period_ I had to call a tow truck _period_"},
    ]
    emb_real = encoder.encode(parlai_turns)
    print(f"  Encoded shape: {emb_real.shape}, norm: {np.linalg.norm(emb_real):.4f}")
    print("  Real session encoding: PASS")

    print("\n=== All tests passed ===")
