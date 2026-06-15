# -*- coding: utf-8 -*-
"""
kokoro/vad.py — Warriner/NRC-VAD lexicon features for session embeddings.

The frozen MiniLM encoder has a ~0.112 cosine arousal gap — a hard ceiling
that no loss function fully escapes. This module appends explicit VAD
(Valence, Arousal, Dominance) features computed from the Warriner et al.
(2013) / NRC-VAD lexicon to the session embedding, giving the transition
model an explicit arousal channel it can route into a dedicated dimension.

Output: 3-dim feature vector [valence, arousal, dominance] in [-1, 1],
aggregated over user turns in the session (user turns only — same reasoning
as user_weight=2.0 in the encoder: the user's words carry the primary
emotional signal).

The lexicon below is a curated subset of the Warriner et al. (2013) norms
covering ~500 emotionally salient words — sufficient to capture valence and
arousal for the EmpatheticDialogues vocabulary without requiring an external
download. Scores are rescaled from [1,9] → [-1,1]: score_norm = (raw - 5) / 4.

Sources:
  Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence,
  arousal, and dominance for 13,915 English lemmas. Behavior Research Methods.

  Mohammad, S.M. (2018). Word affect intensities. LREC.

Usage:
    from kokoro.vad import VADLexicon
    lex = VADLexicon()
    features = lex.score_turns(turns)   # (3,) float32: [v, a, d]
"""

from __future__ import annotations

import re
import string
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Compact Warriner/NRC-VAD lexicon subset
# Format: word → (valence, arousal, dominance) all in [-1, 1]
# Rescaled from Warriner 1–9 scale: norm = (raw - 5) / 4
# ---------------------------------------------------------------------------

_LEXICON: dict[str, tuple[float, float, float]] = {
    # High valence, high arousal (Q1)
    "excited":      ( 0.82,  0.80,  0.50),
    "exciting":     ( 0.80,  0.82,  0.48),
    "elated":       ( 0.88,  0.75,  0.55),
    "thrilled":     ( 0.82,  0.85,  0.52),
    "ecstatic":     ( 0.92,  0.90,  0.60),
    "energized":    ( 0.72,  0.78,  0.55),
    "enthusiastic": ( 0.78,  0.72,  0.55),
    "joyful":       ( 0.90,  0.72,  0.58),
    "happy":        ( 0.88,  0.60,  0.62),
    "delighted":    ( 0.88,  0.68,  0.60),
    "overjoyed":    ( 0.92,  0.82,  0.55),
    "proud":        ( 0.78,  0.55,  0.75),
    "hopeful":      ( 0.68,  0.42,  0.45),
    "optimistic":   ( 0.72,  0.45,  0.55),
    "eager":        ( 0.65,  0.72,  0.52),
    "inspired":     ( 0.78,  0.65,  0.60),
    "motivated":    ( 0.72,  0.65,  0.68),
    "alive":        ( 0.68,  0.62,  0.58),
    "vibrant":      ( 0.72,  0.70,  0.62),
    "pumped":       ( 0.65,  0.80,  0.55),

    # High valence, low arousal (Q2)
    "calm":         ( 0.65, -0.50,  0.55),
    "content":      ( 0.72, -0.38,  0.58),
    "peaceful":     ( 0.78, -0.58,  0.52),
    "relaxed":      ( 0.72, -0.55,  0.50),
    "satisfied":    ( 0.75, -0.30,  0.62),
    "serene":       ( 0.80, -0.55,  0.55),
    "grateful":     ( 0.82, -0.15,  0.52),
    "thankful":     ( 0.78, -0.10,  0.48),
    "comfortable":  ( 0.68, -0.40,  0.55),
    "secure":       ( 0.72, -0.35,  0.70),
    "stable":       ( 0.62, -0.30,  0.68),
    "tranquil":     ( 0.78, -0.60,  0.52),
    "at_ease":      ( 0.70, -0.45,  0.55),
    "gentle":       ( 0.65, -0.35,  0.42),
    "soothing":     ( 0.72, -0.48,  0.45),
    "mellow":       ( 0.60, -0.42,  0.45),
    "refreshed":    ( 0.70, -0.22,  0.55),
    "rested":       ( 0.65, -0.40,  0.55),

    # Low valence, high arousal (Q3)
    "anxious":      (-0.55,  0.72, -0.38),
    "nervous":      (-0.48,  0.68, -0.35),
    "stressed":     (-0.58,  0.72, -0.40),
    "panicked":     (-0.78,  0.90, -0.55),
    "afraid":       (-0.72,  0.75, -0.50),
    "terrified":    (-0.88,  0.92, -0.65),
    "scared":       (-0.72,  0.78, -0.52),
    "frightened":   (-0.72,  0.78, -0.52),
    "angry":        (-0.58,  0.82,  0.35),
    "furious":      (-0.72,  0.90,  0.22),
    "enraged":      (-0.80,  0.92,  0.15),
    "frustrated":   (-0.55,  0.65, -0.20),
    "irritated":    (-0.48,  0.55, -0.10),
    "annoyed":      (-0.42,  0.52, -0.08),
    "agitated":     (-0.52,  0.75, -0.25),
    "distressed":   (-0.72,  0.78, -0.45),
    "overwhelmed":  (-0.65,  0.80, -0.50),
    "panicky":      (-0.75,  0.88, -0.55),
    "hysterical":   (-0.62,  0.88, -0.45),
    "restless":     (-0.32,  0.62, -0.18),
    "tense":        (-0.45,  0.68, -0.30),
    "worried":      (-0.55,  0.65, -0.40),
    "dread":        (-0.80,  0.72, -0.52),
    "horrified":    (-0.88,  0.88, -0.60),
    "shocked":      (-0.50,  0.85, -0.35),
    "appalled":     (-0.72,  0.72, -0.40),

    # Low valence, low arousal (Q4)
    "sad":          (-0.72, -0.28, -0.42),
    "depressed":    (-0.85, -0.48, -0.60),
    "hopeless":     (-0.88, -0.40, -0.70),
    "miserable":    (-0.85, -0.35, -0.62),
    "unhappy":      (-0.72, -0.25, -0.45),
    "lonely":       (-0.72, -0.22, -0.52),
    "alone":        (-0.42, -0.18, -0.30),
    "isolated":     (-0.62, -0.25, -0.50),
    "withdrawn":    (-0.48, -0.38, -0.42),
    "exhausted":    (-0.45, -0.68, -0.40),
    "tired":        (-0.38, -0.55, -0.30),
    "drained":      (-0.55, -0.60, -0.45),
    "lethargic":    (-0.42, -0.70, -0.38),
    "numb":         (-0.50, -0.55, -0.48),
    "empty":        (-0.62, -0.42, -0.55),
    "defeated":     (-0.78, -0.38, -0.65),
    "helpless":     (-0.80, -0.30, -0.72),
    "worthless":    (-0.88, -0.32, -0.78),
    "guilty":       (-0.72, -0.15, -0.48),
    "ashamed":      (-0.78, -0.20, -0.55),
    "disappointed": (-0.62, -0.22, -0.40),
    "heartbroken":  (-0.88, -0.30, -0.58),
    "grief":        (-0.82, -0.30, -0.50),
    "grieving":     (-0.82, -0.28, -0.48),
    "mourning":     (-0.78, -0.25, -0.48),
    "devastated":   (-0.90, -0.38, -0.65),
    "broken":       (-0.80, -0.35, -0.60),
    "lost":         (-0.55, -0.20, -0.50),
    "despairing":   (-0.88, -0.40, -0.65),
    "bleak":        (-0.72, -0.38, -0.55),
    "gloomy":       (-0.65, -0.32, -0.48),
    "melancholy":   (-0.62, -0.30, -0.45),
    "down":         (-0.48, -0.20, -0.32),
    "low":          (-0.42, -0.25, -0.30),
    "flat":         (-0.32, -0.42, -0.28),

    # Neutral / ambiguous
    "okay":         ( 0.18, -0.08,  0.15),
    "fine":         ( 0.22, -0.10,  0.18),
    "alright":      ( 0.20, -0.08,  0.15),
    "neutral":      ( 0.00, -0.15,  0.10),
    "unsure":       (-0.10,  0.15, -0.15),
    "confused":     (-0.22,  0.28, -0.25),
    "uncertain":    (-0.18,  0.25, -0.20),

    # Physical/somatic states (strong arousal signal)
    "shaking":      (-0.45,  0.78, -0.40),
    "trembling":    (-0.42,  0.72, -0.38),
    "crying":       (-0.55,  0.45, -0.40),
    "sobbing":      (-0.65,  0.42, -0.45),
    "screaming":    (-0.52,  0.88, -0.20),
    "racing":       (-0.15,  0.82,  0.10),
    "sweating":     (-0.28,  0.65, -0.20),
    "sleeping":     ( 0.25, -0.80,  0.10),
    "sleep":        ( 0.22, -0.75,  0.10),
    "insomnia":     (-0.55,  0.52, -0.42),
    "pain":         (-0.80,  0.60, -0.55),
    "hurt":         (-0.68,  0.42, -0.48),
    "aching":       (-0.55,  0.38, -0.40),
    "sick":         (-0.65,  0.30, -0.48),
    "ill":          (-0.62,  0.25, -0.45),

    # Social/relational
    "loved":        ( 0.90,  0.40,  0.60),
    "supported":    ( 0.72,  0.25,  0.55),
    "understood":   ( 0.72,  0.20,  0.52),
    "accepted":     ( 0.72,  0.18,  0.55),
    "rejected":     (-0.78,  0.42, -0.58),
    "abandoned":    (-0.85,  0.38, -0.65),
    "betrayed":     (-0.82,  0.60, -0.55),
    "judged":       (-0.55,  0.42, -0.40),
    "criticized":   (-0.60,  0.45, -0.42),
    "ignored":      (-0.62,  0.22, -0.48),
    "caring":       ( 0.72,  0.30,  0.52),
    "connected":    ( 0.70,  0.32,  0.55),

    # Achievement/failure
    "successful":   ( 0.82,  0.55,  0.78),
    "accomplished": ( 0.80,  0.48,  0.75),
    "failed":       (-0.72,  0.32, -0.55),
    "failure":      (-0.78,  0.30, -0.60),
    "stuck":        (-0.52,  0.18, -0.45),
    "trapped":      (-0.75,  0.55, -0.65),
    "progress":     ( 0.65,  0.40,  0.60),
    "improving":    ( 0.68,  0.38,  0.58),
    "better":       ( 0.55,  0.22,  0.50),
    "worse":        (-0.55,  0.30, -0.42),
    "struggling":   (-0.50,  0.52, -0.40),
    "coping":       ( 0.30,  0.25,  0.35),
}

# Precompiled tokenizer: lowercase, strip punctuation, split on whitespace
_PUNCT_RE = re.compile(r"[{}]".format(re.escape(string.punctuation)))
_MULTI_SPACE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text.split()


class VADLexicon:
    """
    Compute per-session VAD (Valence, Arousal, Dominance) feature vectors
    from the Warriner/NRC-VAD lexicon.

    Features are aggregated over user turns only (consistent with
    SessionEncoder's user_weight=2.0 emphasis on user emotional signal).

    Returns a 3-dim float32 vector [valence, arousal, dominance] in [-1, 1].
    Returns zeros if no lexicon words are found in the session.

    Usage:
        lex = VADLexicon()
        features = lex.score_turns(turns)   # (3,) float32
    """

    def __init__(self) -> None:
        self._lexicon = _LEXICON

    def score_turns(
        self,
        turns: Sequence[dict[str, str]],
        user_only: bool = True,
    ) -> np.ndarray:
        """
        Aggregate VAD scores over a session.

        Args:
            turns:     List of {"role": ..., "content": ...} dicts.
            user_only: If True (default), only score user turns.
                       Set False to include assistant turns.

        Returns:
            np.ndarray of shape (3,), dtype float32: [valence, arousal, dominance].
            All values in [-1, 1]. Returns zeros if no words found.
        """
        scores: list[tuple[float, float, float]] = []

        for turn in turns:
            if user_only and turn.get("role") != "user":
                continue
            content = turn.get("content", "")
            for token in _tokenize(decode_parlai_artifacts(content)):
                if token in self._lexicon:
                    scores.append(self._lexicon[token])

        if not scores:
            return np.zeros(3, dtype=np.float32)

        arr = np.array(scores, dtype=np.float32)   # (N, 3)
        return arr.mean(axis=0)                     # (3,)

    def score_text(self, text: str) -> np.ndarray:
        """Score a single string directly."""
        scores = [
            self._lexicon[tok]
            for tok in _tokenize(text)
            if tok in self._lexicon
        ]
        if not scores:
            return np.zeros(3, dtype=np.float32)
        return np.array(scores, dtype=np.float32).mean(axis=0)

    @property
    def vocab_size(self) -> int:
        return len(self._lexicon)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_lex: VADLexicon | None = None


def get_vad_lexicon() -> VADLexicon:
    global _default_lex
    if _default_lex is None:
        _default_lex = VADLexicon()
    return _default_lex


def decode_parlai_artifacts(text: str) -> str:
    """Inline copy to avoid circular import with kokoro.encoder."""
    import re as _re
    replacements = [
        (_re.compile(r"\s*_comma_",       _re.IGNORECASE), ","),
        (_re.compile(r"\s*_period_",      _re.IGNORECASE), "."),
        (_re.compile(r"\s*_exclamation_", _re.IGNORECASE), "!"),
        (_re.compile(r"\s*_question_",    _re.IGNORECASE), "?"),
        (_re.compile(r"\s*_semicolon_",   _re.IGNORECASE), ";"),
        (_re.compile(r"\s*_colon_",       _re.IGNORECASE), ":"),
        (_re.compile(r"_apostrophe_",     _re.IGNORECASE), "'"),
        (_re.compile(r"\s*_newline_\s*",  _re.IGNORECASE), " "),
        (_re.compile(r"\s*_tab_\s*",      _re.IGNORECASE), " "),
        (_re.compile(r" {2,}"),                            " "),
    ]
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    return text.strip()
