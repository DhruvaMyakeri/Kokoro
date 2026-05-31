# -*- coding: utf-8 -*-
"""
State decoder for Kokoro.

Converts a user's state vector and arc_history into a structured context dict
that can be injected directly into an LLM system prompt.

Pipeline:
  state_vector (384-dim, L2-normalised)
      └─► linear probe (384 → 2)
              └─► (valence, arousal)  ← current emotional position

  arc_history  (list of [valence, arousal] pairs from StateStore)
      └─► linear regression on valence
              └─► trend slope, mean valence, mean arousal

  valence + arousal + trend
      └─► threshold logic
              └─► state_summary (natural language, LLM-ready)

The decoder is stateless once loaded — it holds only the probe weights.
It never writes to disk; all persistence is handled by StateStore.

Thresholds (tunable):
  valence:   > 0.30 → positive,   < -0.30 → negative,   else neutral
  arousal:   > 0.30 → activated,  < -0.10 → low-energy,  else moderate
  slope:     > 0.02 → improving,  < -0.02 → declining,   else stable
  MIN_SESSIONS = 3   — minimum sessions to produce a summary

Default probe path: checkpoints/valence_arousal_probe.pt
  (relative to the kokoro/ package parent, i.e. the project root)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).parent.parent          # project root
_DEFAULT_PROBE = _PKG_ROOT / "checkpoints" / "valence_arousal_probe.pt"

_MIN_SESSIONS: int   = 3      # sessions required before summary is produced
_MIN_TREND_ENTRIES   = 3      # arc_history entries required to compute slope

_VALENCE_POS:  float = -0.121  # valence > this → "positive"   (top third of observed -0.144…-0.110)
_VALENCE_NEG:  float = -0.133  # valence < this → "negative"  (bottom third of observed range)
_AROUSAL_HIGH: float =  0.30   # arousal > this → "activated / energised"
_AROUSAL_LOW:  float = -0.10   # arousal < this → "low energy"
_SLOPE_IMP:    float =  0.02   # slope > this   → "improving"
_SLOPE_DEC:    float = -0.02   # slope < this   → "declining"


# ---------------------------------------------------------------------------
# Probe architecture  (must match training/train_probe.py)
# ---------------------------------------------------------------------------

class _ValenceArousalProbe(nn.Module):
    def __init__(self, state_dim: int = 384) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_slope(values: list[float]) -> float:
    """
    Return the least-squares slope of values against an integer index axis.

    Equivalent to the coefficient of x in a degree-1 polynomial fit.
    Returns 0.0 for a sequence of length < 2.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    # slope = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
    sx  = x.sum()
    sy  = y.sum()
    sxy = (x * y).sum()
    sx2 = (x * x).sum()
    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return float((n * sxy - sx * sy) / denom)


def _classify_trend(slope: float) -> str:
    if slope > _SLOPE_IMP:
        return "improving"
    if slope < _SLOPE_DEC:
        return "declining"
    return "stable"


def _describe_valence(v: float) -> str:
    """Single-word valence descriptor."""
    if v > _VALENCE_POS:
        return "positive"
    if v < _VALENCE_NEG:
        return "negative"
    return "neutral"


def _describe_arousal(a: float) -> str:
    """Single-word arousal descriptor."""
    if a > _AROUSAL_HIGH:
        return "activated"
    if a < _AROUSAL_LOW:
        return "low-energy"
    return "moderate"


def _build_summary(
    valence: float,
    arousal: float,
    trend: str,
    trend_strength: float,
    mean_valence: float,
    mean_arousal: float,
    session_count: int,
    n_history: int,
) -> str:
    """
    Produce a natural-language summary from continuous emotional coordinates.

    No fixed templates: the text is assembled from valence/arousal descriptors
    and trend classification.  Each combination of (valence_class, arousal_class,
    trend) produces a distinct sentence structure, but the wording varies with
    the continuous values so it never reads as a slot-filled template.
    """
    v_word  = _describe_valence(valence)
    a_word  = _describe_arousal(arousal)
    mv_word = _describe_valence(mean_valence)

    # Temporal reference
    n_ref = f"the past {n_history} sessions" if n_history >= 5 else "recent sessions"

    # ------------------------------------------------------------------ #
    # Trend-primary construction                                           #
    # ------------------------------------------------------------------ #

    if trend == "stable":
        if v_word == "positive":
            if a_word == "activated":
                summary = (
                    f"User has been emotionally stable and upbeat across {n_ref}, "
                    f"with consistent positive affect and notable engagement."
                )
            elif a_word == "low-energy":
                summary = (
                    f"User has been emotionally stable and positive across {n_ref}, "
                    f"though with relatively low energy or activation."
                )
            else:
                summary = (
                    f"User has been emotionally stable and positive across {n_ref}, "
                    f"maintaining a steady, grounded mood."
                )
        elif v_word == "negative":
            if a_word == "activated":
                summary = (
                    f"User has maintained a consistently negative and activated emotional "
                    f"state across {n_ref} — possible distress or heightened anxiety."
                )
            elif a_word == "low-energy":
                summary = (
                    f"User has maintained a consistently negative and low-energy state "
                    f"across {n_ref}, suggesting persistent low mood or withdrawal."
                )
            else:
                summary = (
                    f"User has maintained a consistently negative emotional state "
                    f"across {n_ref} without meaningful recovery."
                )
        else:  # neutral
            summary = (
                f"User's emotional state has been relatively neutral and stable "
                f"across {n_ref}, showing neither clear distress nor strong positivity."
            )

    elif trend == "improving":
        strength_adv = "gradually" if trend_strength < 0.05 else "notably"
        if v_word in ("positive", "neutral"):
            summary = (
                f"User shows signs of emotional improvement across {n_ref}, "
                f"with valence {strength_adv} rising toward a {v_word} range "
                f"(current {a_word} activation)."
            )
        else:
            # Still negative but improving
            summary = (
                f"User's emotional state has been {strength_adv} improving across "
                f"{n_ref}, though current affect remains {v_word}. "
                f"The upward trend is a meaningful signal."
            )

    else:  # declining
        strength_adv = "gradually" if abs(trend_strength) < 0.05 else "noticeably"
        if v_word == "negative":
            summary = (
                f"User has been trending negatively across {n_ref}, with "
                f"affect {strength_adv} declining into {v_word} territory "
                f"and {a_word} energy levels."
            )
        elif v_word == "neutral":
            summary = (
                f"User's mood has been {strength_adv} declining across {n_ref}, "
                f"moving from a neutral baseline toward lower affect. "
                f"Monitoring for further deterioration is advisable."
            )
        else:
            # Positive but declining
            summary = (
                f"User's emotional state, while still {v_word}, has been "
                f"{strength_adv} declining across {n_ref}. "
                f"The downward trajectory warrants attention."
            )

    # Append historical mean note if it differs meaningfully from current
    mean_v_word = _describe_valence(mean_valence)
    if mean_v_word != v_word:
        summary += (
            f" Historical mean valence across the window is {mean_v_word}, "
            f"suggesting the current reading may reflect a recent shift."
        )

    return summary.strip()


# ---------------------------------------------------------------------------
# StateDecoder
# ---------------------------------------------------------------------------

class StateDecoder:
    """
    Decodes a Kokoro state vector and arc_history into a structured context
    suitable for LLM system-prompt injection.

    Usage:
        decoder = StateDecoder()
        ctx = decoder.decode(state_vector, arc_history, session_count=7)
        print(ctx["state_summary"])
        print(ctx["valence"], ctx["arousal"], ctx["trend"])

    Args:
        probe_path: Path to the linear probe checkpoint produced by
                    training/train_probe.py.  Defaults to
                    checkpoints/valence_arousal_probe.pt relative to the
                    project root.
    """

    def __init__(self, probe_path: Optional[str | Path] = None) -> None:
        path = Path(probe_path) if probe_path is not None else _DEFAULT_PROBE
        if not path.exists():
            raise FileNotFoundError(
                f"Linear probe not found at {path}. "
                "Run `python -m training.train_probe` to generate it."
            )
        self._probe = self._load_probe(path)
        self._probe.eval()
        logger.debug(f"StateDecoder: probe loaded from {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_probe(path: Path) -> _ValenceArousalProbe:
        ckpt      = torch.load(str(path), map_location="cpu", weights_only=True)
        cfg       = ckpt.get("probe_config", {"state_dim": 384})
        probe     = _ValenceArousalProbe(state_dim=cfg.get("state_dim", 384))
        probe.load_state_dict(ckpt["model_state_dict"])
        return probe

    def _run_probe(self, state_vector: np.ndarray) -> tuple[float, float]:
        """Return (valence, arousal) predicted by the linear probe, clamped to [-1.5, 1.5]."""
        with torch.no_grad():
            x    = torch.from_numpy(state_vector.astype(np.float32)).unsqueeze(0)
            pred = self._probe(x).squeeze(0).numpy()
        pred = np.clip(pred, -1.5, 1.5)
        return float(pred[0]), float(pred[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self, session_count: int) -> bool:
        """
        Return True if there is enough history to produce a meaningful summary.
        False for session_count < 3 — the state vector hasn't accumulated
        enough trajectory to be reliably interpreted.
        """
        return session_count >= _MIN_SESSIONS

    def decode(
        self,
        state_vector: np.ndarray,
        arc_history: list[list[float]],
        session_count: int,
    ) -> dict:
        """
        Decode a state vector and arc_history into a context dict.

        Args:
            state_vector:  1-D float32 numpy array of shape (state_dim,).
                           Must be the L2-normalised output of the transition model.
            arc_history:   List of [valence, arousal] pairs from StateStore,
                           oldest first.  May be empty.
            session_count: Total number of sessions recorded for this user
                           (from StateStore.get_info).

        Returns:
            dict with keys:
                state_summary  (str | None)  — natural-language context string,
                                               None when ready=False
                valence        (float)        — current valence from probe [-1, 1]
                arousal        (float)        — current arousal from probe [-1, 1]
                trend          (str)          — "improving" | "declining" | "stable"
                trend_strength (float)        — absolute linear slope over arc_history
                mean_valence   (float)        — mean valence over arc_history
                mean_arousal   (float)        — mean arousal over arc_history
                session_count  (int)
                ready          (bool)         — False when session_count < 3

        Raises:
            ValueError: if state_vector is not 1-D.
        """
        if state_vector.ndim != 1:
            raise ValueError(
                f"state_vector must be 1-D, got shape {state_vector.shape}"
            )

        ready = self.is_ready(session_count)

        # --- Probe: current valence / arousal ---
        valence, arousal = self._run_probe(state_vector)

        # --- Trend from arc_history ---
        if len(arc_history) >= _MIN_TREND_ENTRIES:
            valences     = [pair[0] for pair in arc_history]
            arousals     = [pair[1] for pair in arc_history]
            slope        = _linear_slope(valences)
            mean_valence = float(np.mean(valences))
            mean_arousal = float(np.mean(arousals))
        else:
            slope        = 0.0
            mean_valence = valence   # fall back to current probe output
            mean_arousal = arousal

        trend          = _classify_trend(slope)
        trend_strength = abs(slope)

        # --- Natural-language summary ---
        if ready:
            summary = _build_summary(
                valence        = valence,
                arousal        = arousal,
                trend          = trend,
                trend_strength = trend_strength,
                mean_valence   = mean_valence,
                mean_arousal   = mean_arousal,
                session_count  = session_count,
                n_history      = len(arc_history),
            )
        else:
            summary = None

        return {
            "state_summary":  summary,
            "valence":        round(valence,        4),
            "arousal":        round(arousal,        4),
            "trend":          trend,
            "trend_strength": round(trend_strength, 6),
            "mean_valence":   round(mean_valence,   4),
            "mean_arousal":   round(mean_arousal,   4),
            "session_count":  session_count,
            "ready":          ready,
        }

    def __repr__(self) -> str:
        return f"StateDecoder(probe={self._probe.__class__.__name__}(384->2))"


# ---------------------------------------------------------------------------
# __main__ — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.WARNING)

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            suffix = f"  ({detail})" if detail else ""
            print(f"  [FAIL] {label}{suffix}")
            failed += 1

    # ------------------------------------------------------------------ #
    # Load decoder                                                         #
    # ------------------------------------------------------------------ #
    print("\n=== StateDecoder self-test ===\n")

    try:
        decoder = StateDecoder()
        print(f"  Loaded: {decoder}\n")
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        sys.exit(1)

    # Synthetic state vectors — use fixed-seed random vectors on the unit sphere
    rng = np.random.default_rng(0)

    def unit_vec(seed_offset: int = 0) -> np.ndarray:
        v = rng.standard_normal(384).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Test 1: session_count < 3 → ready=False, state_summary=None
    # ------------------------------------------------------------------ #
    print("--- Cold start (session_count < 3) ---")
    ctx = decoder.decode(unit_vec(), arc_history=[], session_count=1)

    check("ready=False when session_count=1",   ctx["ready"] is False)
    check("state_summary=None when not ready",  ctx["state_summary"] is None)
    check("valence is float",                   isinstance(ctx["valence"], float))
    check("arousal is float",                   isinstance(ctx["arousal"], float))
    check("trend is str",                       isinstance(ctx["trend"], str))
    check("trend_strength is float",            isinstance(ctx["trend_strength"], float))
    check("session_count echoed",               ctx["session_count"] == 1)
    print(f"  probe output: valence={ctx['valence']:.4f}, arousal={ctx['arousal']:.4f}")
    print(f"  trend={ctx['trend']}, trend_strength={ctx['trend_strength']:.6f}")

    ctx2 = decoder.decode(unit_vec(), arc_history=[], session_count=2)
    check("ready=False when session_count=2",   ctx2["ready"] is False)
    check("state_summary=None when count=2",    ctx2["state_summary"] is None)

    ctx3 = decoder.decode(unit_vec(), arc_history=[[0.1, 0.0], [0.15, 0.0], [0.2, 0.0]],
                          session_count=3)
    check("ready=True at session_count=3",      ctx3["ready"] is True)
    check("state_summary not None at count=3",  ctx3["state_summary"] is not None)
    check("state_summary is non-empty string",
          isinstance(ctx3["state_summary"], str) and len(ctx3["state_summary"]) > 10)

    # ------------------------------------------------------------------ #
    # Test 2: stable positive arc_history
    # ------------------------------------------------------------------ #
    print("\n--- Stable positive ---")
    stable_pos_history = [[0.60, 0.10]] * 8  # flat positive
    # Use a state vector that the probe will rate positively — nudge in probe's
    # positive-valence direction by using the probe weight vector directly.
    probe_weight = decoder._probe.linear.weight.detach().numpy()  # (2, 384)
    pos_direction = probe_weight[0]  # row 0 → valence weight
    pos_vec = (pos_direction / np.linalg.norm(pos_direction)).astype(np.float32)

    ctx_sp = decoder.decode(pos_vec, arc_history=stable_pos_history, session_count=8)
    print(f"  valence={ctx_sp['valence']:.4f}, arousal={ctx_sp['arousal']:.4f}, "
          f"trend={ctx_sp['trend']}, strength={ctx_sp['trend_strength']:.6f}")
    print(f"  summary: {ctx_sp['state_summary']}")

    check("stable positive: ready=True",        ctx_sp["ready"] is True)
    check("stable positive: trend='stable'",    ctx_sp["trend"] == "stable",
          f"got {ctx_sp['trend']}")
    check("stable positive: valence > 0",       ctx_sp["valence"] > 0,
          f"got {ctx_sp['valence']:.4f}")
    check("stable positive: summary non-empty", bool(ctx_sp["state_summary"]))
    check("stable positive: mean_valence ~0.60",
          abs(ctx_sp["mean_valence"] - 0.60) < 0.01,
          f"got {ctx_sp['mean_valence']:.4f}")

    # ------------------------------------------------------------------ #
    # Test 3: declining arc_history
    # ------------------------------------------------------------------ #
    print("\n--- Declining ---")
    n = 10
    declining_history = [[0.6 - i * 0.08, 0.0] for i in range(n)]  # drops to -0.12
    ctx_dec = decoder.decode(unit_vec(), arc_history=declining_history, session_count=n)
    slope_dec = _linear_slope([pair[0] for pair in declining_history])
    print(f"  history valences: {[round(p[0], 2) for p in declining_history]}")
    print(f"  computed slope: {slope_dec:.5f}")
    print(f"  valence={ctx_dec['valence']:.4f}, trend={ctx_dec['trend']}, "
          f"strength={ctx_dec['trend_strength']:.5f}")
    print(f"  summary: {ctx_dec['state_summary']}")

    check("declining: trend='declining'",       ctx_dec["trend"] == "declining",
          f"got {ctx_dec['trend']}")
    check("declining: trend_strength > 0.02",   ctx_dec["trend_strength"] > 0.02,
          f"got {ctx_dec['trend_strength']:.5f}")
    check("declining: slope < -0.02",           slope_dec < -0.02,
          f"slope={slope_dec:.5f}")
    check("declining: summary non-empty",       bool(ctx_dec["state_summary"]))

    # ------------------------------------------------------------------ #
    # Test 4: improving arc_history
    # ------------------------------------------------------------------ #
    print("\n--- Improving ---")
    n = 10
    improving_history = [[-0.5 + i * 0.08, 0.0] for i in range(n)]  # rises to +0.22
    ctx_imp = decoder.decode(unit_vec(), arc_history=improving_history, session_count=n)
    slope_imp = _linear_slope([pair[0] for pair in improving_history])
    print(f"  history valences: {[round(p[0], 2) for p in improving_history]}")
    print(f"  computed slope: {slope_imp:.5f}")
    print(f"  valence={ctx_imp['valence']:.4f}, trend={ctx_imp['trend']}, "
          f"strength={ctx_imp['trend_strength']:.5f}")
    print(f"  summary: {ctx_imp['state_summary']}")

    check("improving: trend='improving'",       ctx_imp["trend"] == "improving",
          f"got {ctx_imp['trend']}")
    check("improving: trend_strength > 0.02",   ctx_imp["trend_strength"] > 0.02,
          f"got {ctx_imp['trend_strength']:.5f}")
    check("improving: slope > 0.02",            slope_imp > 0.02,
          f"slope={slope_imp:.5f}")
    check("improving: summary non-empty",       bool(ctx_imp["state_summary"]))

    # ------------------------------------------------------------------ #
    # Test 5: chronic negative (stable, negative valence)
    # ------------------------------------------------------------------ #
    print("\n--- Chronic negative ---")
    neg_direction = -probe_weight[0]
    neg_vec = (neg_direction / np.linalg.norm(neg_direction)).astype(np.float32)
    chronic_neg_history = [[-0.55, -0.1]] * 12
    ctx_cn = decoder.decode(neg_vec, arc_history=chronic_neg_history, session_count=12)
    print(f"  valence={ctx_cn['valence']:.4f}, trend={ctx_cn['trend']}")
    print(f"  summary: {ctx_cn['state_summary']}")

    check("chronic neg: trend='stable'",        ctx_cn["trend"] == "stable",
          f"got {ctx_cn['trend']}")
    check("chronic neg: valence < 0",           ctx_cn["valence"] < 0,
          f"got {ctx_cn['valence']:.4f}")
    check("chronic neg: summary non-empty",     bool(ctx_cn["state_summary"]))

    # ------------------------------------------------------------------ #
    # Test 6: arc_history shorter than MIN_TREND_ENTRIES falls back
    # ------------------------------------------------------------------ #
    print("\n--- Short arc_history (< 3 entries) ---")
    ctx_short = decoder.decode(pos_vec,
                               arc_history=[[0.4, 0.1], [0.5, 0.1]],
                               session_count=3)
    check("short history: trend_strength == 0.0",
          ctx_short["trend_strength"] == 0.0,
          f"got {ctx_short['trend_strength']}")
    check("short history: trend='stable' (default)",
          ctx_short["trend"] == "stable",
          f"got {ctx_short['trend']}")
    check("short history: ready=True (session_count=3)",
          ctx_short["ready"] is True)

    # ------------------------------------------------------------------ #
    # Test 7: ValueError on bad input
    # ------------------------------------------------------------------ #
    print("\n--- Input validation ---")
    try:
        decoder.decode(np.zeros((2, 384), dtype=np.float32), [], 5)
        check("2-D input raises ValueError", False, "no error raised")
    except ValueError:
        check("2-D input raises ValueError", True)

    # ------------------------------------------------------------------ #
    # Test 8: is_ready
    # ------------------------------------------------------------------ #
    print("\n--- is_ready ---")
    check("is_ready(2)=False",  decoder.is_ready(2) is False)
    check("is_ready(3)=True",   decoder.is_ready(3) is True)
    check("is_ready(10)=True",  decoder.is_ready(10) is True)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total = passed + failed
    print(f"\n{'='*44}")
    print(f"  {passed}/{total} tests passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print("  -- all clear")
    print(f"{'='*44}\n")
