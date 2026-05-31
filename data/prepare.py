"""
Dataset loading and preprocessing for Nasmri training data.

Loads facebook/empathetic_dialogues and returns a pool of LabeledSession
objects — each session has:
  - turns: list of {role, content} dicts (the conversation text)
  - valence: float in [-1, 1]   (from circumplex mapping of emotion label)
  - arousal: float in [-1, 1]   (from circumplex mapping of emotion label)
  - emotion_label: str           (original label, kept for debugging only)

The circumplex coordinates are derived from Russell (1980) and the
meta-analysis by Posner, Russell & Peterson (2005, Acta Psychologica).
They are used only to assign conversations to arc positions during
trajectory construction — never fed into the model as supervision targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circumplex mapping for EmpatheticDialogues' 32 emotion labels
#
# Sources:
#   Russell JA (1980). "A circumplex model of affect." J Personality & Social Psychology.
#   Posner J, Russell JA, Peterson BS (2005). "The circumplex model of affect."
#       Acta Psychologica 117(1):37–64.
#   Mohammad SM (2018). "Obtaining reliable human ratings of valence, arousal, and
#       dominance for 20,000 English words." ACL.
#
# Values are approximate centroids; the tolerance in CircumplexZone handles
# the fuzziness of these placements.
# ---------------------------------------------------------------------------

EMOTION_TO_CIRCUMPLEX: dict[str, tuple[float, float]] = {
    # label            valence  arousal
    # — confirmed against the actual facebook/empathetic_dialogues train split —
    # Quadrant I: positive affect, high activation
    "joyful":         (+0.85,  +0.65),
    "excited":        (+0.65,  +0.75),
    "proud":          (+0.70,  +0.50),
    "hopeful":        (+0.55,  +0.30),
    "impressed":      (+0.60,  +0.40),
    "anticipating":   (+0.45,  +0.45),
    "surprised":      (+0.20,  +0.70),  # valence depends on context; centered neutral
    "prepared":       (+0.40,  +0.20),
    "confident":      (+0.65,  +0.35),

    # Quadrant II: positive affect, low activation
    "content":        (+0.55,  -0.25),
    "grateful":       (+0.65,  -0.05),
    "trusting":       (+0.55,  -0.15),
    "caring":         (+0.60,  -0.10),
    "faithful":       (+0.55,  -0.20),
    "nostalgic":      (+0.40,  -0.10),
    "sentimental":    (+0.35,  -0.15),

    # Quadrant III: negative affect, high activation
    "apprehensive":   (-0.30,  +0.40),
    "anxious":        (-0.45,  +0.55),
    "afraid":         (-0.55,  +0.60),
    "terrified":      (-0.80,  +0.80),
    "angry":          (-0.50,  +0.70),
    "furious":        (-0.70,  +0.80),
    "annoyed":        (-0.35,  +0.55),
    "disgusted":      (-0.55,  +0.35),
    "jealous":        (-0.40,  +0.30),

    # Quadrant IV: negative affect, low activation
    "sad":            (-0.55,  -0.25),
    "devastated":     (-0.80,  -0.30),
    "lonely":         (-0.60,  -0.20),
    "ashamed":        (-0.60,  -0.10),
    "embarrassed":    (-0.50,  -0.05),
    "guilty":         (-0.55,  -0.15),
    "disappointed":   (-0.50,  -0.20),
}

# Exact label set from facebook/empathetic_dialogues (verified against train split)
_EXPECTED_LABELS = {
    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive",
    "ashamed", "caring", "confident", "content", "devastated", "disappointed",
    "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
    "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely",
    "nostalgic", "prepared", "proud", "sad", "sentimental", "surprised",
    "terrified", "trusting",
}
assert set(EMOTION_TO_CIRCUMPLEX.keys()) == _EXPECTED_LABELS, (
    f"Mapping mismatch.\n"
    f"  Extra:   {set(EMOTION_TO_CIRCUMPLEX.keys()) - _EXPECTED_LABELS}\n"
    f"  Missing: {_EXPECTED_LABELS - set(EMOTION_TO_CIRCUMPLEX.keys())}"
)


@dataclass
class LabeledSession:
    """A single conversation with its circumplex position."""
    conv_id: str
    turns: list[dict[str, str]]   # [{"role": "user"|"assistant", "content": "..."}]
    valence: float
    arousal: float
    emotion_label: str             # kept for debugging, not used in model


def _build_turns(group_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Convert EmpatheticDialogues rows for one conv_id into a list of turn dicts.

    Schema (from dataset source):
      utterance_idx: int — turn number (0-indexed)
      speaker_idx:   int — 0 = speaker (shares situation), 1 = listener (responds)
      utterance:     str — the turn text
    """
    rows = sorted(group_rows, key=lambda r: r["utterance_idx"])
    turns = []
    for row in rows:
        role = "user" if row["speaker_idx"] == 0 else "assistant"
        text = row["utterance"].strip()
        if text:
            turns.append({"role": role, "content": text})
    return turns


_ED_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
)
_ED_SPLIT_FILES = {
    "train": "empatheticdialogues/train.csv",
    "validation": "empatheticdialogues/valid.csv",
    "test": "empatheticdialogues/test.csv",
}
_ED_CACHE_DIR = Path.home() / ".cache" / "kokoro" / "empathetic_dialogues"


def _download_empathetic_dialogues() -> Path:
    """
    Download the EmpatheticDialogues tarball to a local cache directory.
    Returns the path to the extracted directory.

    The HF datasets loader for this dataset is a legacy loading script that
    newer versions of `datasets` refuse to run. We fetch the original
    source tarball directly from Facebook AI's servers instead.
    """
    import tarfile
    import urllib.request

    _ED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extracted_dir = _ED_CACHE_DIR / "empatheticdialogues"

    if extracted_dir.exists():
        logger.info(f"  Using cached data at {extracted_dir}")
        return extracted_dir

    tarball = _ED_CACHE_DIR / "empatheticdialogues.tar.gz"
    if not tarball.exists():
        logger.info(f"  Downloading {_ED_URL} ...")
        urllib.request.urlretrieve(_ED_URL, tarball)
        logger.info(f"  Downloaded to {tarball}")

    logger.info("  Extracting tarball...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(_ED_CACHE_DIR)

    return extracted_dir


def load_empathetic_dialogues(split: str = "train") -> list[LabeledSession]:
    """
    Load facebook/empathetic_dialogues, group by conv_id, and return a pool
    of LabeledSession objects with circumplex coordinates.

    Downloads the original tarball from Facebook AI's servers on first call;
    subsequent calls use the local cache in ~/.cache/kokoro/.

    Args:
        split: "train", "validation", or "test"

    Returns:
        List of LabeledSession, one per unique conversation.
    """
    import csv

    if split not in _ED_SPLIT_FILES:
        raise ValueError(f"split must be one of {list(_ED_SPLIT_FILES)}, got {split!r}")

    extracted_dir = _download_empathetic_dialogues()
    csv_path = _ED_CACHE_DIR / _ED_SPLIT_FILES[split]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} after extraction — tarball structure may have changed."
        )

    logger.info(f"Loading facebook/empathetic_dialogues ({split}) from {csv_path}")

    # Group rows by conv_id
    conversations: dict[str, list[dict]] = {}
    skipped_unknown_emotion = 0
    total_rows = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            label = row["context"].strip().lower()
            if label not in EMOTION_TO_CIRCUMPLEX:
                skipped_unknown_emotion += 1
                continue
            conv_id = row["conv_id"]
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(row)

    logger.info(f"  Raw rows: {total_rows}")
    if skipped_unknown_emotion:
        logger.warning(
            f"  Skipped {skipped_unknown_emotion} rows with unknown emotion labels"
        )

    # Build LabeledSession per conversation
    sessions: list[LabeledSession] = []
    empty_turn_count = 0

    for conv_id, rows in conversations.items():
        # Parse utterance_idx to int for sorting (CSV values are strings)
        for r in rows:
            r["utterance_idx"] = int(r["utterance_idx"])
            r["speaker_idx"] = int(r["speaker_idx"])

        turns = _build_turns(rows)
        if len(turns) < 2:
            empty_turn_count += 1
            continue

        label = rows[0]["context"].strip().lower()
        valence, arousal = EMOTION_TO_CIRCUMPLEX[label]

        sessions.append(LabeledSession(
            conv_id=conv_id,
            turns=turns,
            valence=valence,
            arousal=arousal,
            emotion_label=label,
        ))

    logger.info(
        f"  Loaded {len(sessions)} sessions "
        f"({empty_turn_count} skipped — fewer than 2 turns)"
    )
    return sessions


def load_mental_health_sessions() -> list[LabeledSession]:
    """
    Load Amod/mental_health_counseling_conversations as supplementary sessions.

    These are single Q&A pairs (not multi-turn), so we format them as
    2-turn conversations. They are placed in low-valence, moderate-arousal
    circumplex region because they represent help-seeking distress contexts.

    Since this dataset has no emotion labels, we assign a diffuse prior:
    valence in [-0.6, -0.2], arousal in [0.1, 0.5] — the distress region.
    The exact values are sampled per-example from a small random jitter so
    the model sees a spread rather than a hard cluster.
    """
    import random

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install 'datasets': pip install datasets") from e

    logger.info("Loading Amod/mental_health_counseling_conversations...")
    ds = load_dataset(
        "Amod/mental_health_counseling_conversations", split="train", trust_remote_code=True
    )
    logger.info(f"  Raw rows: {len(ds)}")

    sessions: list[LabeledSession] = []
    rng = random.Random(42)

    for i, row in enumerate(ds):
        context = (row.get("Context") or "").strip()
        response = (row.get("Response") or "").strip()
        if not context or not response:
            continue

        turns = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": response},
        ]

        # Jitter within the distress prior region
        valence = rng.uniform(-0.65, -0.20)
        arousal = rng.uniform(+0.10, +0.55)

        sessions.append(LabeledSession(
            conv_id=f"mhcc_{i}",
            turns=turns,
            valence=valence,
            arousal=arousal,
            emotion_label="distressed",  # synthetic label for debugging
        ))

    logger.info(f"  Loaded {len(sessions)} mental health sessions")
    return sessions


def pool_statistics(sessions: list[LabeledSession]) -> dict[str, Any]:
    """Return descriptive statistics about the session pool."""
    import statistics

    valences = [s.valence for s in sessions]
    arousals = [s.arousal for s in sessions]
    turn_counts = [len(s.turns) for s in sessions]

    label_dist: dict[str, int] = {}
    for s in sessions:
        label_dist[s.emotion_label] = label_dist.get(s.emotion_label, 0) + 1

    return {
        "total_sessions": len(sessions),
        "valence_mean": statistics.mean(valences),
        "valence_stdev": statistics.stdev(valences),
        "arousal_mean": statistics.mean(arousals),
        "arousal_stdev": statistics.stdev(arousals),
        "turns_mean": statistics.mean(turn_counts),
        "turns_min": min(turn_counts),
        "turns_max": max(turn_counts),
        "emotion_distribution": dict(
            sorted(label_dist.items(), key=lambda x: -x[1])
        ),
    }


def print_pool_statistics(sessions: list[LabeledSession]) -> None:
    stats = pool_statistics(sessions)
    print(f"\n=== Session Pool Statistics ===")
    print(f"Total sessions:  {stats['total_sessions']}")
    print(f"Valence:         {stats['valence_mean']:+.3f} ± {stats['valence_stdev']:.3f}")
    print(f"Arousal:         {stats['arousal_mean']:+.3f} ± {stats['arousal_stdev']:.3f}")
    print(f"Turns/session:   {stats['turns_mean']:.1f} (min {stats['turns_min']}, max {stats['turns_max']})")
    print(f"\nEmotion distribution (top 10):")
    for label, count in list(stats["emotion_distribution"].items())[:10]:
        bar = "#" * (count // 20)
        print(f"  {label:<18} {count:>5}  {bar}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sessions = load_empathetic_dialogues("train")
    print_pool_statistics(sessions)

    print(f"\n--- Sample session (first) ---")
    s = sessions[0]
    print(f"conv_id:  {s.conv_id}")
    print(f"emotion:  {s.emotion_label}  (v={s.valence:+.2f}, a={s.arousal:+.2f})")
    print(f"turns ({len(s.turns)}):")
    for t in s.turns[:4]:
        print(f"  [{t['role']}] {t['content'][:80]}...")
