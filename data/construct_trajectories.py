"""
Synthetic emotional trajectory construction.

Takes a pool of LabeledSession objects (from prepare.py) and arc templates
(from arc_templates.py), and constructs synthetic multi-session trajectories
as if they came from a single user over N weeks.

Output format (one trajectory):
{
    "trajectory_id": "traj_0042",
    "arc_name":      "slow_recovery",
    "n_sessions":    9,
    "sessions": [
        {
            "session_index": 0,
            "week_offset":   0,
            "conv_id":       "hit:5_conv:2",          # source conversation
            "emotion_label": "devastated",             # for debugging
            "valence":       -0.80,
            "arousal":       -0.30,
            "turns": [{"role": "user", "content": "..."}, ...]
        },
        ...
    ]
}

The trajectory_id is stable across runs (seeded RNG). Each trajectory is
constructed such that:
  - Session t+1's emotional zone matches arc_template.sessions[t+1]
  - The same source conversation is NEVER reused within a single trajectory
  - The same source conversation is not overused across trajectories
    (soft cap: each conv_id appears in at most max_uses_per_conv trajectories)

This file is self-contained — run it directly to generate and inspect
trajectories without touching any model code.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data.arc_templates import ARC_TEMPLATES, ArcTemplate, CircumplexZone
from data.prepare import LabeledSession

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySession:
    session_index: int
    week_offset: int
    conv_id: str
    emotion_label: str
    valence: float
    arousal: float
    turns: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    trajectory_id: str
    arc_name: str
    n_sessions: int
    sessions: list[TrajectorySession]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "arc_name": self.arc_name,
            "n_sessions": self.n_sessions,
            "sessions": [s.to_dict() for s in self.sessions],
        }


def _sessions_for_zone(
    pool: list[LabeledSession],
    zone: CircumplexZone,
    exclude: set[str],
    usage_counts: dict[str, int],
    max_uses: int,
) -> list[LabeledSession]:
    """Return pool sessions that fall within zone, excluding already-used IDs."""
    return [
        s for s in pool
        if zone.contains(s.valence, s.arousal)
        and s.conv_id not in exclude
        and usage_counts[s.conv_id] < max_uses
    ]


def _week_offsets(
    n_sessions: int,
    duration_weeks: tuple[int, int],
    rng: random.Random,
) -> list[int]:
    """
    Assign week offsets to sessions, spreading them across the arc duration.
    Sessions are not perfectly evenly spaced — slight random jitter models
    irregular conversation cadence.
    """
    total_weeks = rng.randint(*duration_weeks)
    if n_sessions == 1:
        return [0]

    # Evenly space sessions then jitter by ±0-2 weeks
    spacing = total_weeks / (n_sessions - 1)
    offsets = []
    for i in range(n_sessions):
        base = int(round(i * spacing))
        jitter = rng.randint(-min(1, base), 2) if i > 0 else 0
        offsets.append(max(0, base + jitter))

    # Ensure strictly increasing
    for i in range(1, len(offsets)):
        if offsets[i] <= offsets[i - 1]:
            offsets[i] = offsets[i - 1] + 1

    return offsets


def construct_trajectories(
    pool: list[LabeledSession],
    n_trajectories: int = 5000,
    max_uses_per_conv: int = 8,
    seed: int = 42,
    fallback_tolerance_boost: float = 0.15,
) -> list[Trajectory]:
    """
    Construct synthetic multi-session emotional trajectories.

    Args:
        pool:                   LabeledSession pool from prepare.py
        n_trajectories:         How many trajectories to generate
        max_uses_per_conv:      Soft cap on reuse of a single source conversation
        seed:                   RNG seed for reproducibility
        fallback_tolerance_boost: If a zone has no candidates, expand tolerance by
                                  this amount and retry once before skipping.

    Returns:
        List of Trajectory objects.
    """
    rng = random.Random(seed)
    usage_counts: dict[str, int] = defaultdict(int)

    # Build weighted template list for sampling
    weighted_templates: list[ArcTemplate] = []
    for t in ARC_TEMPLATES:
        weighted_templates.extend([t] * max(1, int(t.weight * 10)))

    trajectories: list[Trajectory] = []
    failed_arcs: dict[str, int] = defaultdict(int)

    for traj_idx in range(n_trajectories):
        template = rng.choice(weighted_templates)
        used_in_this_traj: set[str] = set()
        sessions: list[TrajectorySession] = []
        failed = False

        for step_idx, zone in enumerate(template.sessions):
            candidates = _sessions_for_zone(
                pool, zone, used_in_this_traj, usage_counts, max_uses_per_conv
            )

            # Fallback: relax tolerance once if no candidates found
            if not candidates:
                relaxed_zone = CircumplexZone(
                    valence=zone.valence,
                    arousal=zone.arousal,
                    tolerance=zone.tolerance + fallback_tolerance_boost,
                    label=zone.label,
                )
                candidates = _sessions_for_zone(
                    pool, relaxed_zone, used_in_this_traj, usage_counts, max_uses_per_conv
                )

            if not candidates:
                logger.debug(
                    f"  No candidates for arc '{template.name}' step {step_idx} "
                    f"(zone {zone.label!r}, v={zone.valence:+.2f}, a={zone.arousal:+.2f}). "
                    f"Skipping trajectory."
                )
                failed = True
                failed_arcs[template.name] += 1
                break

            chosen = rng.choice(candidates)
            used_in_this_traj.add(chosen.conv_id)
            usage_counts[chosen.conv_id] += 1

            sessions.append(TrajectorySession(
                session_index=step_idx,
                week_offset=0,  # filled in below
                conv_id=chosen.conv_id,
                emotion_label=chosen.emotion_label,
                valence=chosen.valence,
                arousal=chosen.arousal,
                turns=chosen.turns,
            ))

        if failed or len(sessions) < 2:
            continue

        # Assign realistic week offsets
        offsets = _week_offsets(len(sessions), template.typical_duration_weeks, rng)
        for sess, offset in zip(sessions, offsets):
            sess.week_offset = offset

        trajectories.append(Trajectory(
            trajectory_id=f"traj_{traj_idx:06d}",
            arc_name=template.name,
            n_sessions=len(sessions),
            sessions=sessions,
        ))

    logger.info(
        f"Constructed {len(trajectories)} trajectories "
        f"({n_trajectories - len(trajectories)} failed)"
    )
    if failed_arcs:
        logger.info("  Failed arc counts: " + str(dict(failed_arcs)))

    return trajectories


def trajectory_statistics(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Compute summary statistics over the trajectory set."""
    import statistics

    arc_counts: dict[str, int] = defaultdict(int)
    session_lengths: list[int] = []
    turn_counts_flat: list[int] = []

    for traj in trajectories:
        arc_counts[traj.arc_name] += 1
        session_lengths.append(traj.n_sessions)
        for sess in traj.sessions:
            turn_counts_flat.append(len(sess.turns))

    return {
        "total_trajectories": len(trajectories),
        "arc_distribution": dict(sorted(arc_counts.items(), key=lambda x: -x[1])),
        "sessions_per_trajectory_mean": statistics.mean(session_lengths),
        "sessions_per_trajectory_min": min(session_lengths),
        "sessions_per_trajectory_max": max(session_lengths),
        "turns_per_session_mean": statistics.mean(turn_counts_flat),
    }


def print_trajectory_statistics(trajectories: list[Trajectory]) -> None:
    stats = trajectory_statistics(trajectories)
    print(f"\n=== Trajectory Statistics ===")
    print(f"Total trajectories:       {stats['total_trajectories']}")
    print(f"Sessions/trajectory:      {stats['sessions_per_trajectory_mean']:.1f} "
          f"(min {stats['sessions_per_trajectory_min']}, "
          f"max {stats['sessions_per_trajectory_max']})")
    print(f"Turns/session (mean):     {stats['turns_per_session_mean']:.1f}")
    print(f"\nArc distribution:")
    for arc, count in stats["arc_distribution"].items():
        bar = "#" * (count // 20)
        print(f"  {arc:<30} {count:>5}  {bar}")


def print_sample_trajectory(traj: Trajectory) -> None:
    print(f"\n{'='*60}")
    print(f"Trajectory: {traj.trajectory_id}  |  Arc: {traj.arc_name}")
    print(f"Sessions: {traj.n_sessions}")
    print(f"{'='*60}")
    for sess in traj.sessions:
        print(
            f"\n  [Week {sess.week_offset:>3}] Session {sess.session_index}  "
            f"emotion={sess.emotion_label:<16} "
            f"v={sess.valence:+.2f}  a={sess.arousal:+.2f}"
        )
        for i, turn in enumerate(sess.turns[:2]):
            role_tag = "U:" if turn["role"] == "user" else "A:"
            print(f"    {role_tag} {turn['content'][:90]}")
        if len(sess.turns) > 2:
            print(f"    ... ({len(sess.turns) - 2} more turns)")


def save_trajectories(
    trajectories: list[Trajectory],
    output_path: str | Path,
    indent: int | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in trajectories], f, indent=indent)
    logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")


def load_trajectories(input_path: str | Path) -> list[dict[str, Any]]:
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate synthetic emotional trajectory sequences."
    )
    parser.add_argument(
        "n_trajectories", type=int, nargs="?", default=500,
        help="Number of trajectories to generate (default: 500)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=(
            "Output JSON path. Defaults to data/trajectories_sample.json "
            "for n<=500 and data/trajectories_{n//1000}k.json otherwise."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-uses", type=int, default=8,
        help="Max times a source conversation can appear across trajectories (default: 8)",
    )
    parser.add_argument(
        "--sample", type=int, default=3,
        help="Number of sample trajectories to print (default: 3)",
    )
    parser.add_argument(
        "--augment-low-arousal", action="store_true",
        help="Extend the session pool with synthetic deep-low-arousal sessions "
             "(data/low_arousal_pool.py). Breaks the EmpatheticDialogues arousal "
             "floor of ~-0.30 so exhausted/depressed/shutdown/serene zones sample "
             "genuinely deactivated content instead of falling back.",
    )
    parser.add_argument(
        "--n-low-arousal", type=int, default=600,
        help="Number of synthetic low-arousal sessions to add (default: 600)",
    )
    parser.add_argument(
        "--holdout-conv-fraction", type=float, default=0.0,
        help="If > 0, reserve this fraction of source CONVERSATIONS before "
             "construction and build a separate validation trajectory file "
             "(<out>_val.json) exclusively from them. This is the "
             "leakage-free-by-construction alternative to post-hoc splitting: "
             "train and val trajectories share zero source conversations and "
             "no generated trajectories are wasted.",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.out is not None:
        out = Path(args.out)
    elif args.n_trajectories <= 500:
        out = Path("data/trajectories_sample.json")
    else:
        label = f"{args.n_trajectories // 1000}k"
        out = Path(f"data/trajectories_{label}.json")

    print("Loading session pool...")
    from data.prepare import load_empathetic_dialogues, print_pool_statistics
    pool = load_empathetic_dialogues("train")
    if args.augment_low_arousal:
        from data.low_arousal_pool import load_low_arousal_sessions
        extra = load_low_arousal_sessions(n=args.n_low_arousal, seed=args.seed)
        pool += extra
        print(f"Augmented pool with {len(extra)} synthetic deep-low-arousal sessions "
              f"(arousal down to -0.80; ED floor was ~-0.30).")
    print_pool_statistics(pool)

    val_pool: list = []
    if args.holdout_conv_fraction > 0:
        rng_split = random.Random(args.seed)
        shuffled = list(pool)
        rng_split.shuffle(shuffled)
        n_val_pool = int(len(shuffled) * args.holdout_conv_fraction)
        val_pool = shuffled[:n_val_pool]
        pool     = shuffled[n_val_pool:]
        print(f"Held out {len(val_pool)} conversations for the val trajectory file "
              f"({args.holdout_conv_fraction:.0%} of pool); train pool: {len(pool)}")

    print(f"\nConstructing {args.n_trajectories} trajectories (seed={args.seed})...")
    trajectories = construct_trajectories(
        pool,
        n_trajectories=args.n_trajectories,
        max_uses_per_conv=args.max_uses,
        seed=args.seed,
    )
    print_trajectory_statistics(trajectories)

    if val_pool:
        n_val_trajs = max(50, args.n_trajectories // 5)
        print(f"\nConstructing {n_val_trajs} VAL trajectories from held-out conversations...")
        val_trajectories = construct_trajectories(
            val_pool,
            n_trajectories=n_val_trajs,
            max_uses_per_conv=args.max_uses,
            seed=args.seed + 1,
        )
        print_trajectory_statistics(val_trajectories)

    if args.sample > 0:
        print(f"\n--- {args.sample} sample trajectories ---")
        rng = random.Random(0)
        samples = rng.sample(trajectories, min(args.sample, len(trajectories)))
        for t in samples:
            print_sample_trajectory(t)

    save_trajectories(trajectories, out, indent=None)   # no indent = smaller file
    print(f"\nSaved to {out}")

    if val_pool:
        val_out = out.with_name(out.stem + "_val" + out.suffix)
        save_trajectories(val_trajectories, val_out, indent=None)
        print(f"Saved conversation-disjoint val trajectories to {val_out}")
