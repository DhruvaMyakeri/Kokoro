"""
training/split.py — train/val splitting with conversation-level leakage control.

The original pipeline split at the TRAJECTORY level only. Because trajectory
construction reuses each source conversation in up to `max_uses_per_conv`
trajectories (data/construct_trajectories.py), the same EmpatheticDialogues
conversation — the exact same text, hence the exact same cached embedding —
routinely appears in both the train and val splits. Every published val metric
(val loss, probe Pearson r, participation ratio, arc separation) was therefore
computed on partially-seen inputs and is optimistically biased.

conv_disjoint_split() removes this: after the trajectory-level split, any val
trajectory that shares at least one conv_id with the train split is dropped
(train is kept intact so model capacity/coverage is unchanged). The function
also returns a leakage report so the before/after contamination is quantified
in the training log — reviewers can see exactly how much overlap existed.

Usage (train.py / train_probe.py):
    from training.split import conv_disjoint_split
    train_trajs, val_trajs, report = conv_disjoint_split(trajectories, 0.2, seed=42)
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


def _conv_ids(traj: dict[str, Any]) -> set[str]:
    return {s["conv_id"] for s in traj["sessions"]}


def leakage_report(
    train_trajs: list[dict[str, Any]],
    val_trajs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Quantify conversation-level overlap between two trajectory splits."""
    train_convs: set[str] = set()
    for t in train_trajs:
        train_convs |= _conv_ids(t)

    val_convs: set[str] = set()
    contaminated = 0
    for t in val_trajs:
        cids = _conv_ids(t)
        val_convs |= cids
        if cids & train_convs:
            contaminated += 1

    shared = train_convs & val_convs
    return {
        "n_train_trajectories":       len(train_trajs),
        "n_val_trajectories":         len(val_trajs),
        "n_train_convs":              len(train_convs),
        "n_val_convs":                len(val_convs),
        "n_shared_convs":             len(shared),
        "shared_conv_fraction_of_val": (len(shared) / len(val_convs)) if val_convs else 0.0,
        "n_contaminated_val_trajs":   contaminated,
        "contaminated_val_fraction":  (contaminated / len(val_trajs)) if val_trajs else 0.0,
    }


def conv_disjoint_split(
    trajectories: list[dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int = 42,
    enforce_disjoint: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split trajectories into train/val with optional conv-level disjointness.

    Because each source conversation is reused across up to max_uses_per_conv
    trajectories, a naive "drop contaminated val trajectories" filter removes
    the ENTIRE val set on the 10k dataset (measured: 100% of legacy val
    trajectories shared at least one conversation with train). So instead of
    filtering, this function PARTITIONS greedily:

      1. Shuffle trajectories (seeded).
      2. Assign each trajectory to the side (train/val) that its
         already-committed conversations belong to; if its conversations are
         split across both sides, drop it. Uncommitted trajectories go to
         whichever side is furthest below its quota. All of the trajectory's
         conversations are then committed to that side.

    Some trajectories are inevitably dropped (the conversation-reuse graph is
    dense); the report records how many. The resulting splits share ZERO
    source conversations.

    Returns (train, val, report). The report includes the legacy-split
    contamination numbers (computed on the same seed's naive 80/20 split) so
    the bias of previously published metrics is documented alongside.
    """
    trajs = list(trajectories)
    rng = random.Random(seed)
    rng.shuffle(trajs)

    # Document how contaminated the legacy naive split was (same shuffle/seed)
    split = int((1.0 - val_fraction) * len(trajs))
    report = leakage_report(trajs[:split], trajs[split:])
    report["enforced_disjoint"] = bool(enforce_disjoint)

    if not enforce_disjoint:
        train_trajs, val_trajs = trajs[:split], trajs[split:]
        logger.info(
            "LEGACY split: %d train / %d val | %.1f%% of val trajectories share "
            "conversations with train — val metrics are optimistically biased.",
            len(train_trajs), len(val_trajs),
            report["contaminated_val_fraction"] * 100,
        )
        return train_trajs, val_trajs, report

    val_trajs:   list[dict[str, Any]] = []
    train_trajs: list[dict[str, Any]] = []
    dropped = 0
    val_convs:   set[str] = set()
    train_convs: set[str] = set()

    for t in trajs:
        cids = _conv_ids(t)
        in_train = bool(cids & train_convs)
        in_val   = bool(cids & val_convs)
        if in_train and in_val:
            dropped += 1           # straddles both sides — unusable
            continue
        if in_train:
            side = "train"
        elif in_val:
            side = "val"
        else:
            # Fully uncommitted: send to whichever side is below quota
            n_assigned = len(train_trajs) + len(val_trajs)
            val_share  = (len(val_trajs) / n_assigned) if n_assigned else 0.0
            side = "val" if val_share < val_fraction else "train"
        if side == "val":
            val_trajs.append(t)
            val_convs |= cids
        else:
            train_trajs.append(t)
            train_convs |= cids

    report["n_val_after_partition"]   = len(val_trajs)
    report["n_train_after_partition"] = len(train_trajs)
    report["n_dropped"]               = dropped

    logger.info(
        "Conv-disjoint partition: %d train / %d val / %d dropped "
        "(legacy naive split had %.1f%% of val trajectories contaminated)",
        len(train_trajs), len(val_trajs), dropped,
        report["contaminated_val_fraction"] * 100,
    )
    if not val_trajs or not train_trajs:
        raise ValueError(
            "conv_disjoint_split could not build a non-empty disjoint partition — "
            "the conversation pool is too heavily reused. Regenerate trajectories "
            "with a lower --max-uses, or lower val_fraction."
        )
    return train_trajs, val_trajs, report
