"""
diagnostics/per_arc_val_loss.py

Compute validation cosine sim loss broken down per arc template.

Flags if the 3 new arousal-primary arcs (anxiety_to_depression,
depression_to_anxiety, excitement_to_contentment) show materially higher
loss than valence arcs like gradual_decline — which would indicate the
transition model still can't predict along the arousal axis.

"Materially higher" threshold: > 0.05 above the mean valence-arc loss.

Run from project root:
    python -m diagnostics.per_arc_val_loss
    python -m diagnostics.per_arc_val_loss --checkpoint checkpoints/transition_v2.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.transition import TransitionModel
from diagnostics.common import (
    load_transition_model,
    make_encoder,
    build_embedding_cache as _shared_embedding_cache,
)

from kokoro.transition import predict_from_state

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Arc classification: which arcs are arousal-primary vs valence-primary
AROUSAL_PRIMARY = {
    "anxiety_to_depression", "depression_to_anxiety", "excitement_to_contentment",
    "depressive_shutdown", "unwinding_to_serenity",
}
VALENCE_PRIMARY = {
    "gradual_decline", "slow_recovery", "grief_arc", "stable_positive",
    "stable_negative", "post_traumatic_growth", "social_confidence_growth",
}

FLAG_THRESHOLD = 0.05   # arousal arc loss > mean(valence arc loss) + this → flag


def per_arc_sim_loss(
    model: TransitionModel,
    trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, list[float]]:
    """Return per-arc lists of trajectory-level cosine sim losses."""
    arc_losses: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for traj in trajectories:
            embs = [emb_cache[s["conv_id"]] for s in traj["sessions"]]
            if len(embs) < 2:
                continue
            state = model.initial_state(device=device)
            total = 0.0
            for t in range(len(embs) - 1):
                state = model(state, embs[t])
                state_n = F.normalize(predict_from_state(model, state), dim=-1)
                # MiniLM portion only — embeddings may carry 3 extra VAD dims
                target  = F.normalize(embs[t + 1][:model.state_dim], dim=-1)
                total  += (1.0 - (state_n * target).sum()).item()
            arc_losses[traj["arc_name"]].append(total / (len(embs) - 1))
    return dict(arc_losses)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Per-arc validation loss breakdown")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "trajectories_10k_v2_val.json"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "transition_v2.pt"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    data_path = Path(args.data)
    for p in [ckpt_path, data_path]:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    with open(data_path) as f:
        all_trajs = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)
    n_val = int(len(all_trajs) * args.val_fraction)
    val_trajs = all_trajs[-n_val:]
    logger.info(f"Val set: {n_val} trajectories")

    device = torch.device("cpu")
    model, cfg = load_transition_model(ckpt_path)
    encoder    = make_encoder(cfg)

    emb_cache = _shared_embedding_cache(val_trajs, encoder)
    arc_losses = per_arc_sim_loss(model, val_trajs, emb_cache, device)

    # --- Compute per-arc mean losses ---
    arc_means = {arc: float(np.mean(losses)) for arc, losses in arc_losses.items()}
    arc_stds  = {arc: float(np.std(losses))  for arc, losses in arc_losses.items()}

    # Valence-arc reference mean (only arcs present in val set)
    valence_losses = [arc_means[a] for a in VALENCE_PRIMARY if a in arc_means]
    valence_mean   = float(np.mean(valence_losses)) if valence_losses else float("nan")
    overall_mean   = float(np.mean(list(arc_means.values())))

    # --- Print ranked table ---
    print()
    print("=" * 70)
    print("  Per-arc validation cosine sim loss  (lower = better prediction)")
    print("=" * 70)
    print(f"  {'Arc':<35} {'Mean loss':>10}  {'Std':>8}  {'N':>5}  {'Type':<8}")
    print("-" * 70)

    for arc, mean_loss in sorted(arc_means.items(), key=lambda x: x[1]):
        n   = len(arc_losses[arc])
        std = arc_stds[arc]
        if arc in AROUSAL_PRIMARY:
            arc_type = "AROUSAL"
        elif arc in VALENCE_PRIMARY:
            arc_type = "valence"
        else:
            arc_type = "mixed"
        flag = "  *** HIGH ***" if (arc in AROUSAL_PRIMARY and mean_loss > valence_mean + FLAG_THRESHOLD) else ""
        print(f"  {arc:<35} {mean_loss:>10.4f}  {std:>8.4f}  {n:>5}  {arc_type:<8}{flag}")

    print("-" * 70)
    print(f"  {'Overall mean':<35} {overall_mean:>10.4f}")
    print(f"  {'Valence-arc mean (reference)':<35} {valence_mean:>10.4f}")
    print()

    # --- Arousal vs valence gap ---
    arousal_losses = [arc_means[a] for a in AROUSAL_PRIMARY if a in arc_means]
    arousal_mean   = float(np.mean(arousal_losses)) if arousal_losses else float("nan")
    gap = arousal_mean - valence_mean

    print(f"  Arousal-primary arc mean loss:  {arousal_mean:.4f}")
    print(f"  Valence-primary arc mean loss:  {valence_mean:.4f}")
    print(f"  Gap (arousal - valence):        {gap:+.4f}")
    print()

    if not np.isnan(gap):
        if gap > FLAG_THRESHOLD:
            print(f"  FLAG: arousal arcs are {gap:.4f} above valence arcs (threshold {FLAG_THRESHOLD}).")
            print(f"  The model is still struggling to predict along the arousal axis.")
            print(f"  Consider: stronger var_weight, more arousal-primary data, or better encoder.")
        elif gap > 0:
            print(f"  OK: arousal arcs are {gap:.4f} above valence arcs — small gap, within threshold.")
            print(f"  The model predicts arousal trajectories nearly as well as valence ones.")
        else:
            print(f"  GOOD: arousal arcs have lower loss than valence arcs — arousal is well-learned.")

    print("=" * 70)


if __name__ == "__main__":
    main()
