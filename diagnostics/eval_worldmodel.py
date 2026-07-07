"""
World model prediction accuracy evaluation.

The transition model is trained with a next-session prediction objective:
  state_t = TransitionModel(state_{t-1}, e_t)
  loss    = 1 - cosine_sim(normalize(state_t), normalize(e_{t+1}))

This means state_t is a *prediction* of what e_{t+1} will look like, not just
a memory summary.  This script makes that prediction quality explicit:

  Model prediction:  cosine_sim(normalize(state_t), normalize(e_{t+1}))
  Naive baseline:    cosine_sim(normalize(e_t),     normalize(e_{t+1}))
                     ("just repeat the last session as prediction")
  Random baseline:   ~0.0 in expectation (high-dim unit sphere)

A model that outperforms the naive baseline is genuinely learning trajectory
dynamics, not just echoing recent sessions.

Results are broken down by arc type so you can see where prediction is hard
(e.g. volatile arcs) vs. easy (e.g. stable arcs).

Run from project root:
    python -m diagnostics.eval_worldmodel
    python -m diagnostics.eval_worldmodel --checkpoint checkpoints/transition_v2.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from kokoro.transition import predict_from_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_prediction_accuracy(
    model,
    trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:
    """
    For every consecutive pair (session_t, session_{t+1}) in the val set:

      model_sim[i]    = cosine_sim(normalize(state_t),  normalize(e_{t+1}[:state_dim]))
      baseline_sim[i] = cosine_sim(normalize(e_t[:state_dim]), normalize(e_{t+1}[:state_dim]))

    state_t is computed by rolling the trajectory from scratch each time
    (same protocol as training).

    Returns a dict with per-sample arrays and per-arc-type summaries.
    """
    from kokoro.transition import TransitionModel

    model.eval()
    state_dim = model.STATE_DIM

    model_sims:    list[float] = []
    baseline_sims: list[float] = []
    arc_names:     list[str]   = []

    with torch.no_grad():
        for traj in trajectories:
            sessions = traj["sessions"]
            arc      = traj.get("arc_name", "unknown")
            if len(sessions) < 2:
                continue

            embs  = [emb_cache[s["conv_id"]] for s in sessions]
            state = torch.zeros(state_dim, device=device)

            for t in range(len(embs) - 1):
                state = model(state, embs[t])

                # Model prediction: predict(state_t) vs e_{t+1}
                # (GRU decouples state from prediction; MLP falls through)
                z      = predict_from_state(model, state)
                s_norm = F.normalize(z,              dim=-1)
                e_norm = F.normalize(embs[t + 1][:state_dim], dim=-1)
                model_sims.append((s_norm * e_norm).sum().item())

                # Naive baseline: e_t vs e_{t+1}
                b_norm = F.normalize(embs[t][:state_dim], dim=-1)
                baseline_sims.append((b_norm * e_norm).sum().item())

                arc_names.append(arc)

    model_arr    = np.array(model_sims,    dtype=np.float32)
    baseline_arr = np.array(baseline_sims, dtype=np.float32)

    # Per-arc breakdown
    arcs_unique = sorted(set(arc_names))
    arc_results = {}
    for arc in arcs_unique:
        idx = [i for i, a in enumerate(arc_names) if a == arc]
        arc_results[arc] = {
            "n":            len(idx),
            "model_mean":   float(model_arr[idx].mean()),
            "baseline_mean": float(baseline_arr[idx].mean()),
            "delta":        float(model_arr[idx].mean() - baseline_arr[idx].mean()),
        }

    return {
        "model_sims":    model_arr,
        "baseline_sims": baseline_arr,
        "arc_names":     arc_names,
        "arc_results":   arc_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, n_val_trajs: int) -> None:
    model_arr    = results["model_sims"]
    baseline_arr = results["baseline_sims"]
    arc_results  = results["arc_results"]
    N            = len(model_arr)

    W = 70

    def hline(c="-"): print(c * W)

    print()
    hline("=")
    print("  World Model Prediction Accuracy")
    hline("=")
    print(f"  {'Val trajectories':<35} {n_val_trajs:>10,}")
    print(f"  {'Prediction pairs (t -> t+1)':<35} {N:>10,}")
    hline()
    print(f"  {'Metric':<35} {'Model':>10}  {'Baseline':>10}  {'Delta':>8}")
    hline()

    def row(label, m_val, b_val):
        delta = m_val - b_val
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<35} {m_val:>10.4f}  {b_val:>10.4f}  {sign}{delta:>7.4f}")

    row("Mean cosine sim",       model_arr.mean(),              baseline_arr.mean())
    row("Median cosine sim",     float(np.median(model_arr)),   float(np.median(baseline_arr)))
    row("Std cosine sim",        model_arr.std(),               baseline_arr.std())
    row("p25 cosine sim",        float(np.percentile(model_arr, 25)), float(np.percentile(baseline_arr, 25)))
    row("p75 cosine sim",        float(np.percentile(model_arr, 75)), float(np.percentile(baseline_arr, 75)))
    hline()

    # Fraction of steps where model beats baseline
    beats = (model_arr > baseline_arr).mean()
    print(f"  {'Model > baseline (% of steps)':<35} {beats*100:>9.1f}%")
    hline()

    # Interpretation
    mean_delta = float(model_arr.mean() - baseline_arr.mean())
    print("  INTERPRETATION")
    if mean_delta > 0.05:
        print(f"  Model outperforms naive baseline by {mean_delta:+.4f} — strong predictive signal.")
    elif mean_delta > 0.01:
        print(f"  Model outperforms naive baseline by {mean_delta:+.4f} — modest predictive signal.")
    elif mean_delta > -0.01:
        print(f"  Model is approximately on par with naive baseline ({mean_delta:+.4f}).")
    else:
        print(f"  Model underperforms naive baseline ({mean_delta:+.4f}) — investigate sim loss weight.")

    model_mean = float(model_arr.mean())
    print()
    if model_mean > 0.6:
        print(f"  Mean cosine sim = {model_mean:.4f}: strong next-session prediction.")
    elif model_mean > 0.4:
        print(f"  Mean cosine sim = {model_mean:.4f}: moderate next-session prediction.")
    elif model_mean > 0.2:
        print(f"  Mean cosine sim = {model_mean:.4f}: weak but above-random prediction.")
    else:
        print(f"  Mean cosine sim = {model_mean:.4f}: near-random (possible collapse).")

    hline()

    # Per-arc breakdown
    print()
    hline("=")
    print("  Per-arc prediction accuracy")
    hline("=")
    print(f"  {'Arc':<28}  {'N':>6}  {'Model':>8}  {'Baseline':>8}  {'Delta':>7}")
    hline()
    for arc, r in sorted(arc_results.items(), key=lambda x: -x[1]["model_mean"]):
        sign = "+" if r["delta"] >= 0 else ""
        print(
            f"  {arc:<28}  {r['n']:>6}  "
            f"{r['model_mean']:>8.4f}  {r['baseline_mean']:>8.4f}  "
            f"{sign}{r['delta']:>6.4f}"
        )
    hline("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate world model next-session prediction accuracy"
    )
    parser.add_argument(
        "--data", type=str,
        default=str(Path(__file__).parent.parent / "data" / "trajectories_10k_v2_val.json"),
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Path(__file__).parent.parent / "checkpoints" / "transition_v2.pt"),
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)
    for p in [data_path, ckpt_path]:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    # Load model
    logger.info("Loading transition model...")
    from kokoro.transition import load_transition_checkpoint
    model, cfg = load_transition_checkpoint(ckpt_path)
    logger.info(f"  arch={cfg.get('arch', 'mlp')}, session_dim={cfg.get('session_dim', 384)}")

    # Load encoder
    logger.info("Loading encoder...")
    from kokoro.encoder import SessionEncoder
    from training.train import precompute_embeddings

    use_vad = cfg.get("session_dim", 384) > 384
    encoder = SessionEncoder(device="cpu", use_vad_features=use_vad)
    _ = encoder.model

    # Load trajectories (same val split as training)
    logger.info(f"Loading {data_path.name}...")
    with open(data_path) as f:
        all_trajs = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)
    n_val     = int(len(all_trajs) * args.val_split)
    val_trajs = all_trajs[len(all_trajs) - n_val:]
    logger.info(f"  Val set: {len(val_trajs):,} trajectories")

    # Precompute embeddings
    logger.info("Precomputing embeddings...")
    emb_cache = precompute_embeddings(val_trajs, encoder, device)

    # Evaluate
    logger.info("Evaluating prediction accuracy...")
    results = eval_prediction_accuracy(model, val_trajs, emb_cache, device)

    print_report(results, n_val_trajs=len(val_trajs))
