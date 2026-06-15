"""
Arc-type out-of-distribution (OOD) evaluation.

Evaluates a transition model checkpoint on a SPECIFIED subset of arc types only.

Use case A — OOD test (true holdout):
  1. Train with:   python -m training.train --holdout-arcs grief_arc post_traumatic_growth
                   --checkpoint checkpoints/transition_ood.pt
  2. Evaluate on withheld arcs:
     python -m diagnostics.arc_ood_eval --checkpoint checkpoints/transition_ood.pt
                                         --eval-arcs grief_arc post_traumatic_growth
  3. Compare to full-model baseline:
     python -m diagnostics.arc_ood_eval --checkpoint checkpoints/transition_v1.pt
                                         --eval-arcs grief_arc post_traumatic_growth

Use case B — current model per-arc breakdown (all arcs seen during training):
  python -m diagnostics.arc_ood_eval  # defaults: all arcs, full checkpoint

Metric: world model cosine sim on eval-arc trajectories vs naive baseline
  model_sim[t]    = cosine_sim(normalize(state_t), normalize(e_{t+1}[:384]))
  baseline_sim[t] = cosine_sim(normalize(e_t[:384]), normalize(e_{t+1}[:384]))

If the OOD model achieves significantly lower delta on withheld arcs compared to
a full-model baseline, the model has overfit to arc structure.
If it holds, it has learned general emotional dynamics.

Run:
    python -m diagnostics.arc_ood_eval
    python -m diagnostics.arc_ood_eval --eval-arcs grief_arc post_traumatic_growth weekend_oscillation
    python -m diagnostics.arc_ood_eval --checkpoint checkpoints/transition_ood.pt --eval-arcs grief_arc
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)

W = 74


def hline(c: str = "-") -> None:
    print(c * W)


# All 14 arc types defined in data/arc_templates.py
ALL_ARCS = [
    "gradual_decline",
    "slow_recovery",
    "acute_stress_stabilization",
    "grief_arc",
    "chronic_low_grade_anxiety",
    "stable_positive",
    "stable_negative",
    "relapse_dip",
    "anxiety_to_depression",
    "depression_to_anxiety",
    "post_traumatic_growth",
    "social_confidence_growth",
    "excitement_to_contentment",
    "weekend_oscillation",
]

# Recommended holdout set for OOD experiment:
#   - grief_arc: long, non-linear (drop then slow rise)
#   - post_traumatic_growth: surpasses baseline after deep drop
#   - weekend_oscillation: repeating cycle, unlike other arcs
RECOMMENDED_HOLDOUT = ["grief_arc", "post_traumatic_growth", "weekend_oscillation"]


def eval_on_arcs(
    model,
    trajectories: list[dict],
    emb_cache: dict,
    eval_arcs: set[str],
    device: torch.device,
) -> dict:
    """
    Evaluate world model prediction accuracy on trajectories from specified arc types.

    Returns per-step model_sim and baseline_sim arrays, plus per-arc summary.
    """
    model.eval()
    state_dim = model.STATE_DIM

    model_sims:    list[float] = []
    baseline_sims: list[float] = []
    arc_names:     list[str]   = []

    with torch.no_grad():
        for traj in trajectories:
            arc = traj.get("arc_name", "unknown")
            if eval_arcs and arc not in eval_arcs:
                continue
            sessions = traj["sessions"]
            if len(sessions) < 2:
                continue
            embs  = [emb_cache[s["conv_id"]] for s in sessions]
            state = torch.zeros(state_dim, device=device)

            for t in range(len(embs) - 1):
                state = model(state, embs[t])
                s_norm = F.normalize(state,                    dim=-1)
                e_norm = F.normalize(embs[t + 1][:state_dim], dim=-1)
                b_norm = F.normalize(embs[t][:state_dim],     dim=-1)
                model_sims.append((s_norm * e_norm).sum().item())
                baseline_sims.append((b_norm * e_norm).sum().item())
                arc_names.append(arc)

    if not model_sims:
        return {"error": "No trajectories matched the specified arc types."}

    model_arr    = np.array(model_sims,    dtype=np.float32)
    baseline_arr = np.array(baseline_sims, dtype=np.float32)

    # Per-arc breakdown
    arcs_unique = sorted(set(arc_names))
    arc_results = {}
    for arc in arcs_unique:
        idx = [i for i, a in enumerate(arc_names) if a == arc]
        arc_results[arc] = {
            "n":             len(idx),
            "model_mean":    float(model_arr[idx].mean()),
            "baseline_mean": float(baseline_arr[idx].mean()),
            "delta":         float(model_arr[idx].mean() - baseline_arr[idx].mean()),
            "pct_beats":     float((model_arr[idx] > baseline_arr[idx]).mean()) * 100,
        }

    return {
        "model_sims":    model_arr,
        "baseline_sims": baseline_arr,
        "arc_names":     arc_names,
        "arc_results":   arc_results,
        "n_trajs":       len(set(arc_names)),
        "n_pairs":       len(model_sims),
    }


def print_report(results: dict, eval_arcs: set[str], checkpoint_name: str) -> None:
    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    model_arr    = results["model_sims"]
    baseline_arr = results["baseline_sims"]
    arc_results  = results["arc_results"]

    print()
    hline("=")
    print(f"  Arc OOD Evaluation")
    print(f"  Checkpoint: {checkpoint_name}")
    if eval_arcs:
        print(f"  Evaluated arcs: {sorted(eval_arcs)}")
    else:
        print(f"  Evaluated arcs: ALL (baseline mode)")
    hline("=")
    print(f"  Trajectories covered: {len(arc_results)} arc types")
    print(f"  Prediction pairs:     {len(model_arr):,}")
    hline()
    print(f"  {'Metric':<30}  {'Model':>10}  {'Baseline':>10}  {'Delta':>8}")
    hline()

    def row(label, mv, bv):
        d = mv - bv
        s = "+" if d >= 0 else ""
        print(f"  {label:<30}  {mv:>10.4f}  {bv:>10.4f}  {s}{d:>7.4f}")

    row("Mean cosine sim",   model_arr.mean(),              baseline_arr.mean())
    row("Median cosine sim", float(np.median(model_arr)),   float(np.median(baseline_arr)))
    row("Std cosine sim",    model_arr.std(),               baseline_arr.std())
    hline()
    beats = (model_arr > baseline_arr).mean()
    print(f"  {'Model > baseline (% of steps)':<30}  {beats*100:>9.1f}%")
    hline()

    # Per-arc table
    print()
    hline("=")
    print(f"  Per-arc breakdown")
    hline("=")
    print(f"  {'Arc':<30}  {'N':>6}  {'Model':>7}  {'Base':>7}  {'Delta':>7}  {'Beats%':>7}")
    hline()
    for arc, r in sorted(arc_results.items(), key=lambda x: -x[1]["model_mean"]):
        d = r["delta"]
        s = "+" if d >= 0 else ""
        tag = "  [OOD]" if arc in eval_arcs else ""
        print(
            f"  {arc:<30}  {r['n']:>6}  {r['model_mean']:>7.4f}  "
            f"{r['baseline_mean']:>7.4f}  {s}{d:>6.4f}  {r['pct_beats']:>6.1f}%{tag}"
        )
    hline()

    # OOD interpretation
    if eval_arcs:
        ood_deltas    = [arc_results[a]["delta"] for a in eval_arcs if a in arc_results]
        non_ood_arcs  = set(arc_results.keys()) - eval_arcs
        non_ood_deltas= [arc_results[a]["delta"] for a in non_ood_arcs if a in arc_results]
        if ood_deltas and non_ood_deltas:
            ood_mean     = np.mean(ood_deltas)
            non_ood_mean = np.mean(non_ood_deltas)
            drop_pct     = (ood_mean - non_ood_mean) / non_ood_mean * 100 if non_ood_mean else 0
            print()
            print(f"  OOD delta mean:       {ood_mean:+.4f}")
            print(f"  Non-OOD delta mean:   {non_ood_mean:+.4f}")
            print(f"  OOD gap:              {drop_pct:+.1f}%")
            if drop_pct < -20:
                print(f"  Interpretation: MODEL OVERFIT TO ARC STRUCTURE (>20% degradation on OOD arcs)")
            elif drop_pct < -5:
                print(f"  Interpretation: Moderate degradation on OOD arcs — partial overfitting")
            else:
                print(f"  Interpretation: Robust — OOD and non-OOD arcs perform similarly")
    hline("=")

    print()
    print("  NOTE: For a TRUE OOD experiment, this must be run on a model that was")
    print("  trained WITHOUT the eval arcs (--holdout-arcs flag in training/train.py).")
    print("  Running on the full checkpoint only shows per-arc breakdown, not OOD generalization.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Arc-type OOD evaluation")
    parser.add_argument("--data",       type=str, default="data/trajectories_10k.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transition_v1.pt")
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument(
        "--eval-arcs", nargs="*", default=[],
        metavar="ARC",
        help=(
            "Arc types to evaluate on (empty = all arcs). "
            f"Recommended holdout set: {RECOMMENDED_HOLDOUT}"
        ),
    )
    args = parser.parse_args()

    root      = Path(__file__).parent.parent
    data_path = root / args.data
    ckpt_path = root / args.checkpoint

    for p in [data_path, ckpt_path]:
        if not p.exists():
            sys.exit(f"Not found: {p}")

    eval_arcs = set(args.eval_arcs)
    if eval_arcs:
        unknown = eval_arcs - set(ALL_ARCS)
        if unknown:
            logger.warning(f"Unknown arc types: {unknown}. Known arcs: {ALL_ARCS}")

    device = torch.device("cpu")

    # Load model
    logger.info("Loading transition model...")
    from kokoro.transition import TransitionModel
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = ckpt.get("model_config", {})
    model = TransitionModel(**{k: v for k, v in cfg.items() if k in ("state_dim", "session_dim", "hidden_dim")})
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_vad = cfg.get("session_dim", 384) > 384
    logger.info(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # Load encoder
    logger.info("Loading encoder...")
    from kokoro.encoder import SessionEncoder
    from training.train import precompute_embeddings
    encoder = SessionEncoder(device="cpu", use_vad_features=use_vad)

    # Load trajectories
    logger.info(f"Loading {data_path.name}...")
    with open(data_path) as f:
        all_trajs = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)
    n_val     = int(len(all_trajs) * args.val_split)
    val_trajs = all_trajs[len(all_trajs) - n_val:]
    logger.info(f"  Val set: {len(val_trajs):,} trajectories")

    arc_counts = {}
    for t in val_trajs:
        arc = t.get("arc_name", "unknown")
        arc_counts[arc] = arc_counts.get(arc, 0) + 1
    logger.info(f"  Arc distribution: {dict(sorted(arc_counts.items()))}")

    # Precompute embeddings
    logger.info("Precomputing embeddings...")
    emb_cache = precompute_embeddings(val_trajs, encoder, device)

    # Evaluate
    logger.info(f"Evaluating on arcs: {sorted(eval_arcs) if eval_arcs else 'ALL'}...")
    results = eval_on_arcs(model, val_trajs, emb_cache, eval_arcs, device)

    print_report(results, eval_arcs, ckpt_path.name)
