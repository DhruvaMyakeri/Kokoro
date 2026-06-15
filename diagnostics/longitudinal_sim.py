"""
Longitudinal study simulation.

Simulates what a real longitudinal user study would measure:
  - Tracks 20 synthetic "users" following different arcs over many sessions
  - After each session: compares predicted (valence, arousal) to ground truth
  - Measures cold-start latency (how many sessions before predictions stabilize)
  - Measures arc-change detection (does the model react when trajectory shifts?)
  - Computes per-arc tracking RMSE and MAE over time

This is a simulation using synthetic data. A real longitudinal study would need
real users over real weeks — this measures whether the model COULD track them
if deployed, using the same trajectories used for training evaluation.

Key metrics:
  - Valence/arousal MAE after warm-up (session >= 3)
  - Cold-start RMSE (sessions 1-3 vs sessions 4+)
  - Arc-change detection lag: sessions before model reacts to a trajectory shift
  - Steady-state tracking accuracy per arc type

Run:
    python -m diagnostics.longitudinal_sim
    python -m diagnostics.longitudinal_sim --n-users 50 --seed 0
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

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)

W = 74


def hline(c: str = "-") -> None:
    print(c * W)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_user(
    model,
    probe,
    trajectory: dict,
    emb_cache: dict,
    device: torch.device,
) -> list[dict]:
    """
    Simulate a single user following one trajectory.

    At each session t, the model predicts (valence, arousal) from state_t.
    Ground truth is the trajectory's labeled (valence, arousal).

    Returns a list of per-session records.
    """
    sessions = trajectory["sessions"]
    arc      = trajectory.get("arc_name", "unknown")

    records = []
    state   = torch.zeros(model.STATE_DIM, device=device)

    with torch.no_grad():
        for t, sess in enumerate(sessions):
            cid = sess["conv_id"]
            if cid not in emb_cache:
                continue
            emb = emb_cache[cid].to(device)

            # Update state
            state = model(state, emb)

            # Probe prediction from current state
            va = probe(state.unsqueeze(0)).squeeze(0)
            pred_v = va[0].item()
            pred_a = va[1].item()

            # Ground truth
            gt_v = sess.get("valence", 0.0)
            gt_a = sess.get("arousal", 0.0)

            records.append({
                "session_idx":   t,
                "arc":           arc,
                "gt_valence":    gt_v,
                "gt_arousal":    gt_a,
                "pred_valence":  pred_v,
                "pred_arousal":  pred_a,
                "err_valence":   abs(pred_v - gt_v),
                "err_arousal":   abs(pred_a - gt_a),
                "se_valence":    (pred_v - gt_v) ** 2,
                "se_arousal":    (pred_a - gt_a) ** 2,
                "warm":          t >= 2,  # after 3 sessions (0-indexed: 0,1,2)
            })

    return records


def simulate_arc_change(
    model,
    probe,
    traj_a: dict,
    traj_b: dict,
    emb_cache: dict,
    device: torch.device,
    change_at: int = 4,
) -> list[dict]:
    """
    Simulate a user who transitions from arc A to arc B at session `change_at`.

    First `change_at` sessions come from traj_a's arc pattern.
    Remaining sessions come from traj_b's arc pattern (different arc type).

    Returns per-session records with `arc_segment` = "pre" | "post".
    """
    sessions_a = traj_a["sessions"]
    sessions_b = traj_b["sessions"]
    arc_a = traj_a.get("arc_name", "arc_A")
    arc_b = traj_b.get("arc_name", "arc_B")

    # Splice: first `change_at` from A, rest from B
    combined = (
        [(s, arc_a, "pre")  for s in sessions_a[:change_at]] +
        [(s, arc_b, "post") for s in sessions_b[:max(6, len(sessions_b))]]
    )

    records = []
    state   = torch.zeros(model.STATE_DIM, device=device)

    with torch.no_grad():
        for t, (sess, arc, segment) in enumerate(combined):
            cid = sess["conv_id"]
            if cid not in emb_cache:
                continue
            emb = emb_cache[cid].to(device)
            state = model(state, emb)

            va = probe(state.unsqueeze(0)).squeeze(0)
            pred_v = va[0].item()
            pred_a = va[1].item()
            gt_v   = sess.get("valence", 0.0)
            gt_a   = sess.get("arousal", 0.0)

            records.append({
                "session_idx":   t,
                "arc_a":         arc_a,
                "arc_b":         arc_b,
                "arc_segment":   segment,
                "change_at":     change_at,
                "gt_valence":    gt_v,
                "gt_arousal":    gt_a,
                "pred_valence":  pred_v,
                "pred_arousal":  pred_a,
                "err_valence":   abs(pred_v - gt_v),
                "err_arousal":   abs(pred_a - gt_a),
            })

    return records


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def cold_start_analysis(all_records: list[dict]) -> dict:
    """
    Compare prediction error in cold-start (sessions 0-2) vs steady-state (3+).
    """
    cold    = [r for r in all_records if not r["warm"]]
    warm    = [r for r in all_records if r["warm"]]

    def mae(records, key):
        return np.mean([r[key] for r in records]) if records else float("nan")

    return {
        "cold_mae_valence":  mae(cold, "err_valence"),
        "cold_mae_arousal":  mae(cold, "err_arousal"),
        "warm_mae_valence":  mae(warm, "err_valence"),
        "warm_mae_arousal":  mae(warm, "err_arousal"),
        "n_cold": len(cold),
        "n_warm": len(warm),
    }


def per_session_error(all_records: list[dict], max_session: int = 12) -> dict:
    """
    MAE by session index (0 = first session, 1 = second, ...).
    Shows how tracking improves with more sessions.
    """
    by_session: dict[int, list[float]] = defaultdict(list)
    for r in all_records:
        idx = r["session_idx"]
        if idx < max_session:
            by_session[idx].append(r["err_valence"])

    return {
        idx: np.mean(errs) for idx, errs in sorted(by_session.items())
    }


def per_arc_error(all_records: list[dict]) -> dict:
    """MAE per arc type (warm sessions only)."""
    by_arc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in all_records:
        if r["warm"]:
            by_arc[r["arc"]].append((r["err_valence"], r["err_arousal"]))
    return {
        arc: {
            "n":           len(pairs),
            "mae_valence": float(np.mean([p[0] for p in pairs])),
            "mae_arousal": float(np.mean([p[1] for p in pairs])),
        }
        for arc, pairs in by_arc.items()
    }


def arc_change_lag(change_records: list[dict]) -> dict:
    """
    After the arc changes at session `change_at`, how many sessions before
    the model's prediction moves in the correct direction?

    "Correct direction" = the sign of (gt_B_valence - gt_A_valence) matches
    the sign of (pred_valence - pre_change_mean_valence).
    """
    if not change_records:
        return {}

    change_at = change_records[0]["change_at"]
    pre  = [r for r in change_records if r["arc_segment"] == "pre"]
    post = [r for r in change_records if r["arc_segment"] == "post"]

    if not pre or not post:
        return {}

    pre_mean_v  = np.mean([r["gt_valence"]  for r in pre])
    post_gt_v   = np.mean([r["gt_valence"]  for r in post])
    direction_v = np.sign(post_gt_v - pre_mean_v)

    # Find first post-change session where prediction moves in correct direction
    baseline_pred_v = np.mean([r["pred_valence"] for r in pre[-2:]]) if len(pre) >= 2 else pre[-1]["pred_valence"]
    lag = None
    for r in post:
        if direction_v * (r["pred_valence"] - baseline_pred_v) > 0.02:
            lag = r["session_idx"] - change_at
            break

    arc_a = change_records[0].get("arc_a", "?")
    arc_b = change_records[0].get("arc_b", "?")

    return {
        "arc_a":          arc_a,
        "arc_b":          arc_b,
        "change_at":      change_at,
        "direction":      "positive" if direction_v > 0 else "negative",
        "gt_delta_v":     float(post_gt_v - pre_mean_v),
        "lag_sessions":   lag,  # None = not detected in available sessions
        "pre_mean_v":     float(pre_mean_v),
        "post_gt_v":      float(post_gt_v),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    all_records: list[dict],
    cold_start: dict,
    per_session: dict,
    arc_errors: dict,
    change_results: list[dict],
    n_users: int,
) -> None:
    print()
    hline("=")
    print("  Longitudinal Tracking Simulation")
    hline("=")
    print(f"  Simulated users:     {n_users}")
    print(f"  Total session steps: {len(all_records):,}")
    print(f"  Warm steps (>= 3):   {sum(1 for r in all_records if r['warm']):,}")
    hline()

    # Overall warm MAE
    warm = [r for r in all_records if r["warm"]]
    if warm:
        mae_v = np.mean([r["err_valence"] for r in warm])
        mae_a = np.mean([r["err_arousal"] for r in warm])
        rmse_v = np.sqrt(np.mean([r["se_valence"] for r in warm]))
        rmse_a = np.sqrt(np.mean([r["se_arousal"] for r in warm]))
        print(f"  Steady-state (sessions >= 3):")
        print(f"    Valence  MAE={mae_v:.3f}  RMSE={rmse_v:.3f}")
        print(f"    Arousal  MAE={mae_a:.3f}  RMSE={rmse_a:.3f}")
    hline()

    # Cold-start analysis
    print()
    print("  Cold-start vs steady-state comparison:")
    hline()
    cv  = cold_start["cold_mae_valence"]
    wv  = cold_start["warm_mae_valence"]
    ca  = cold_start["cold_mae_arousal"]
    wa  = cold_start["warm_mae_arousal"]
    impv = (cv - wv) / cv * 100 if cv > 0 else 0
    impa = (ca - wa) / ca * 100 if ca > 0 else 0
    print(f"  {'Condition':<22}  {'V-MAE':>8}  {'A-MAE':>8}  {'V-improv':>10}")
    hline()
    print(f"  {'Cold-start (0-2)':<22}  {cv:>8.3f}  {ca:>8.3f}")
    print(f"  {'Warm (3+)':<22}  {wv:>8.3f}  {wa:>8.3f}  {impv:>+9.1f}%")
    hline()

    # Per-session error curve
    print()
    print("  Valence MAE by session index (learning curve):")
    hline()
    print(f"  {'Session':>8}  {'V-MAE':>8}  {'Bar':<30}")
    hline()
    errs = list(per_session.values())
    max_err = max(errs) if errs else 1.0
    for idx, mae_val in sorted(per_session.items()):
        bar = "#" * int(mae_val / max_err * 28)
        marker = " <-- warm start" if idx == 2 else ""
        print(f"  {idx:>8}  {mae_val:>8.3f}  {bar:<30}{marker}")
    hline()

    # Per-arc errors
    print()
    print("  Per-arc steady-state MAE (warm sessions):")
    hline()
    print(f"  {'Arc':<30}  {'N':>6}  {'V-MAE':>8}  {'A-MAE':>8}")
    hline()
    for arc, stats in sorted(arc_errors.items(), key=lambda x: x[1]["mae_valence"]):
        print(f"  {arc:<30}  {stats['n']:>6}  {stats['mae_valence']:>8.3f}  {stats['mae_arousal']:>8.3f}")
    hline()

    # Arc-change detection
    if change_results:
        print()
        print("  Arc-change detection (simulated trajectory switches):")
        hline()
        print(f"  {'Transition':<35}  {'Direction':>10}  {'GT-dV':>8}  {'Lag':>6}")
        hline()
        lags = []
        for cr in change_results:
            lag_str = str(cr["lag_sessions"]) if cr["lag_sessions"] is not None else "missed"
            if cr["lag_sessions"] is not None:
                lags.append(cr["lag_sessions"])
            transition = f"{cr['arc_a'][:15]} -> {cr['arc_b'][:14]}"
            print(
                f"  {transition:<35}  {cr['direction']:>10}  "
                f"{cr['gt_delta_v']:>+8.3f}  {lag_str:>6}"
            )
        hline()
        if lags:
            print(f"  Mean detection lag: {np.mean(lags):.1f} sessions  "
                  f"(detected {len(lags)}/{len(change_results)} transitions)")
    hline("=")

    # Interpretation
    print()
    if warm:
        if mae_v < 0.15:
            print(f"  Valence tracking: STRONG (MAE {mae_v:.3f} < 0.15)")
        elif mae_v < 0.25:
            print(f"  Valence tracking: MODERATE (MAE {mae_v:.3f})")
        else:
            print(f"  Valence tracking: WEAK (MAE {mae_v:.3f} > 0.25)")

        if impv > 10:
            print(f"  Cold-start warm-up confirmed: {impv:.1f}% MAE improvement after session 3")
        else:
            print(f"  Cold-start effect minimal ({impv:.1f}%) — model adapts quickly")

    print()
    print("  NOTE: This is a simulation on synthetic in-distribution data.")
    print("  Real longitudinal performance on naturalistic conversations may differ.")
    print("  The probe generalization failure documented in the paper (range -0.144 to -0.110)")
    print("  would manifest here as uniformly low MAE improvement across emotional states.")
    hline("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Longitudinal tracking simulation")
    parser.add_argument("--data",              type=str, default="data/trajectories_10k.json")
    parser.add_argument("--checkpoint",        type=str, default="checkpoints/transition_v1.pt")
    parser.add_argument("--probe-checkpoint",  type=str, default="checkpoints/valence_arousal_probe.pt")
    parser.add_argument("--val-split",         type=float, default=0.2)
    parser.add_argument("--n-users",           type=int,   default=100,
                        help="Number of synthetic users to simulate (default: 100)")
    parser.add_argument("--n-arc-changes",     type=int,   default=10,
                        help="Number of arc-change simulations (default: 10)")
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    root      = Path(__file__).parent.parent
    data_path = root / args.data
    ckpt_path = root / args.checkpoint
    probe_path= root / args.probe_checkpoint

    for p in [data_path, ckpt_path, probe_path]:
        if not p.exists():
            sys.exit(f"Not found: {p}")

    rng    = random.Random(args.seed)
    device = torch.device("cpu")

    # Load transition model
    logger.info("Loading transition model...")
    from kokoro.transition import TransitionModel
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("model_config", {})
    model = TransitionModel(**{k: v for k, v in cfg.items() if k in ("state_dim", "session_dim", "hidden_dim")})
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_vad = cfg.get("session_dim", 384) > 384
    logger.info(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # Load probe
    logger.info("Loading linear probe...")
    from training.train_probe import ValenceArousalProbe
    probe_ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
    probe_cfg  = probe_ckpt.get("probe_config", {})
    probe = ValenceArousalProbe(state_dim=probe_cfg.get("state_dim", 384))
    probe.load_state_dict(probe_ckpt["model_state_dict"])
    probe.eval()

    # Load encoder
    logger.info("Loading encoder...")
    from kokoro.encoder import SessionEncoder
    from training.train import precompute_embeddings
    encoder = SessionEncoder(device="cpu", use_vad_features=use_vad)

    # Load trajectories
    logger.info(f"Loading {data_path.name}...")
    with open(data_path) as f:
        all_trajs = json.load(f)
    rng.shuffle(all_trajs)
    n_val     = int(len(all_trajs) * args.val_split)
    val_trajs = all_trajs[len(all_trajs) - n_val:]
    logger.info(f"  Val set: {len(val_trajs):,} trajectories")

    # Sample users
    users = rng.sample(val_trajs, min(args.n_users, len(val_trajs)))
    logger.info(f"  Simulating {len(users)} users...")

    # Precompute embeddings for selected users + arc-change pairs
    arc_change_pairs: list[tuple[dict, dict]] = []
    arc_groups: dict[str, list[dict]] = defaultdict(list)
    for t in val_trajs:
        arc_groups[t.get("arc_name", "unknown")].append(t)

    # Pick arc-change pairs: positive → negative (gradual_decline) and vice versa
    change_templates = [
        ("stable_positive",     "gradual_decline"),
        ("stable_negative",     "slow_recovery"),
        ("stable_positive",     "chronic_low_grade_anxiety"),
        ("slow_recovery",       "relapse_dip"),
        ("excitement_to_contentment", "anxiety_to_depression"),
        ("social_confidence_growth",  "gradual_decline"),
        ("stable_positive",     "grief_arc"),
        ("slow_recovery",       "chronic_low_grade_anxiety"),
        ("excitement_to_contentment", "stable_negative"),
        ("post_traumatic_growth",     "relapse_dip"),
    ]
    for arc_a, arc_b in change_templates[:args.n_arc_changes]:
        if arc_groups.get(arc_a) and arc_groups.get(arc_b):
            ta = rng.choice(arc_groups[arc_a])
            tb = rng.choice(arc_groups[arc_b])
            arc_change_pairs.append((ta, tb))

    # All trajectories we need embeddings for
    needed = users + [t for pair in arc_change_pairs for t in pair]
    logger.info("Precomputing embeddings...")
    emb_cache = precompute_embeddings(needed, encoder, device)

    # Simulate users
    logger.info("Simulating longitudinal tracking...")
    all_records: list[dict] = []
    for traj in users:
        recs = simulate_user(model, probe, traj, emb_cache, device)
        all_records.extend(recs)
    logger.info(f"  {len(all_records):,} session records collected")

    # Simulate arc changes
    logger.info(f"Simulating {len(arc_change_pairs)} arc-change transitions...")
    change_results: list[dict] = []
    for ta, tb in arc_change_pairs:
        recs  = simulate_arc_change(model, probe, ta, tb, emb_cache, device, change_at=4)
        lag   = arc_change_lag(recs)
        if lag:
            change_results.append(lag)

    # Analyze
    cold_start  = cold_start_analysis(all_records)
    per_session = per_session_error(all_records)
    arc_errors  = per_arc_error(all_records)

    print_report(all_records, cold_start, per_session, arc_errors, change_results, len(users))
