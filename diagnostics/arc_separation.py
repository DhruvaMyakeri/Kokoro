"""
Arc separation analysis on the current transition model (PR=339.7).

Measures how well the model's final state vectors cluster by arc type.

Metric:
  separation_ratio = mean_between_arc_distance / mean_within_arc_distance
  where distance = 1 - cosine_similarity (cosine distance in [0, 2])

A ratio of 1.0 means arc type has no effect on clustering.
A ratio > 1.0 means between-arc distances exceed within-arc — arc type structure.

Three representations are compared:
  1. Last-session embedding (baseline) — raw MiniLM, no model
  2. Trained model final state       — transition model output after all sessions
  3. EWMA baseline                   — exponential moving average, no params

Shuffled-label control: permutes arc labels, recomputes ratio.
  If ratio collapses → structure was real.
  If ratio holds → structure is an artifact of vector geometry.

Paper reference (original PR=1.4 model):
  Last-session baseline sep ratio: 1.0471
  Trained model sep ratio:         3.1111  (+197% over baseline)
  Shuffled control:                0.9807  (-68.5% from real labels)

Run:
    python -m diagnostics.arc_separation
    python -m diagnostics.arc_separation --n-shuffles 20 --max-pairs 50000
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

W = 72


def hline(c: str = "-") -> None:
    print(c * W)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_separation(
    vecs: np.ndarray,          # (N, D) float32, need NOT be normalized
    labels: list[str],         # arc label per vector
    max_pairs: int = 100_000,  # cap on pairwise comparisons (for speed)
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Compute arc separation ratio from a set of vectors and their arc labels.

    Returns dict with:
      within_dist  — mean cosine distance within same-arc pairs
      between_dist — mean cosine distance between different-arc pairs
      sep_ratio    — between / within
      arc_stats    — per-arc within-arc distance
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = len(vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    vecs_n = vecs / norms  # (N, D) normalized

    label_arr = np.array(labels)
    unique_arcs = sorted(set(labels))

    # Build index arrays
    arc_to_idx: dict[str, np.ndarray] = {
        a: np.where(label_arr == a)[0] for a in unique_arcs
    }

    # --- Within-arc pairs ---
    within_dists: list[float] = []
    arc_within: dict[str, float] = {}

    for arc, idx in arc_to_idx.items():
        if len(idx) < 2:
            arc_within[arc] = float("nan")
            continue
        # pairwise for this arc
        v = vecs_n[idx]           # (k, D)
        sim = v @ v.T             # (k, k) cosine similarity
        # upper triangle, excluding diagonal
        k = len(idx)
        row, col = np.triu_indices(k, k=1)
        dists = 1.0 - sim[row, col]
        if len(dists) > max_pairs:
            chosen = rng.choice(len(dists), max_pairs, replace=False)
            dists = dists[chosen]
        arc_within[arc] = float(dists.mean())
        within_dists.extend(dists.tolist())

    # --- Between-arc pairs (sample from all cross-arc pairs) ---
    # Enumerate all arc pairs, sample proportionally
    arc_list = unique_arcs
    between_dists: list[float] = []

    for i, arc_a in enumerate(arc_list):
        for j, arc_b in enumerate(arc_list):
            if j <= i:
                continue
            idx_a = arc_to_idx[arc_a]
            idx_b = arc_to_idx[arc_b]
            va = vecs_n[idx_a]  # (na, D)
            vb = vecs_n[idx_b]  # (nb, D)
            sim_cross = va @ vb.T  # (na, nb)
            dists = (1.0 - sim_cross).ravel()
            n_sample = min(len(dists), max(100, max_pairs // (len(arc_list) ** 2 // 2 + 1)))
            if len(dists) > n_sample:
                chosen = rng.choice(len(dists), n_sample, replace=False)
                dists = dists[chosen]
            between_dists.extend(dists.tolist())

    mean_within  = float(np.mean(within_dists))  if within_dists  else float("nan")
    mean_between = float(np.mean(between_dists)) if between_dists else float("nan")
    sep_ratio    = mean_between / mean_within if mean_within > 1e-9 else float("nan")

    return {
        "within_dist":  mean_within,
        "between_dist": mean_between,
        "sep_ratio":    sep_ratio,
        "n_within":     len(within_dists),
        "n_between":    len(between_dists),
        "arc_within":   arc_within,
    }


def shuffled_control(
    vecs: np.ndarray,
    labels: list[str],
    n_shuffles: int = 10,
    max_pairs: int = 50_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run shuffled-label control: permute labels, recompute sep ratio N times."""
    if rng is None:
        rng = np.random.default_rng(99)

    labels_arr = list(labels)
    ratios = []
    for _ in range(n_shuffles):
        shuffled = labels_arr.copy()
        rng.shuffle(shuffled)
        r = compute_separation(vecs, shuffled, max_pairs=max_pairs, rng=rng)
        ratios.append(r["sep_ratio"])

    ratios_arr = np.array([r for r in ratios if not np.isnan(r)])
    return {
        "mean_ratio":   float(ratios_arr.mean()) if len(ratios_arr) else float("nan"),
        "std_ratio":    float(ratios_arr.std())  if len(ratios_arr) else float("nan"),
        "all_ratios":   ratios,
        "n_shuffles":   n_shuffles,
    }


# ---------------------------------------------------------------------------
# Roll out trajectories
# ---------------------------------------------------------------------------

def collect_final_states(
    model,
    trajectories: list[dict],
    emb_cache: dict,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """
    For each trajectory, roll out all sessions through the transition model
    and return the FINAL state vector + arc label.

    Returns:
        states: np.ndarray (N, 384)
        arcs:   list[str]  (N,)
    """
    states = []
    arcs   = []

    model.eval()
    state_dim = model.STATE_DIM

    with torch.no_grad():
        for traj in trajectories:
            sessions = traj.get("sessions", [])
            arc      = traj.get("arc_name", "unknown")
            if not sessions:
                continue

            state = torch.zeros(state_dim, device=device)
            for sess in sessions:
                cid = sess["conv_id"]
                if cid not in emb_cache:
                    continue
                emb   = emb_cache[cid].to(device)
                state = model(state, emb)

            states.append(state.cpu().numpy())
            arcs.append(arc)

    return np.stack(states).astype(np.float32), arcs


def collect_last_session_embs(
    trajectories: list[dict],
    emb_cache: dict,
    state_dim: int,
) -> tuple[np.ndarray, list[str]]:
    """Baseline: just use the last session embedding (truncated to state_dim)."""
    vecs = []
    arcs = []
    for traj in trajectories:
        sessions = traj.get("sessions", [])
        arc      = traj.get("arc_name", "unknown")
        if not sessions:
            continue
        cid = sessions[-1]["conv_id"]
        if cid not in emb_cache:
            continue
        emb = emb_cache[cid].cpu().numpy()[:state_dim]
        vecs.append(emb)
        arcs.append(arc)
    return np.stack(vecs).astype(np.float32), arcs


def collect_ewma_states(
    trajectories: list[dict],
    emb_cache: dict,
    state_dim: int,
    alpha: float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    """
    EWMA baseline: exponential moving average of session embeddings, no params.
    new_state = alpha * state + (1 - alpha) * emb[:state_dim]
    """
    vecs = []
    arcs = []
    for traj in trajectories:
        sessions = traj.get("sessions", [])
        arc      = traj.get("arc_name", "unknown")
        if not sessions:
            continue
        state = None
        for sess in sessions:
            cid = sess["conv_id"]
            if cid not in emb_cache:
                continue
            emb = emb_cache[cid].cpu().numpy()[:state_dim]
            if state is None:
                state = emb.copy()
            else:
                state = alpha * state + (1.0 - alpha) * emb
        if state is not None:
            vecs.append(state)
            arcs.append(arc)
    return np.stack(vecs).astype(np.float32), arcs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result_row(label: str, r: dict, baseline_ratio: float | None = None) -> None:
    ratio = r["sep_ratio"]
    if baseline_ratio is not None and not np.isnan(baseline_ratio) and baseline_ratio > 0:
        improvement = (ratio - baseline_ratio) / baseline_ratio * 100
        pct_str = f"  ({improvement:+.0f}% vs baseline)"
    else:
        pct_str = ""
    print(
        f"  {label:<30}  within={r['within_dist']:.4f}  "
        f"between={r['between_dist']:.4f}  sep={ratio:.4f}{pct_str}"
    )


def print_report(
    model_result: dict,
    baseline_result: dict,
    ewma_result: dict,
    shuffled: dict,
    model_states: np.ndarray,
    model_arcs: list[str],
) -> None:
    print()
    hline("=")
    print("  Arc Separation Analysis — current model (PR=339.7)")
    hline("=")
    print(f"  Trajectories analyzed: {len(model_arcs):,}")
    print(f"  Arc types:             {len(set(model_arcs))}")
    print(f"  State dimension:       {model_states.shape[1]}")
    hline()
    print(f"  {'Representation':<30}  {'Within':>10}  {'Between':>10}  {'Sep ratio':>10}")
    hline()

    baseline_ratio = baseline_result["sep_ratio"]

    print_result_row("Last-session emb (baseline)", baseline_result, None)
    print_result_row("EWMA alpha=0.3 (no params)",  ewma_result,     baseline_ratio)
    print_result_row("Trained model (final state)", model_result,    baseline_ratio)
    hline()

    # Shuffled control
    sc_mean = shuffled["mean_ratio"]
    sc_std  = shuffled["std_ratio"]
    real    = model_result["sep_ratio"]
    if not np.isnan(sc_mean) and not np.isnan(real):
        collapse_pct = (sc_mean - real) / real * 100
        print(f"  Shuffled-label control ({shuffled['n_shuffles']} runs):")
        print(f"    Shuffled sep ratio:   {sc_mean:.4f} +/- {sc_std:.4f}")
        print(f"    Real sep ratio:       {real:.4f}")
        print(f"    Change after shuffle: {collapse_pct:+.1f}%")
        if collapse_pct < -30:
            print(f"    -> Structure is GENUINE (collapse confirmed)")
        elif collapse_pct < -10:
            print(f"    -> Structure likely real (moderate collapse)")
        else:
            print(f"    -> Weak collapse — may be geometric artifact")
    hline()

    # Paper comparison
    print()
    print("  Comparison with paper (original PR=1.4 model):")
    print(f"  {'Condition':<35}  {'Sep ratio':>10}  {'Shuffled':>10}")
    hline()
    print(f"  {'Original model (PR=1.4)':<35}  {'3.1111':>10}  {'0.9807':>10}")
    print(f"  {'Current model (PR=339.7)':<35}  {real:>10.4f}  {sc_mean:>10.4f}")
    hline()

    # Per-arc breakdown
    print()
    print("  Within-arc cosine distances (current model):")
    hline()
    print(f"  {'Arc':<30}  {'Within dist':>12}  {'N':>6}")
    hline()
    arc_within = model_result["arc_within"]
    for arc in sorted(arc_within, key=lambda a: arc_within[a] if not np.isnan(arc_within[a]) else 999):
        d = arc_within[arc]
        d_str = f"{d:.4f}" if not np.isnan(d) else "  N/A"
        n = sum(1 for a in model_arcs if a == arc)
        print(f"  {arc:<30}  {d_str:>12}  {n:>6}")
    hline("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Arc separation analysis")
    parser.add_argument("--data",       type=str, default="data/trajectories_10k.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transition_v1.pt")
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--n-shuffles", type=int,   default=10,
                        help="Number of shuffled-label control runs")
    parser.add_argument("--max-pairs",  type=int,   default=80_000,
                        help="Max pairwise comparisons per condition (speed cap)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    data_path = root / args.data
    ckpt_path = root / args.checkpoint

    for p in [data_path, ckpt_path]:
        if not p.exists():
            sys.exit(f"Not found: {p}")

    rng_np  = np.random.default_rng(args.seed)
    rng_py  = random.Random(args.seed)
    device  = torch.device("cpu")

    # Load model
    logger.info("Loading transition model...")
    from kokoro.transition import TransitionModel
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt.get("model_config", {})
    model  = TransitionModel(**{k: v for k, v in cfg.items() if k in ("state_dim", "session_dim", "hidden_dim")})
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    state_dim  = model.STATE_DIM
    use_vad    = cfg.get("session_dim", 384) > 384
    logger.info(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, use_vad={use_vad}")

    # Load encoder
    logger.info("Loading encoder...")
    from kokoro.encoder import SessionEncoder
    from training.train import precompute_embeddings
    encoder = SessionEncoder(device="cpu", use_vad_features=use_vad)

    # Load trajectories
    logger.info(f"Loading {data_path.name}...")
    with open(data_path) as f:
        all_trajs = json.load(f)
    rng_py.shuffle(all_trajs)
    n_val     = int(len(all_trajs) * args.val_split)
    val_trajs = all_trajs[len(all_trajs) - n_val:]
    logger.info(f"  Val set: {len(val_trajs):,} trajectories")

    # Precompute embeddings
    logger.info("Precomputing embeddings...")
    emb_cache = precompute_embeddings(val_trajs, encoder, device)

    # Collect representations
    logger.info("Rolling out trajectories (trained model)...")
    model_states, model_arcs = collect_final_states(model, val_trajs, emb_cache, device)
    logger.info(f"  Collected {len(model_arcs):,} final state vectors")

    logger.info("Collecting last-session embeddings (baseline)...")
    baseline_vecs, baseline_arcs = collect_last_session_embs(val_trajs, emb_cache, state_dim)

    logger.info("Collecting EWMA states (no-params baseline)...")
    ewma_vecs, ewma_arcs = collect_ewma_states(val_trajs, emb_cache, state_dim, alpha=0.3)

    # Compute separation
    logger.info("Computing separation ratios...")
    model_result   = compute_separation(model_states,    model_arcs,    args.max_pairs, rng_np)
    baseline_result = compute_separation(baseline_vecs,  baseline_arcs, args.max_pairs, rng_np)
    ewma_result    = compute_separation(ewma_vecs,       ewma_arcs,     args.max_pairs, rng_np)

    logger.info(f"Running shuffled-label control ({args.n_shuffles} shuffles)...")
    shuffled = shuffled_control(model_states, model_arcs, args.n_shuffles,
                                max(10_000, args.max_pairs // 4), rng_np)

    print_report(model_result, baseline_result, ewma_result, shuffled, model_states, model_arcs)
