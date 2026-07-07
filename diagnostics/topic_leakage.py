"""
diagnostics/topic_leakage.py

Test whether emotional retrieval (cosine(state, session_emb)) has collapsed
into semantic retrieval (cosine(current_session_emb, session_emb)).

If the two retrieval axes are highly correlated (Spearman r > 0.8), VICReg
expanded the state space but the emotional retrieval feature is essentially
redundant with semantic retrieval — the product's core differentiator is broken.

Method:
  For each query trajectory in a sample, take the final state s_T and the
  final session embedding e_T. Score all other sessions in the val pool by:
    - emotional score: cosine(s_T, e_i)
    - semantic score:  cosine(e_T, e_i)
  Compute Spearman rank correlation between the two score vectors.
  Report mean correlation and distribution.

Run from project root:
    python -m diagnostics.topic_leakage
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


def spearmanr(a: np.ndarray, b: np.ndarray) -> tuple[float, None]:
    """Spearman rank correlation via numpy (drop-in for scipy.stats.spearmanr).

    scipy is deliberately not imported here: loading scipy before
    sentence-transformers segfaults on this project's target environment
    (OpenMP runtime clash with torch).
    """
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan"), None
    return float(np.corrcoef(ra, rb)[0, 1]), None

sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnostics.common import (
    load_transition_model,
    make_encoder,
    build_embedding_cache,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).parent.parent
LEAKAGE_THRESHOLD = 0.8   # Spearman r above this → flag


def get_final_states(model, trajectories, emb_cache):
    """Return (state_T, emb_T[:384]) for each trajectory's final session.

    Embeddings may be 387-dim (VAD checkpoint); the semantic comparison axis
    uses only the MiniLM portion so both sides live in the same 384-dim space.
    """
    results = []
    model.eval()
    D = model.state_dim
    with torch.no_grad():
        for traj in trajectories:
            sessions = traj["sessions"]
            state = model.initial_state()
            for sess in sessions:
                emb   = emb_cache[sess["conv_id"]]
                state = model(state, emb)
            state_n = F.normalize(state, dim=-1).numpy()
            emb_T_n = emb_cache[sessions[-1]["conv_id"]].numpy()[:D]
            emb_T_n = emb_T_n / (np.linalg.norm(emb_T_n) + 1e-12)
            results.append((state_n, emb_T_n))
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Topic leakage: emotional vs semantic retrieval axes")
    parser.add_argument("--data",       default=str(PROJECT_ROOT / "data" / "trajectories_10k_v2_val.json"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "transition_v2.pt"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-queries",  type=int, default=200,
                        help="Number of query trajectories to sample (default 200)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    for p in [Path(args.checkpoint), Path(args.data)]:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    with open(args.data) as f:
        all_trajs = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)
    n_val     = int(len(all_trajs) * args.val_fraction)
    val_trajs = all_trajs[-n_val:]
    logger.info(f"Val set: {n_val} trajectories")

    model, cfg = load_transition_model(Path(args.checkpoint))
    encoder    = make_encoder(cfg)

    emb_cache = build_embedding_cache(val_trajs, encoder)

    logger.info("Computing final states for all val trajectories...")
    final_state_emb = get_final_states(model, val_trajs, emb_cache)

    # Pool of all unique session embeddings to score against (MiniLM portion only)
    all_conv_ids  = list(emb_cache.keys())
    pool_matrix   = np.stack(
        [emb_cache[c].numpy()[: model.state_dim] for c in all_conv_ids]
    )   # (M, 384)
    # Pre-normalize pool
    norms = np.linalg.norm(pool_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pool_normed = pool_matrix / norms    # (M, 384)

    n_queries = min(args.n_queries, len(final_state_emb))
    query_indices = rng.sample(range(len(final_state_emb)), n_queries)

    logger.info(f"Computing retrieval rank correlations for {n_queries} queries...")
    correlations = []

    for idx in query_indices:
        state_n, emb_n = final_state_emb[idx]    # both unit-norm

        emotional_scores = pool_normed @ state_n  # (M,)
        semantic_scores  = pool_normed @ emb_n    # (M,)

        r, _ = spearmanr(emotional_scores, semantic_scores)
        if not np.isnan(r):
            correlations.append(float(r))

    correlations = np.array(correlations)
    mean_r  = float(np.mean(correlations))
    med_r   = float(np.median(correlations))
    pct90_r = float(np.percentile(correlations, 90))

    print()
    print("=" * 65)
    print("  Topic Leakage: Emotional vs Semantic Retrieval Axes")
    print("=" * 65)
    print(f"  Queries evaluated:          {len(correlations)}")
    print(f"  Mean Spearman r:            {mean_r:.4f}")
    print(f"  Median Spearman r:          {med_r:.4f}")
    print(f"  90th percentile r:          {pct90_r:.4f}")
    print()
    print(f"  Distribution of correlations:")
    bins = [(-1.0, 0.0), (0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = int(((correlations >= lo) & (correlations < hi)).sum())
        pct   = count / len(correlations) * 100
        bar   = "#" * (count // max(1, len(correlations) // 40))
        print(f"    r in [{lo:+.1f}, {hi:+.2f}):  {count:>4} ({pct:5.1f}%)  {bar}")
    print()

    if mean_r > LEAKAGE_THRESHOLD:
        print(f"  FLAG: mean r={mean_r:.4f} > {LEAKAGE_THRESHOLD} threshold.")
        print(f"  Emotional retrieval is highly correlated with semantic retrieval.")
        print(f"  The state vector is not adding an independent emotional axis.")
        print(f"  VICReg may have expanded dimensions but they track semantic content.")
    elif mean_r > 0.6:
        print(f"  WARN: mean r={mean_r:.4f}. Some leakage — axes are not fully independent.")
        print(f"  Emotional retrieval has partial independence from semantic retrieval.")
    else:
        print(f"  OK: mean r={mean_r:.4f} < {LEAKAGE_THRESHOLD}.")
        print(f"  Emotional and semantic retrieval axes are meaningfully independent.")
        print(f"  The state vector adds information beyond the current session embedding.")

    print("=" * 65)


if __name__ == "__main__":
    main()
