"""
Validation experiment for Kokoro transition model.
Produces Table 1 (core) and Table 1 (Extended) for the research paper.

Table 1 — core comparison:
  Baseline  Last-session encoder embedding as "state" (no transition model)
  Model     Full transition model applied sequentially to all sessions

  Metrics (both methods, val split):
    within_arc_dist   Mean pairwise cosine distance between final vectors of
                      the SAME arc type. Lower = more consistent arc repr.
    between_arc_dist  Mean pairwise cosine distance between final vectors of
                      DIFFERENT arc types. Higher = arcs more distinguishable.
    separation_ratio  between_arc_dist / within_arc_dist. Headline metric.

  Secondary metric — PC1 / valence polarity:
    Arcs labelled net_positive (end valence > 0) or net_negative (<= 0).
    Point-biserial r_pb between PC1 and polarity; Fisher-z 95% CI.

Table 1 (Extended) — additional baselines and controls:
  Identity / persistence baseline
    State never updates from cold-start zeros.  Floor null control.
    Tests: is "nothing changed" already a strong predictor?

  EWMA baseline (alpha=0.3)
    state_t = normalize(0.3 * emb_t + 0.7 * state_{t-1})
    No learned parameters.  Tests whether a parameter-free moving average
    beats the trained model.

  Shuffled-label control
    Model state vectors unchanged; arc_type labels randomly permuted.
    Tests whether separation reflects genuine emotional structure or
    arc-construction artefacts (trajectory length, session ordering).

  Silhouette score
    sklearn silhouette_score (cosine) for all four conditions.

  Effective dimensionality
    Participation ratio = (sum lambda_i)^2 / sum(lambda_i^2), range [1, 384].
    Plus variance explained by PC1–PC5.

Run from project root:
    python -m training.validate

Requires:
  data/trajectories_10k.json       (or whatever --data points to)
  checkpoints/transition_v1_10k.pt (or whatever --checkpoint points to)
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
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.transition import TransitionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arc valence polarity — derived directly from ARC_TEMPLATES end-zone valence.
# net_positive: end valence > 0
# net_negative: end valence <= 0
# This is the only place arc structure enters the validation — and it is used
# only to label groups for the secondary metric, not in any model component.
# ---------------------------------------------------------------------------

from data.arc_templates import ARC_TEMPLATES

ARC_END_VALENCE: dict[str, float] = {
    t.name: t.sessions[-1].valence for t in ARC_TEMPLATES
}

ARC_POLARITY: dict[str, str] = {
    name: ("net_positive" if v > 0.0 else "net_negative")
    for name, v in ARC_END_VALENCE.items()
}


# ---------------------------------------------------------------------------
# Embedding precomputation
# ---------------------------------------------------------------------------

def build_embedding_cache(
    trajectories: list[dict],
    encoder,
    device: torch.device,
    chunk_size: int = 1500,
) -> dict[str, torch.Tensor]:
    """Encode all unique sessions; return conv_id -> (384,) tensor."""
    from kokoro.encoder import decode_parlai_artifacts

    unique: dict[str, list] = {}
    for traj in trajectories:
        for sess in traj["sessions"]:
            if sess["conv_id"] not in unique:
                unique[sess["conv_id"]] = sess["turns"]

    all_ids = list(unique.keys())
    n = len(all_ids)
    logger.info(f"  Encoding {n} unique sessions...")
    t0 = time.perf_counter()
    cache: dict[str, torch.Tensor] = {}

    for start in range(0, n, chunk_size):
        chunk_ids = all_ids[start: start + chunk_size]
        texts_all, slices, weights_all = [], [], []

        for conv_id in chunk_ids:
            texts, weights = [], []
            for turn in unique[conv_id]:
                cleaned = decode_parlai_artifacts(turn.get("content", ""))
                if cleaned:
                    texts.append(cleaned)
                    weights.append(2.0 if turn.get("role") == "user" else 1.0)
            if not texts:
                texts, weights = [""], [1.0]
            s = len(texts_all)
            texts_all.extend(texts)
            slices.append((s, s + len(texts)))
            weights_all.append(weights)

        embs = encoder.model.encode(
            texts_all, batch_size=256, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=False,
        )
        for i, conv_id in enumerate(chunk_ids):
            s, e = slices[i]
            w = np.array(weights_all[i], dtype=np.float32)
            w /= w.sum()
            pooled = (embs[s:e] * w[:, np.newaxis]).sum(axis=0).astype(np.float32)
            cache[conv_id] = torch.tensor(pooled, device=device)

    logger.info(f"  Done in {time.perf_counter()-t0:.1f}s")
    return cache


# ---------------------------------------------------------------------------
# Final-vector extraction
# ---------------------------------------------------------------------------

def get_baseline_vectors(
    trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
) -> tuple[list[np.ndarray], list[str]]:
    """
    Baseline: last session's raw encoder embedding, L2-normalised.
    This is the "no transition model" control — it has access to exactly as
    much text information as the model, but no sequential state integration.
    """
    vecs, labels = [], []
    for traj in trajectories:
        last_conv_id = traj["sessions"][-1]["conv_id"]
        v = emb_cache[last_conv_id]
        v_norm = F.normalize(v, dim=0).cpu().numpy()
        vecs.append(v_norm)
        labels.append(traj["arc_name"])
    return vecs, labels


def get_model_vectors(
    trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
    model,
    device: torch.device,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Model: run all sessions through transition model; use final state vector.
    Output is already L2-normalised (TransitionModel always normalises).
    """
    vecs, labels = [], []
    model.eval()
    with torch.no_grad():
        for traj in trajectories:
            state = TransitionModel.initial_state(device=device)
            for sess in traj["sessions"]:
                emb = emb_cache[sess["conv_id"]]
                state = model(state, emb)
            vecs.append(state.cpu().numpy())
            labels.append(traj["arc_name"])
    return vecs, labels


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2] for L2-normalised vectors: 1 - cos_sim."""
    return float(1.0 - np.dot(a, b))


class DistanceResult(NamedTuple):
    within_arc_dist:   float
    between_arc_dist:  float
    separation_ratio:  float
    within_per_arc:    dict[str, float]     # per-arc within-distance
    n_within_pairs:    int
    n_between_pairs:   int


def compute_separation_metrics(
    vectors: list[np.ndarray],
    labels: list[str],
    max_pairs_per_cell: int = 2000,
    seed: int = 42,
) -> DistanceResult:
    """
    Compute within-arc and between-arc cosine distances.

    For each arc pair (same / different), we sample up to max_pairs_per_cell
    random pairs to keep runtime manageable when there are many trajectories.

    Args:
        vectors:             List of L2-normalised (384,) arrays.
        labels:              Parallel list of arc name strings.
        max_pairs_per_cell:  Max pairs sampled per (arc_i, arc_j) cell.
        seed:                RNG seed.

    Returns:
        DistanceResult with within/between distances and separation ratio.
    """
    rng = random.Random(seed)

    # Group indices by arc
    by_arc: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_arc[label].append(i)

    arc_names = sorted(by_arc.keys())

    within_dists: list[float] = []
    between_dists: list[float] = []
    within_per_arc: dict[str, list[float]] = {arc: [] for arc in arc_names}

    def _sample_pairs(
        idx_a: list[int], idx_b: list[int], same: bool
    ) -> list[tuple[int, int]]:
        """Generate up to max_pairs_per_cell pairs from idx_a x idx_b."""
        if same:
            # Within-arc: pairs from same group without replacement
            all_pairs = [
                (idx_a[i], idx_a[j])
                for i in range(len(idx_a))
                for j in range(i + 1, len(idx_a))
            ]
        else:
            all_pairs = [(a, b) for a in idx_a for b in idx_b]
        if len(all_pairs) > max_pairs_per_cell:
            return rng.sample(all_pairs, max_pairs_per_cell)
        return all_pairs

    # Within-arc distances (diagonal cells)
    for arc in arc_names:
        idx = by_arc[arc]
        if len(idx) < 2:
            continue
        for i, j in _sample_pairs(idx, idx, same=True):
            d = cosine_distance(vectors[i], vectors[j])
            within_dists.append(d)
            within_per_arc[arc].append(d)

    # Between-arc distances (off-diagonal cells)
    for k, arc_a in enumerate(arc_names):
        for arc_b in arc_names[k + 1:]:
            for i, j in _sample_pairs(by_arc[arc_a], by_arc[arc_b], same=False):
                between_dists.append(cosine_distance(vectors[i], vectors[j]))

    within_mean  = float(np.mean(within_dists))  if within_dists  else 0.0
    between_mean = float(np.mean(between_dists)) if between_dists else 0.0
    ratio = between_mean / within_mean if within_mean > 0 else float("nan")

    return DistanceResult(
        within_arc_dist   = within_mean,
        between_arc_dist  = between_mean,
        separation_ratio  = ratio,
        within_per_arc    = {arc: float(np.mean(v)) if v else float("nan")
                             for arc, v in within_per_arc.items()},
        n_within_pairs    = len(within_dists),
        n_between_pairs   = len(between_dists),
    )


# ---------------------------------------------------------------------------
# PC1 / polarity correlation
# ---------------------------------------------------------------------------

class PolarityResult(NamedTuple):
    r_pb:          float    # point-biserial correlation
    p_value:       float
    ci_low:        float    # 95% CI lower bound (Fisher z)
    ci_high:       float    # 95% CI upper bound
    n_positive:    int
    n_negative:    int
    pc1_var_exp:   float    # % variance explained by PC1


def compute_polarity_correlation(
    vectors: list[np.ndarray],
    labels: list[str],
) -> PolarityResult:
    """
    Fit PCA on the final state vectors, then test whether PC1 correlates
    with arc valence polarity (net_positive vs net_negative).

    Uses point-biserial correlation (equivalent to Pearson r between a
    continuous and a binary variable) with Fisher-z 95% CI.

    Arcs not present in ARC_POLARITY are excluded silently.
    """
    from sklearn.decomposition import PCA
    from scipy.stats import pointbiserialr

    # Filter to arcs with known polarity
    vecs_filtered, binary_labels = [], []
    for vec, arc in zip(vectors, labels):
        if arc in ARC_POLARITY:
            vecs_filtered.append(vec)
            binary_labels.append(1 if ARC_POLARITY[arc] == "net_positive" else 0)

    if len(vecs_filtered) < 10:
        raise ValueError(f"Too few labelled vectors: {len(vecs_filtered)}")

    mat = np.stack(vecs_filtered, axis=0)   # (N, 384)

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(mat).ravel()    # (N,)
    var_exp = float(pca.explained_variance_ratio_[0]) * 100

    binary_arr = np.array(binary_labels, dtype=float)

    r_pb, p_val = pointbiserialr(binary_arr, pc1)

    # Sign convention: positive r means PC1 is higher for net_positive arcs.
    # Flip sign if needed so the correlation is always positive (so that
    # "higher PC1 = more positive" is the consistent interpretation).
    if r_pb < 0:
        r_pb = -r_pb
        pc1  = -pc1   # flip projection for CI calculation consistency

    # Fisher-z 95% confidence interval
    n = len(pc1)
    z  = np.arctanh(np.clip(r_pb, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    ci_lo = float(np.tanh(z_lo))
    ci_hi = float(np.tanh(z_hi))

    return PolarityResult(
        r_pb        = float(r_pb),
        p_value     = float(p_val),
        ci_low      = ci_lo,
        ci_high     = ci_hi,
        n_positive  = int(binary_arr.sum()),
        n_negative  = int((1 - binary_arr).sum()),
        pc1_var_exp = var_exp,
    )


# ---------------------------------------------------------------------------
# Additional baselines and controls
# ---------------------------------------------------------------------------

def get_identity_vectors(
    trajectories: list[dict],
    state_dim: int = 384,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Identity / persistence baseline: state never updates from cold-start zeros.

    Every trajectory produces the same final vector (the all-zero initial_state).
    This is the null-floor baseline: without any integration of session content
    the representation carries no arc-discriminative information.

    Consequence: within-arc distance == between-arc distance == 1.0 (cosine
    distance between two zero vectors = 1 - dot(0,0) = 1.0), so separation
    ratio = 1.0.  Silhouette is undefined (all vectors identical).
    """
    zero_vec = np.zeros(state_dim, dtype=np.float32)
    vecs   = [zero_vec.copy() for _ in trajectories]
    labels = [traj["arc_name"] for traj in trajectories]
    return vecs, labels


def get_ewma_vectors(
    trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
    alpha: float = 0.3,
) -> tuple[list[np.ndarray], list[str]]:
    """
    EWMA baseline (no learned parameters).

    At each session t:
        state_t = alpha * normalize(emb_t) + (1 - alpha) * state_{t-1}
        state_t = normalize(state_t)

    Re-normalising after each blend keeps the vector on the unit sphere,
    making the final vectors directly comparable to the trained model output.
    alpha=0.3 means 70% persistence (slow decay of older sessions).
    """
    vecs, labels = [], []
    with torch.no_grad():
        for traj in trajectories:
            state = None
            for sess in traj["sessions"]:
                emb   = emb_cache[sess["conv_id"]]
                emb_n = F.normalize(emb, dim=0)
                if state is None:
                    state = emb_n
                else:
                    state = alpha * emb_n + (1.0 - alpha) * state
                    state = F.normalize(state, dim=0)
            if state is None:
                state = torch.zeros(384)
            vecs.append(state.cpu().numpy())
            labels.append(traj["arc_name"])
    return vecs, labels


def compute_silhouette(
    vectors: list[np.ndarray],
    labels: list[str],
) -> float:
    """
    Silhouette score (cosine distance, arc_type labels).
    Returns NaN when the score is undefined (all vectors identical, or < 2 labels).
    """
    from sklearn.metrics import silhouette_score

    mat = np.stack(vectors, axis=0)

    if np.allclose(mat, mat[0]):          # all vectors identical (e.g. identity baseline)
        return float("nan")
    if len(set(labels)) < 2:
        return float("nan")

    try:
        return float(silhouette_score(mat, labels, metric="cosine"))
    except Exception:
        return float("nan")


def compute_shuffled_separation(
    vectors: list[np.ndarray],
    labels: list[str],
    max_pairs_per_cell: int = 2000,
    seed: int = 42,
) -> DistanceResult:
    """
    Shuffled-label control on the model's final state vectors.

    Randomly permutes arc_type labels, then recomputes separation metrics on
    the same (unchanged) state vectors.

    If separation ratio collapses after shuffling, the model learned genuine
    emotional structure (label identity is load-bearing).
    If separation stays high, the model may have learned arc-construction
    artefacts (trajectory length, session ordering) that are label-independent.
    """
    rng = random.Random(seed + 999)   # distinct offset so shuffle != train/val split
    shuffled = labels[:]
    rng.shuffle(shuffled)
    return compute_separation_metrics(
        vectors, shuffled, max_pairs_per_cell=max_pairs_per_cell, seed=seed
    )


class EffDimResult(NamedTuple):
    participation_ratio: float        # (sum λ_i)² / sum(λ_i²), range [1, D]
    var_exp_per_pc:      list[float]  # % variance explained by PC1 .. PC5
    cumulative_var_exp:  float        # cumulative PC1–PC5 variance %
    n_components_fit:    int          # number of PCA components actually fitted


def compute_effective_dimensionality(
    vectors: list[np.ndarray],
    n_pcs: int = 5,
) -> EffDimResult:
    """
    Participation ratio of the model's final state vectors.

    Participation ratio = (Σ λ_i)² / Σ (λ_i²)
    where λ_i are eigenvalues of the sample covariance matrix (PCA explained variances).

    Interpretation:
      PR = 1.0   All variance concentrated along one axis (collapsed space)
      PR = 384.0 Variance perfectly uniform across all dimensions (maximal richness)

    Also reports variance explained by PC1 through n_pcs.
    """
    from sklearn.decomposition import PCA

    mat = np.stack(vectors, axis=0)                        # (N, D)
    n_components = min(len(vectors) - 1, mat.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(mat)

    eigenvalues = pca.explained_variance_                  # λ_i (covariance matrix)
    pr = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    ratios      = pca.explained_variance_ratio_
    var_exp_pcs = [
        float(ratios[i] * 100) if i < len(ratios) else 0.0
        for i in range(n_pcs)
    ]
    cumulative = float(ratios[:n_pcs].sum() * 100)

    return EffDimResult(
        participation_ratio = pr,
        var_exp_per_pc      = var_exp_pcs,
        cumulative_var_exp  = cumulative,
        n_components_fit    = n_components,
    )


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table(
    baseline_dist: DistanceResult,
    model_dist:    DistanceResult,
    baseline_pol:  PolarityResult,
    model_pol:     PolarityResult,
    n_val:         int,
    arc_names:     list[str],
) -> None:
    W = 72

    def hline(char="-"):
        print(char * W)

    def row(label, b_val, m_val, fmt=".4f", highlight_higher=True):
        b_str = f"{b_val:{fmt}}"
        m_str = f"{m_val:{fmt}}"
        if highlight_higher:
            winner = "model" if m_val > b_val else "base"
        else:
            winner = "model" if m_val < b_val else "base"
        marker = " <-- better" if winner == "model" else ""
        print(f"  {label:<34} {b_str:>10}   {m_str:>10}{marker}")

    print()
    hline("=")
    print(f"  Table 1: Transition Model Validation  (n_val = {n_val})")
    hline("=")
    print(f"  {'Metric':<34} {'Baseline':>10}   {'Model':>10}")
    print(f"  {'(last-session embedding)':<34} {'':>10}   {'(full traj)':>10}")
    hline()
    print("  ARC SEPARATION")
    row("Within-arc cosine dist",     baseline_dist.within_arc_dist,  model_dist.within_arc_dist,  highlight_higher=False)
    row("Between-arc cosine dist",    baseline_dist.between_arc_dist, model_dist.between_arc_dist, highlight_higher=True)
    row("Separation ratio (B/W)",     baseline_dist.separation_ratio, model_dist.separation_ratio, highlight_higher=True)
    print(f"  {'Within pairs evaluated':<34} {baseline_dist.n_within_pairs:>10,}   {model_dist.n_within_pairs:>10,}")
    print(f"  {'Between pairs evaluated':<34} {baseline_dist.n_between_pairs:>10,}   {model_dist.n_between_pairs:>10,}")
    hline()
    print("  PC1 / VALENCE POLARITY CORRELATION")
    row("Point-biserial r_pb",        baseline_pol.r_pb,      model_pol.r_pb,      highlight_higher=True)

    def p_str(p):
        if   p < 0.001: return "< 0.001"
        elif p < 0.01:  return f"{p:.3f}"
        else:           return f"{p:.3f}"

    print(f"  {'p-value':<34} {p_str(baseline_pol.p_value):>10}   {p_str(model_pol.p_value):>10}")
    print(f"  {'95% CI lower':<34} {baseline_pol.ci_low:>10.4f}   {model_pol.ci_low:>10.4f}")
    print(f"  {'95% CI upper':<34} {baseline_pol.ci_high:>10.4f}   {model_pol.ci_high:>10.4f}")
    print(f"  {'PC1 variance explained (%)':<34} {baseline_pol.pc1_var_exp:>10.1f}   {model_pol.pc1_var_exp:>10.1f}")
    print(f"  {'n_positive arcs':<34} {baseline_pol.n_positive:>10}   {model_pol.n_positive:>10}")
    print(f"  {'n_negative arcs':<34} {baseline_pol.n_negative:>10}   {model_pol.n_negative:>10}")
    hline()

    print("  WITHIN-ARC DISTANCE BY ARC TYPE")
    print(f"  {'Arc':<30} {'Baseline':>10}   {'Model':>10}   {'Delta':>8}")
    hline("-")
    for arc in sorted(arc_names):
        b = baseline_dist.within_per_arc.get(arc, float("nan"))
        m = model_dist.within_per_arc.get(arc, float("nan"))
        if np.isnan(b) or np.isnan(m):
            continue
        delta = m - b
        arrow = " v" if delta < 0 else " ^"
        polarity = ARC_POLARITY.get(arc, "?")
        pol_tag = "(+)" if polarity == "net_positive" else "(-)"
        print(f"  {arc:<30} {b:>10.4f}   {m:>10.4f}   {delta:>+8.4f}{arrow}  {pol_tag}")
    hline()

    print("  INTERPRETATION NOTES")
    print(f"  Separation ratio > 1.0 means between-arc > within-arc (desired).")
    sr_b = baseline_dist.separation_ratio
    sr_m = model_dist.separation_ratio
    if sr_m > sr_b:
        lift = (sr_m - sr_b) / sr_b * 100
        print(f"  Model improves separation ratio by {lift:.1f}% over baseline.")
    else:
        diff = (sr_m - sr_b) / sr_b * 100
        print(f"  Model changes separation ratio by {diff:+.1f}% vs baseline.")
    if model_pol.p_value < 0.05:
        print(f"  PC1 / polarity correlation is significant (p {p_str(model_pol.p_value)}).")
    else:
        print(f"  PC1 / polarity correlation is NOT significant (p = {model_pol.p_value:.3f}).")
    hline("=")


# ---------------------------------------------------------------------------
# Extended table (additional baselines and controls)
# ---------------------------------------------------------------------------

def print_extended_table(
    baseline_dist: DistanceResult,
    identity_dist: DistanceResult,
    ewma_dist:     DistanceResult,
    model_dist:    DistanceResult,
    shuffled_dist: DistanceResult,
    baseline_sil:  float,
    identity_sil:  float,
    ewma_sil:      float,
    model_sil:     float,
    eff_dim:       EffDimResult,
    n_val:         int,
) -> None:
    W = 78

    def hline(char="-"):
        print(char * W)

    def _sil(s: float) -> str:
        return "undefined" if np.isnan(s) else f"{s:.4f}"

    def _ratio(r: float) -> str:
        return "undefined" if np.isnan(r) else f"{r:.4f}"

    # ------------------------------------------------------------------ #
    #  Condition comparison table                                          #
    # ------------------------------------------------------------------ #
    print()
    hline("=")
    print(f"  Table 1 (Extended): Additional Baselines and Controls   (n_val = {n_val})")
    hline("=")
    print("  CONDITION COMPARISON")
    print()
    print(
        f"  {'Condition':<25} {'Silhouette':>10}  "
        f"{'Sep.Ratio':>10}  {'Within':>8}  {'Between':>8}"
    )
    hline()

    conditions = [
        ("Baseline (last-sess)",  baseline_sil, baseline_dist),
        ("Identity (null floor)", identity_sil, identity_dist),
        ("EWMA (alpha=0.3)",      ewma_sil,     ewma_dist),
        ("Trained model",         model_sil,    model_dist),
    ]
    for name, sil, dist in conditions:
        print(
            f"  {name:<25} {_sil(sil):>10}  "
            f"{_ratio(dist.separation_ratio):>10}  "
            f"{dist.within_arc_dist:>8.4f}  "
            f"{dist.between_arc_dist:>8.4f}"
        )
    hline()
    print()
    print("  Notes:")
    print("  - Identity: state = zeros (cold-start), never updated. All trajectories")
    print("    produce the same vector -> silhouette undefined, sep.ratio = 1.0.")
    print("  - EWMA: no learned parameters; blends session embeddings on the sphere.")
    print("  - Silhouette in [-1, 1]; higher = more arc-discriminative structure.")
    print("  - Sep.Ratio > 1 means between-arc spread exceeds within-arc spread.")

    # ------------------------------------------------------------------ #
    #  Shuffled-label control                                              #
    # ------------------------------------------------------------------ #
    print()
    hline("=")
    print("  SHUFFLED-LABEL CONTROL")
    hline("=")
    print("  Arc labels randomly permuted; model state vectors unchanged.")
    print("  If sep.ratio collapses: the model learned genuine emotional structure")
    print("  (label identity is load-bearing for the observed separation).")
    print("  If sep.ratio stays high: separation may reflect arc-construction")
    print("  artefacts (trajectory length, session ordering) independent of content.")
    print()

    true_r     = model_dist.separation_ratio
    shuffled_r = shuffled_dist.separation_ratio
    if true_r > 0 and not np.isnan(true_r):
        drop_pct = (true_r - shuffled_r) / true_r * 100
    else:
        drop_pct = float("nan")

    print(f"  {'Separation ratio (true labels)':<44} {true_r:>8.4f}")
    print(f"  {'Separation ratio (shuffled labels)':<44} {shuffled_r:>8.4f}")
    if np.isnan(drop_pct):
        print(f"  {'Relative drop after shuffling':<44} {'undefined':>8}")
    else:
        print(f"  {'Relative drop after shuffling':<44} {drop_pct:>7.1f}%")
    hline()
    print()
    if not np.isnan(drop_pct):
        if drop_pct > 20:
            print(
                f"  Ratio dropped {drop_pct:.1f}% after label shuffle. The separation"
            )
            print(
                f"  depends on correct arc labelling, consistent with learned emotional"
            )
            print(f"  structure rather than arc-construction artefacts.")
        else:
            print(
                f"  Ratio dropped only {drop_pct:.1f}% after label shuffle. Separation"
            )
            print(
                f"  is largely label-independent; the model may be responding to"
            )
            print(
                f"  arc-construction patterns (length, order) rather than content."
            )

    # ------------------------------------------------------------------ #
    #  Effective dimensionality                                            #
    # ------------------------------------------------------------------ #
    print()
    hline("=")
    print("  EFFECTIVE DIMENSIONALITY  (trained model final state vectors)")
    hline("=")
    print("  Participation ratio = (sum lambda_i)^2 / sum(lambda_i^2)")
    print("  Range: 1.0 = all variance in one direction, 384.0 = maximally distributed.")
    print()

    pr     = eff_dim.participation_ratio
    pr_pct = pr / 384.0 * 100
    print(f"  {'Participation ratio':<44} {pr:>8.1f}  / 384.0")
    print(f"  {'  (% of maximum possible)':<44} {pr_pct:>7.1f}%")
    print()
    for i, var in enumerate(eff_dim.var_exp_per_pc, start=1):
        print(f"  {'Variance explained by PC' + str(i):<44} {var:>7.1f}%")
    print(f"  {'Cumulative PC1-PC5':<44} {eff_dim.cumulative_var_exp:>7.1f}%")
    hline()
    print()
    if pr < 10:
        print(
            f"  PR={pr:.1f}: state space is strongly collapsed. Arc structure is"
        )
        print(f"  concentrated along fewer than ~10 effective dimensions.")
    elif pr < 50:
        print(
            f"  PR={pr:.1f}: moderate effective dimensionality ({pr_pct:.1f}% of 384 dims)."
        )
        print(f"  The model uses a modest but non-trivial fraction of its capacity.")
    else:
        print(
            f"  PR={pr:.1f}: high effective dimensionality ({pr_pct:.1f}% of 384 dims)."
        )
        print(f"  Arc structure is distributed across a large portion of the state space.")

    hline("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Validate Kokoro transition model")
    parser.add_argument(
        "--data", type=str,
        default=str(Path(__file__).parent.parent / "data" / "trajectories_10k.json"),
        help="Trajectories JSON path",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Path(__file__).parent.parent / "checkpoints" / "transition_v1_10k.pt"),
        help="Transition model checkpoint",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of trajectories to use as validation set (default: 0.2)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=2000,
        help="Max pairs sampled per arc-pair cell for distance computation (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)

    for p in [data_path, ckpt_path]:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    # ----- Load trajectories -----
    logger.info(f"Loading trajectories from {data_path.name}...")
    with open(data_path) as f:
        all_trajectories = json.load(f)

    rng_split = random.Random(args.seed)
    rng_split.shuffle(all_trajectories)
    n_val = int(len(all_trajectories) * args.val_fraction)
    val_trajectories = all_trajectories[-n_val:]
    logger.info(f"  {n_val} validation trajectories ({len(set(t['arc_name'] for t in val_trajectories))} arc types)")

    # ----- Load model -----
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg  = ckpt.get("model_config", {"state_dim": 384, "hidden_dim": 512})
    model = TransitionModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"  Model: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # ----- Load encoder -----
    from kokoro.encoder import SessionEncoder
    encoder = SessionEncoder(device="cpu")
    _ = encoder.model

    # ----- Encode sessions -----
    logger.info("Encoding val sessions...")
    emb_cache = build_embedding_cache(val_trajectories, encoder, device)

    # ----- Extract final vectors -----
    logger.info("Extracting baseline vectors (last-session embedding)...")
    baseline_vecs, baseline_labels = get_baseline_vectors(val_trajectories, emb_cache)

    logger.info("Extracting model vectors (full transition model)...")
    model_vecs, model_labels = get_model_vectors(val_trajectories, emb_cache, model, device)

    assert baseline_labels == model_labels, "Label mismatch between baseline and model"

    # ----- Compute separation metrics -----
    logger.info("Computing separation metrics...")
    baseline_dist = compute_separation_metrics(
        baseline_vecs, baseline_labels, max_pairs_per_cell=args.max_pairs, seed=args.seed
    )
    model_dist = compute_separation_metrics(
        model_vecs, model_labels, max_pairs_per_cell=args.max_pairs, seed=args.seed
    )

    # ----- Compute polarity correlation -----
    logger.info("Computing PC1 / polarity correlation...")
    baseline_pol = compute_polarity_correlation(baseline_vecs, baseline_labels)
    model_pol    = compute_polarity_correlation(model_vecs,    model_labels)

    # ----- Additional baselines and controls -----
    state_dim = cfg.get("state_dim", 384)

    logger.info("Extracting identity vectors (null floor baseline)...")
    identity_vecs, identity_labels = get_identity_vectors(val_trajectories, state_dim)

    logger.info("Extracting EWMA vectors (alpha=0.3, no learned parameters)...")
    ewma_vecs, ewma_labels = get_ewma_vectors(val_trajectories, emb_cache, alpha=0.3)

    assert identity_labels == baseline_labels, "Identity label mismatch"
    assert ewma_labels     == baseline_labels, "EWMA label mismatch"

    logger.info("Computing identity separation metrics...")
    identity_dist = compute_separation_metrics(
        identity_vecs, identity_labels, max_pairs_per_cell=args.max_pairs, seed=args.seed
    )
    logger.info("Computing EWMA separation metrics...")
    ewma_dist = compute_separation_metrics(
        ewma_vecs, ewma_labels, max_pairs_per_cell=args.max_pairs, seed=args.seed
    )
    logger.info("Computing shuffled-label control...")
    shuffled_dist = compute_shuffled_separation(
        model_vecs, model_labels, max_pairs_per_cell=args.max_pairs, seed=args.seed
    )

    logger.info("Computing silhouette scores (4 conditions)...")
    baseline_sil = compute_silhouette(baseline_vecs, baseline_labels)
    identity_sil = compute_silhouette(identity_vecs, identity_labels)
    ewma_sil     = compute_silhouette(ewma_vecs,     ewma_labels)
    model_sil    = compute_silhouette(model_vecs,    model_labels)

    logger.info("Computing effective dimensionality of model state vectors...")
    eff_dim = compute_effective_dimensionality(model_vecs)

    # ----- Print tables -----
    arc_names = sorted(set(baseline_labels))
    print_table(baseline_dist, model_dist, baseline_pol, model_pol, n_val, arc_names)
    print_extended_table(
        baseline_dist = baseline_dist,
        identity_dist = identity_dist,
        ewma_dist     = ewma_dist,
        model_dist    = model_dist,
        shuffled_dist = shuffled_dist,
        baseline_sil  = baseline_sil,
        identity_sil  = identity_sil,
        ewma_sil      = ewma_sil,
        model_sil     = model_sil,
        eff_dim       = eff_dim,
        n_val         = n_val,
    )
