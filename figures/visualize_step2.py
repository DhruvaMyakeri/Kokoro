"""
Step 2 visualizations: session encoder inspection.

Produces two publication-quality figures:
  fig4_tsne_embeddings.{png,pdf}    — t-SNE of 500 encoded sessions
  fig5_valence_correlation.{png,pdf} — ground-truth valence vs PC1 of embeddings

Run from project root:
    python figures/visualize_step2.py
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prepare import (
    EMOTION_TO_CIRCUMPLEX,
    load_empathetic_dialogues,
    LabeledSession,
)
from kokoro.encoder import SessionEncoder

# Import the same color/quadrant scheme from step1 so figures are consistent
from figures.visualize_step1 import (
    apply_style,
    build_emotion_colors,
    emotion_to_quadrant,
    _QUADRANT_BASE_COLORS,
    _QUADRANT_LABELS,
    _Q1_EMOTIONS, _Q2_EMOTIONS, _Q3_EMOTIONS, _Q4_EMOTIONS,
    save,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Balanced sample construction
# ---------------------------------------------------------------------------

def load_balanced_sample(
    sessions: list[LabeledSession],
    n_total: int = 500,
    seed: int = 42,
) -> list[LabeledSession]:
    """
    Sample up to n_total sessions with equal representation across emotion
    labels. With 32 labels and n_total=500, this gives ~15 sessions per label.

    If a label has fewer than its allocation, we take all of them and
    distribute the remainder to other labels.
    """
    rng = np.random.RandomState(seed)

    by_label: dict[str, list[LabeledSession]] = defaultdict(list)
    for s in sessions:
        by_label[s.emotion_label].append(s)

    n_labels = len(by_label)
    per_label = max(1, n_total // n_labels)

    sample: list[LabeledSession] = []
    for label_sessions in by_label.values():
        idx = rng.choice(len(label_sessions), size=min(per_label, len(label_sessions)),
                         replace=False)
        sample.extend(label_sessions[i] for i in idx)

    # If we're short of n_total (some labels had fewer sessions), top up
    sampled_ids = {s.conv_id for s in sample}
    remaining = [s for s in sessions if s.conv_id not in sampled_ids]
    if len(sample) < n_total and remaining:
        extra_idx = rng.choice(len(remaining),
                               size=min(n_total - len(sample), len(remaining)),
                               replace=False)
        sample.extend(remaining[i] for i in extra_idx)

    rng.shuffle(sample)
    return sample[:n_total]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_sessions(
    sessions: list[LabeledSession],
    encoder: SessionEncoder,
) -> np.ndarray:
    """
    Encode all sessions in batches. Returns array of shape (n, 384).
    Prints a progress indicator.
    """
    print(f"  Encoding {len(sessions)} sessions with {encoder._model_name}...")
    t0 = time.perf_counter()

    # Flatten all turn texts with their weights so we can batch the entire
    # corpus in one encoder call, then split and pool per-session.
    # This is significantly faster than encoding session by session.

    all_texts: list[str] = []
    session_slices: list[tuple[int, int]] = []   # (start, end) index into all_texts
    session_weights: list[list[float]] = []

    from kokoro.encoder import decode_parlai_artifacts

    for sess in sessions:
        texts, weights = [], []
        for turn in sess.turns:
            cleaned = decode_parlai_artifacts(turn.get("content", ""))
            if cleaned:
                texts.append(cleaned)
                weights.append(2.0 if turn.get("role") == "user" else 1.0)
        if not texts:
            texts = [""]
            weights = [1.0]
        start = len(all_texts)
        all_texts.extend(texts)
        session_slices.append((start, start + len(texts)))
        session_weights.append(weights)

    # Single batch encode all turn texts
    all_embeddings = encoder.model.encode(
        all_texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    # Weighted pool per session
    result = np.zeros((len(sessions), 384), dtype=np.float32)
    for i, (start, end) in enumerate(session_slices):
        embs = all_embeddings[start:end]        # (n_turns, 384)
        w = np.array(session_weights[i], dtype=np.float32)
        w /= w.sum()
        result[i] = (embs * w[:, np.newaxis]).sum(axis=0)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed / len(sessions) * 1000:.1f} ms/session)")
    return result


# ---------------------------------------------------------------------------
# Figure 4: t-SNE of session embeddings
# ---------------------------------------------------------------------------

def fig4_tsne_embeddings(
    sessions: list[LabeledSession],
    embeddings: np.ndarray,
) -> plt.Figure:
    print("  Running t-SNE (perplexity=30, n_iter=1000)...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Standardise before t-SNE (zero mean, unit variance per feature)
    scaled = StandardScaler().fit_transform(embeddings)

    t0 = time.perf_counter()
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    )
    coords = tsne.fit_transform(scaled)  # (n, 2)
    print(f"  t-SNE done in {time.perf_counter() - t0:.1f}s")

    emotion_colors = build_emotion_colors()
    quadrant_labels = {
        "q1": "Excited (+ val, + ar)",
        "q2": "Content (+ val, − ar)",
        "q3": "Distressed (− val, + ar)",
        "q4": "Depressed (− val, − ar)",
    }

    fig, ax = plt.subplots(figsize=(9, 8))

    # Plot per quadrant so legend is clean (4 entries, not 32)
    for quad, emotions, marker in zip(
        ["q1", "q2", "q3", "q4"],
        [_Q1_EMOTIONS, _Q2_EMOTIONS, _Q3_EMOTIONS, _Q4_EMOTIONS],
        ["o", "s", "^", "D"],
    ):
        mask = np.array([s.emotion_label in emotions for s in sessions])
        if not mask.any():
            continue
        # Color each point by its exact emotion label
        point_colors = [emotion_colors[s.emotion_label]
                        for s, m in zip(sessions, mask) if m]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=point_colors, s=22, alpha=0.70,
            linewidths=0, marker=marker,
            label=quadrant_labels[quad],
            zorder=3,
            rasterized=True,
        )

    # Annotate quadrant centroids with quadrant labels
    for quad, emotions in zip(
        ["q1", "q2", "q3", "q4"],
        [_Q1_EMOTIONS, _Q2_EMOTIONS, _Q3_EMOTIONS, _Q4_EMOTIONS],
    ):
        mask = np.array([s.emotion_label in emotions for s in sessions])
        if not mask.any():
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.text(
            cx, cy, quad.upper(),
            fontsize=10, fontweight="bold",
            color=_QUADRANT_BASE_COLORS[quad],
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5,
        )

    ax.set_xlabel("t-SNE dimension 1", labelpad=8)
    ax.set_ylabel("t-SNE dimension 2", labelpad=8)
    ax.set_title(
        f"t-SNE of {len(sessions)} Session Embeddings (all-MiniLM-L6-v2)\n"
        f"Colored by circumplex quadrant — emotionally similar sessions should cluster",
        pad=12,
    )
    ax.legend(title="Circumplex quadrant", title_fontsize=9,
              loc="upper right", markerscale=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Ground-truth valence vs PC1 of embeddings
# ---------------------------------------------------------------------------

def _pearson_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (r, p_value) using the t-distribution approximation."""
    from scipy.stats import pearsonr as _scipy_pr
    return _scipy_pr(x, y)


def fig5_valence_correlation(
    sessions: list[LabeledSession],
    embeddings: np.ndarray,
) -> plt.Figure:
    print("  Running PCA for valence correlation...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=5, random_state=42)
    components = pca.fit_transform(scaled)  # (n, 5)

    valences = np.array([s.valence for s in sessions])
    arousals = np.array([s.arousal for s in sessions])

    # Find which PC correlates best with valence (may not be PC1)
    pc_correlations = []
    for i in range(components.shape[1]):
        r, p = _pearson_r(components[:, i], valences)
        pc_correlations.append((i, r, p))
    pc_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    best_pc_idx, best_r, best_p = pc_correlations[0]

    # Always use PC1 for display (as specified), but note if it's not best
    pc1 = components[:, 0]
    r_pc1, p_pc1 = _pearson_r(pc1, valences)

    # Flip PC1 sign if negatively correlated with valence (for readability)
    if r_pc1 < 0:
        pc1 = -pc1
        r_pc1 = -r_pc1

    emotion_colors = build_emotion_colors()
    point_colors = [emotion_colors[s.emotion_label] for s in sessions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Left: valence vs PC1 ---
    ax = axes[0]
    ax.scatter(valences, pc1, c=point_colors, s=25, alpha=0.65,
               linewidths=0, zorder=3, rasterized=True)

    # Regression line
    m, b = np.polyfit(valences, pc1, 1)
    v_line = np.linspace(valences.min(), valences.max(), 100)
    ax.plot(v_line, m * v_line + b, color="#333333", linewidth=1.5,
            linestyle="--", zorder=4, alpha=0.8, label="Linear fit")

    # Annotate r
    p_str = f"p < 0.001" if p_pc1 < 0.001 else f"p = {p_pc1:.3f}"
    ax.text(0.05, 0.95,
            f"Pearson r = {r_pc1:.3f}\n{p_str}",
            transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
            zorder=5)

    ax.set_xlabel("Ground-truth valence (circumplex mapping)", labelpad=8)
    ax.set_ylabel("PC1 of session embeddings", labelpad=8)
    ax.set_title("Valence vs. First Principal Component\nof all-MiniLM-L6-v2 Embeddings", pad=10)
    ax.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=1)
    ax.axvline(0, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=1)

    # --- Right: PCA variance explained ---
    ax2 = axes[1]
    var_explained = pca.explained_variance_ratio_ * 100
    pc_labels = [f"PC{i+1}" for i in range(len(var_explained))]
    bar_colors = ["#2C3E7A" if i == 0 else "#99aacc" for i in range(len(var_explained))]
    bars = ax2.bar(pc_labels, var_explained, color=bar_colors,
                   edgecolor="white", linewidth=0.5)

    # Annotate correlation with valence on each bar
    for i, (bar, (pc_i, r_i, p_i)) in enumerate(
        zip(bars, sorted(pc_correlations, key=lambda x: x[0]))
    ):
        r_abs = abs(r_i)
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"r={r_i:+.2f}",
            ha="center", va="bottom", fontsize=8.5,
            color="#333333",
        )

    ax2.set_xlabel("Principal component", labelpad=8)
    ax2.set_ylabel("Explained variance (%)", labelpad=8)
    ax2.set_title("PCA Variance Explained\n(r = Pearson correlation with ground-truth valence)", pad=10)

    # Highlight PC most correlated with valence
    if best_pc_idx < len(bars):
        bars[best_pc_idx].set_color(_QUADRANT_BASE_COLORS["q1"])
        ax2.text(
            bars[best_pc_idx].get_x() + bars[best_pc_idx].get_width() / 2,
            -1.8,
            "best\nvalence\npredictor",
            ha="center", va="top", fontsize=7.5,
            color=_QUADRANT_BASE_COLORS["q1"], style="italic",
        )

    # Annotate variance explained by top PC for valence
    var_top = pca.explained_variance_ratio_[best_pc_idx] * 100
    ax2.set_ylim(bottom=-2.5)

    fig.suptitle(
        "Session Embedding Structure: Does the Encoder Capture Emotional Valence?",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(w_pad=3.5)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    apply_style()

    # scipy is needed for pearsonr
    try:
        import scipy.stats
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])

    print("Loading session pool...")
    sessions_full = load_empathetic_dialogues("train")
    print(f"  {len(sessions_full)} sessions")

    print("Selecting balanced sample of 500 sessions...")
    sessions = load_balanced_sample(sessions_full, n_total=500)
    label_counts = {}
    for s in sessions:
        label_counts[s.emotion_label] = label_counts.get(s.emotion_label, 0) + 1
    print(f"  {len(sessions)} sessions, {len(label_counts)} unique emotion labels")
    print(f"  Range per label: {min(label_counts.values())}–{max(label_counts.values())}")

    print("\nEncoding sessions...")
    encoder = SessionEncoder()
    embeddings = encode_sessions(sessions, encoder)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("\nFigure 4: t-SNE")
    fig4 = fig4_tsne_embeddings(sessions, embeddings)
    save(fig4, "fig4_tsne_embeddings")
    plt.close(fig4)

    print("Figure 5: Valence correlation")
    fig5 = fig5_valence_correlation(sessions, embeddings)
    save(fig5, "fig5_valence_correlation")
    plt.close(fig5)

    print("\nDone.")
