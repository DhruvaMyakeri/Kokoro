"""
Step 3 visualizations: transition model behaviour.

Execution order is strict:
  1. Load encoder (sentence transformer)
  2. Load transition model from checkpoint
  3. Encode all sessions needed
  4. Run all state trajectory computations
  5. Run all PCA reductions
  6. THEN import matplotlib and draw figures

This order is required because sentence-transformers and matplotlib
conflict when loaded in the same process on CUDA-enabled systems —
the sentence transformer's memory-mapped weights corrupt matplotlib's
Agg renderer if matplotlib is initialized first.

Run from project root:
    python figures/visualize_step3.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR  = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARC_COLORS: dict[str, str] = {
    "gradual_decline":            "#C0392B",
    "slow_recovery":              "#27AE60",
    "acute_stress_stabilization": "#E67E22",
    "chronic_low_grade_anxiety":  "#8E44AD",
    "weekend_oscillation":        "#2980B9",
    "grief_arc":                  "#2C3E50",
    "post_traumatic_growth":      "#16A085",
    "social_confidence_growth":   "#F39C12",
    "relapse_dip":                "#D35400",
    "stable_positive":            "#1ABC9C",
    "stable_negative":            "#7F8C8D",
}


# ===========================================================================
# PHASE 1 — Load ML models
# ===========================================================================

logger.info("Phase 1: loading ML models")

import torch

device = torch.device("cpu")   # CPU avoids CUDA memory fragmentation in viz

from kokoro.encoder import SessionEncoder
from kokoro.encoder import decode_parlai_artifacts
from kokoro.transition import TransitionModel

encoder = SessionEncoder(device="cpu")
_ = encoder.model   # trigger sentence transformer download/load NOW, before anything else
logger.info("  Encoder loaded")

# Current model: split-LN + VAD + VICReg (PR=339.7, best epoch 24)
ckpt_current_path = PROJECT_ROOT / "checkpoints" / "transition_v1.pt"
hist_current_path = ckpt_current_path.with_suffix(".history.json")
# Old model: joint-LN, no VAD, no VICReg (PR=1.4, best epoch 4) — kept for fig6 comparison
ckpt_old_path     = PROJECT_ROOT / "checkpoints" / "transition_v1_10k.pt"
hist_old_path     = ckpt_old_path.with_suffix(".history.json")
traj_10k_path     = PROJECT_ROOT / "data" / "trajectories_10k.json"

for p in [ckpt_current_path, hist_current_path, traj_10k_path]:
    if not p.exists():
        sys.exit(f"Required file missing: {p}\nRun training steps first.")

# Use current (fixed) model for fig7 and fig8
ckpt_current = torch.load(ckpt_current_path, map_location="cpu", weights_only=False)
cfg = ckpt_current.get("model_config", {"state_dim": 384, "hidden_dim": 512})
model = TransitionModel(**{k: v for k, v in cfg.items() if k in ("state_dim", "session_dim", "hidden_dim")})
model.load_state_dict(ckpt_current["model_state_dict"])
model.eval()
use_vad = cfg.get("session_dim", 384) > 384
logger.info(f"  Current model loaded (epoch {ckpt_current['epoch']}, val_loss={ckpt_current['val_loss']:.4f}, use_vad={use_vad})")
# Override encoder to use VAD if needed
if use_vad:
    encoder = SessionEncoder(device="cpu", use_vad_features=True)
    _ = encoder.model


# ===========================================================================
# PHASE 2 — Load data
# ===========================================================================

logger.info("Phase 2: loading data")

with open(hist_current_path) as f:
    history_current = json.load(f)
# Load old history if available (for architecture comparison in fig6)
history_old = None
if hist_old_path.exists():
    with open(hist_old_path) as f:
        history_old = json.load(f)
logger.info(
    f"  Current history: {len(history_current['train_loss'])} epochs, "
    f"best epoch {history_current.get('best_epoch')}, PR={history_current.get('participation_ratio', '?'):.1f}"
)

logger.info("  Loading 10k trajectories...")
with open(traj_10k_path) as f:
    trajectories = json.load(f)
logger.info(f"  {len(trajectories)} trajectories loaded")


# ===========================================================================
# PHASE 3 — Encode sessions
# ===========================================================================

logger.info("Phase 3: encoding sessions")

def encode_unique_sessions(
    trajs: list[dict],
    chunk_size: int = 1500,
) -> dict[str, np.ndarray]:
    """
    Encode all unique sessions referenced across trajectories.
    Returns conv_id -> (384,) float32 numpy array.
    Processes in chunks to cap per-chunk text list size.
    """
    unique: dict[str, list] = {}
    for traj in trajs:
        for sess in traj["sessions"]:
            if sess["conv_id"] not in unique:
                unique[sess["conv_id"]] = sess["turns"]

    all_ids = list(unique.keys())
    n = len(all_ids)
    logger.info(f"  Encoding {n} unique sessions in chunks of {chunk_size}...")
    t0 = time.perf_counter()

    cache: dict[str, np.ndarray] = {}
    for start in range(0, n, chunk_size):
        chunk_ids = all_ids[start: start + chunk_size]
        texts_all, slices, weights_all = [], [], []

        for conv_id in chunk_ids:
            turns = unique[conv_id]
            texts, weights = [], []
            for turn in turns:
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
            if use_vad:
                from kokoro.vad import VADLexicon
                _vad = encoder._get_vad()
                vad_scores = _vad.score_turns(unique[conv_id])
                pooled = np.concatenate([pooled, vad_scores])
            cache[conv_id] = pooled

        logger.info(f"    {min(start + chunk_size, n)}/{n} sessions encoded")

    logger.info(f"  Done in {time.perf_counter() - t0:.1f}s")
    return cache


# Use val split only (same as training evaluation) — speeds up embedding step
import random
random.seed(42)
_trajs_shuffled = list(trajectories)
random.shuffle(_trajs_shuffled)
n_val = int(len(_trajs_shuffled) * 0.2)
val_trajs = _trajs_shuffled[len(_trajs_shuffled) - n_val:]
logger.info(f"  Using val split: {len(val_trajs)} trajectories for fig7/fig8")
emb_cache = encode_unique_sessions(val_trajs)


# ===========================================================================
# PHASE 4 — Compute state trajectories
# ===========================================================================

logger.info("Phase 4: computing state trajectories")

TARGET_ARCS = ["gradual_decline", "slow_recovery", "grief_arc"]

def run_trajectory(traj: dict) -> np.ndarray:
    """
    Run one trajectory through the transition model.
    Returns array of shape (n_sessions + 1, 384):
      row 0        = initial zero state
      row t+1      = state after processing session t
    """
    state = TransitionModel.initial_state()   # (384,) zeros
    rows = [state.numpy().copy()]
    with torch.no_grad():
        for sess in traj["sessions"]:
            emb = torch.tensor(emb_cache[sess["conv_id"]])
            state = model(state, emb)
            rows.append(state.numpy().copy())
    return np.stack(rows, axis=0)   # (n+1, 384)


# Find the longest trajectory of each target arc type (from val set only)
by_arc: dict[str, list[dict]] = defaultdict(list)
for t in val_trajs:
    if all(s["conv_id"] in emb_cache for s in t["sessions"]):
        by_arc[t["arc_name"]].append(t)

traj_examples: dict[str, dict] = {}
for arc in TARGET_ARCS:
    if arc in by_arc:
        traj_examples[arc] = max(by_arc[arc], key=lambda t: t["n_sessions"])

# Run each through the model
state_arrays: dict[str, np.ndarray] = {}
for arc, traj in traj_examples.items():
    state_arrays[arc] = run_trajectory(traj)
    logger.info(f"  {arc}: {state_arrays[arc].shape[0]} states computed")

# PCA on all non-zero states combined
from sklearn.decomposition import PCA

nonzero_states = np.concatenate([s[1:] for s in state_arrays.values()], axis=0)
pca_traj = PCA(n_components=2, random_state=42)
pca_traj.fit(nonzero_states)
logger.info(
    f"  PCA variance explained: "
    f"PC1={pca_traj.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2={pca_traj.explained_variance_ratio_[1]*100:.1f}%"
)

# Project each arc's states (including initial zero)
projected_states: dict[str, np.ndarray] = {
    arc: pca_traj.transform(s)
    for arc, s in state_arrays.items()
}


# ===========================================================================
# PHASE 5 — Compute arc separation (final states)
# ===========================================================================

logger.info("Phase 5: computing arc separation")

SAMPLES_PER_ARC = 50
rng = np.random.RandomState(42)

sampled_final_states: list[np.ndarray] = []
sampled_arc_labels: list[str] = []

for arc, arc_trajs in by_arc.items():
    n = min(SAMPLES_PER_ARC, len(arc_trajs))
    idx = rng.choice(len(arc_trajs), size=n, replace=False)
    for i in idx:
        traj = arc_trajs[i]
        state = TransitionModel.initial_state()
        with torch.no_grad():
            for sess in traj["sessions"]:
                emb = torch.tensor(emb_cache[sess["conv_id"]])
                state = model(state, emb)
        sampled_final_states.append(state.numpy().copy())
        sampled_arc_labels.append(arc)

final_states_arr = np.stack(sampled_final_states, axis=0)   # (N, 384)
logger.info(f"  {len(final_states_arr)} final states ({len(by_arc)} arc types)")

pca_sep = PCA(n_components=2, random_state=42)
final_coords = pca_sep.fit_transform(final_states_arr)
logger.info(
    f"  PCA variance explained: "
    f"PC1={pca_sep.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2={pca_sep.explained_variance_ratio_[1]*100:.1f}%"
)

arc_centroids: dict[str, np.ndarray] = {}
for arc in set(sampled_arc_labels):
    mask = np.array([l == arc for l in sampled_arc_labels])
    arc_centroids[arc] = final_coords[mask].mean(axis=0)


# ===========================================================================
# PHASE 6 — Import matplotlib and draw figures
# ===========================================================================

logger.info("Phase 6: drawing figures (importing matplotlib now)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe


def apply_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.titlepad":      10,
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.grid":          True,
        "grid.alpha":         0.2,
        "grid.linewidth":     0.5,
        "grid.color":         "#999999",
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path)
        logger.info(f"  Saved {path.name}")


apply_style()


# ---------------------------------------------------------------------------
# Figure 6: Loss curves
# ---------------------------------------------------------------------------

def draw_fig6() -> plt.Figure:
    """
    Left panel:  old architecture (joint-LN, no VICReg) training curves
    Right panel: current architecture (split-LN + VAD + VICReg) training curves

    If old history is unavailable, right panel fills the full width.
    """
    n_panels = 2 if history_old is not None else 1
    fig, axes_raw = plt.subplots(1, n_panels, figsize=(13 if n_panels == 2 else 8, 5))
    axes = [axes_raw] if n_panels == 1 else list(axes_raw)

    panels = []
    if history_old is not None:
        panels.append((
            axes[0], history_old,
            "Original Architecture\n(Joint LayerNorm, no VICReg, no VAD)",
            "#888888", False,
        ))
    panels.append((
        axes[-1], history_current,
        "Current Architecture\n(Split LayerNorm + VICReg + VAD)",
        "#2980B9", True,
    ))

    for ax, hist, title, train_color, is_current in panels:
        n_epochs  = len(hist["train_loss"])
        epochs    = list(range(1, n_epochs + 1))
        train_arr = np.array(hist["train_loss"])
        val_arr   = np.array(hist["val_loss"])

        # Detect incompatible scales (VICReg train loss >> cosine val loss)
        dual_scale = is_current and (train_arr.min() > val_arr.max() * 3)

        if dual_scale:
            # Left axis: val loss (cosine).  Right axis: train loss (VICReg total)
            ax2 = ax.twinx()

            ax2.plot(epochs, train_arr, color=train_color, linewidth=1.8,
                     label="Train (total, incl. VICReg)", zorder=3, alpha=0.75)
            ax2.set_ylabel("Total train loss  (cosine + std + cov)", color=train_color,
                           labelpad=8, fontsize=9)
            ax2.tick_params(axis="y", labelcolor=train_color)
            tr_lo = train_arr.min() - 0.5
            tr_hi = train_arr.max() + 0.5
            ax2.set_ylim(tr_lo, tr_hi)

            ax.plot(epochs, val_arr, color="#C0392B", linewidth=1.8,
                    linestyle="--", label="Val cosine loss", zorder=4)
            ax.set_ylabel("Val cosine loss  (1 − cos sim)", labelpad=8)

            # Combine legends
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        else:
            ax.plot(epochs, train_arr, color=train_color, linewidth=1.8,
                    label="Train", zorder=3)
            ax.plot(epochs, val_arr, color="#C0392B", linewidth=1.8,
                    linestyle="--", label="Validation", zorder=3)
            ax.set_ylabel("Val cosine prediction loss  (1 − cos sim)", labelpad=8)
            ax.legend(loc="upper right")

        best_ep  = int(np.argmin(val_arr)) + 1
        best_val = val_arr[best_ep - 1]
        ax.axvline(best_ep, color="#C0392B", linewidth=0.9, linestyle=":",
                   alpha=0.6, zorder=2)
        ax.scatter([best_ep], [best_val], color="#C0392B", s=55, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(
            best_ep + max(n_epochs * 0.02, 0.8), best_val,
            f"best val {best_val:.3f}\n(epoch {best_ep})",
            fontsize=8, color="#C0392B", va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

        # Shade overfit region if on same scale (val diverges > 0.05 from train)
        if not dual_scale:
            gap = val_arr - train_arr
            overfit_start = next((i for i in range(len(gap)) if gap[i] > 0.05), None)
            if overfit_start is not None:
                ax.axvspan(overfit_start + 1, n_epochs, color="#C0392B", alpha=0.06,
                           zorder=1, label=f"Overfit region (ep {overfit_start+1}+)")

        # Annotate PR on current model
        if is_current:
            pr = hist.get("participation_ratio")
            if pr:
                ax.text(0.97, 0.06, f"Final PR = {pr:.0f} / 384",
                        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                        color="#27AE60", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.9))

        val_lo = max(0.0, val_arr.min() - 0.04)
        val_hi = val_arr.max() + 0.06
        ax.set_xlim(1, n_epochs)
        ax.set_ylim(val_lo, val_hi)
        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_title(title, pad=10)

    fig.suptitle(
        "Transition Model Training: Architecture Comparison",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(w_pad=3.5)
    return fig


# ---------------------------------------------------------------------------
# Figure 7: State trajectory paths through PCA space
# ---------------------------------------------------------------------------

TRAJ_COLORS = {
    "gradual_decline": "#C0392B",
    "slow_recovery":   "#27AE60",
    "grief_arc":       "#2C3E50",
}
TRAJ_CMAPS = {
    "gradual_decline": "Reds",
    "slow_recovery":   "Greens",
    "grief_arc":       "Blues",
}

def draw_fig7() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 8))

    for arc in TARGET_ARCS:
        if arc not in projected_states:
            continue
        proj  = projected_states[arc]   # (n+1, 2)
        color = TRAJ_COLORS[arc]
        cmap  = plt.get_cmap(TRAJ_CMAPS[arc])
        n     = len(proj)
        step_colors = [cmap(0.35 + 0.65 * i / max(n - 1, 1)) for i in range(n)]

        # Gradient arrows between consecutive states
        for i in range(n - 1):
            ax.annotate(
                "", xy=(proj[i+1, 0], proj[i+1, 1]),
                xytext=(proj[i, 0], proj[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=step_colors[i],
                    lw=1.7,
                    mutation_scale=13,
                    shrinkA=8, shrinkB=8,
                ),
                zorder=4,
            )

        # Session markers
        for i, (x, y) in enumerate(proj):
            face = "white" if i == 0 else step_colors[i]
            ax.scatter(x, y, s=110 if i == 0 else 75,
                       color=face, edgecolors=color,
                       linewidths=1.5, zorder=5)
            label = "init" if i == 0 else f"S{i}"
            ax.text(x, y, label, fontsize=6.5, ha="center", va="center",
                    color=color if i == 0 else "white",
                    fontweight="bold", zorder=6)

        # Legend proxy
        ax.plot([], [], color=color, linewidth=2, marker="o", markersize=7,
                label=arc.replace("_", " ").title())

    v1, v2 = pca_traj.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1  ({v1:.1f}% variance explained)", labelpad=8)
    ax.set_ylabel(f"PC2  ({v2:.1f}% variance explained)", labelpad=8)
    ax.set_title(
        "User State Vectors Through Time: Three Arc Types in PCA Space\n"
        "Each point = model state after one session  |  Arrows show temporal direction",
        pad=12,
    )
    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.legend(title="Arc type", fontsize=9, title_fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 8: Arc separation in final-state PCA space
# ---------------------------------------------------------------------------

def draw_fig8() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))

    present_arcs = sorted(set(sampled_arc_labels))
    for arc in present_arcs:
        mask   = np.array([l == arc for l in sampled_arc_labels])
        color  = ARC_COLORS.get(arc, "#888888")
        coords = final_coords[mask]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=color, s=30, alpha=0.65, linewidths=0,
            label=arc.replace("_", " ").title(),
            zorder=3,
        )

    # Centroid diamonds + arc name labels
    for arc, centroid in arc_centroids.items():
        if arc not in present_arcs:
            continue
        color = ARC_COLORS.get(arc, "#888888")
        ax.scatter(*centroid, s=160, color=color, edgecolors="white",
                   linewidths=1.5, zorder=6, marker="D")
        ax.text(
            centroid[0], centroid[1] + 0.01,
            arc.replace("_", " "),
            fontsize=7.5, ha="center", va="bottom",
            color=color, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=7,
        )

    v1, v2 = pca_sep.explained_variance_ratio_ * 100
    pr_val = history_current.get("participation_ratio", None)
    pr_str = f"  |  PR = {pr_val:.0f}" if pr_val else ""
    ax.set_xlabel(f"PC1  ({v1:.1f}% var. explained)", labelpad=8)
    ax.set_ylabel(f"PC2  ({v2:.1f}% var. explained)", labelpad=8)
    ax.set_title(
        f"Final State Vectors by Arc Type  ({SAMPLES_PER_ARC} samples each){pr_str}\n"
        "Current model (PR=339.7): arc info distributed across 340 dims — low PC1/PC2 variance expected",
        pad=12,
    )
    ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=1)
    ax.legend(title="Arc type", fontsize=8.5, title_fontsize=9,
              markerscale=1.6, framealpha=0.92, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Generate and save
# ---------------------------------------------------------------------------

print("\nFigure 6: loss curves")
fig6 = draw_fig6()
save_figure(fig6, "fig6_loss_curves")
plt.close(fig6)

print("Figure 7: state trajectories")
fig7 = draw_fig7()
save_figure(fig7, "fig7_state_trajectories")
plt.close(fig7)

print("Figure 8: arc separation")
fig8 = draw_fig8()
save_figure(fig8, "fig8_arc_separation")
plt.close(fig8)

print("\nDone.")
