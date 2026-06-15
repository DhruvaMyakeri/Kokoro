"""
Step 4 visualizations: new evaluation figures for current model (PR=339.7).

Produces five publication-quality figures:
  fig9_world_model_accuracy.{png,pdf}   — model vs baseline cosine sim distribution
  fig10_probe_generalization.{png,pdf}  — probe on naturalistic text, circumplex scatter
  fig11_probe_comparison.{png,pdf}      — valence/arousal r across 3 training runs
  fig12_longitudinal_curves.{png,pdf}   — MAE learning curve + per-arc tracking
  fig13_pr_progression.{png,pdf}        — PR 1.4 → 245 → 339.7 across architecture changes

Run from project root:
    python figures/visualize_step4.py
"""

from __future__ import annotations

import json
import logging
import random
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


# ===========================================================================
# PHASE 1 — Load ML models (BEFORE matplotlib to avoid renderer conflicts)
# ===========================================================================

logger.info("Phase 1: loading ML models")

import torch
import torch.nn.functional as F

from kokoro.encoder import SessionEncoder
from kokoro.transition import TransitionModel
from training.train_probe import ValenceArousalProbe

device = torch.device("cpu")

# Encoder
encoder = SessionEncoder(device="cpu", use_vad_features=True)
_ = encoder.model
logger.info("  Encoder loaded (use_vad=True)")

# Transition model (current: split-LN + VAD + VICReg, PR=339.7)
ckpt_path  = PROJECT_ROOT / "checkpoints" / "transition_v1.pt"
probe_path = PROJECT_ROOT / "checkpoints" / "valence_arousal_probe.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg  = ckpt.get("model_config", {})
model = TransitionModel(**{k: v for k, v in cfg.items() if k in ("state_dim", "session_dim", "hidden_dim")})
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
logger.info(f"  Transition model: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# Linear probe
probe_ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
probe_cfg  = probe_ckpt.get("probe_config", {})
probe = ValenceArousalProbe(state_dim=probe_cfg.get("state_dim", 384))
probe.load_state_dict(probe_ckpt["model_state_dict"])
probe.eval()
r_v = probe_ckpt.get("val_r_valence", 0)
r_a = probe_ckpt.get("val_r_arousal", 0)
logger.info(f"  Probe: val_r_valence={r_v:.4f}, val_r_arousal={r_a:.4f}")


# ===========================================================================
# PHASE 2 — Load data and precompute embeddings
# ===========================================================================

logger.info("Phase 2: loading data")

traj_path = PROJECT_ROOT / "data" / "trajectories_10k.json"
with open(traj_path) as f:
    all_trajs = json.load(f)

random.seed(42)
random.shuffle(all_trajs)
n_val     = int(len(all_trajs) * 0.2)
val_trajs = all_trajs[len(all_trajs) - n_val:]
logger.info(f"  Val set: {len(val_trajs):,} trajectories")

# Sample for speed (fig9 and fig12 need many trajectories; use all val)
from training.train import precompute_embeddings
logger.info("  Precomputing embeddings...")
emb_cache = precompute_embeddings(val_trajs, encoder, device)
logger.info(f"  Cache: {len(emb_cache):,} sessions")


# ===========================================================================
# PHASE 3 — Compute world model accuracy (fig9 data)
# ===========================================================================

logger.info("Phase 3: world model accuracy")

model_sims:    list[float] = []
baseline_sims: list[float] = []
arc_names:     list[str]   = []
state_dim = model.STATE_DIM

model.eval()
with torch.no_grad():
    for traj in val_trajs:
        sessions = traj["sessions"]
        arc      = traj.get("arc_name", "unknown")
        if len(sessions) < 2:
            continue
        embs  = [emb_cache[s["conv_id"]] for s in sessions]
        state = torch.zeros(state_dim, device=device)
        for t in range(len(embs) - 1):
            state = model(state, embs[t])
            s_n = F.normalize(state,               dim=-1)
            e_n = F.normalize(embs[t+1][:state_dim], dim=-1)
            b_n = F.normalize(embs[t][:state_dim],   dim=-1)
            model_sims.append((s_n * e_n).sum().item())
            baseline_sims.append((b_n * e_n).sum().item())
            arc_names.append(arc)

model_arr    = np.array(model_sims,    dtype=np.float32)
baseline_arr = np.array(baseline_sims, dtype=np.float32)
logger.info(f"  {len(model_arr):,} pairs | model={model_arr.mean():.4f} baseline={baseline_arr.mean():.4f}")


# ===========================================================================
# PHASE 4 — Probe on naturalistic scenarios (fig10 data)
# ===========================================================================

logger.info("Phase 4: probe generalization")

# Import scenarios from probe_generalization.py
from diagnostics.probe_generalization import SCENARIOS

nat_results = []
for sc in SCENARIOS:
    emb = encoder.encode(sc["turns"])
    emb_t = torch.tensor(emb, dtype=torch.float32)
    state = torch.zeros(state_dim)
    with torch.no_grad():
        state = model(state, emb_t)
        va = probe(state.unsqueeze(0)).squeeze(0)
    nat_results.append({
        "label":    sc["label"],
        "exp_quad": sc["expected_quadrant"],
        "exp_v":    sc["expected_valence"],
        "exp_a":    sc["expected_arousal"],
        "pred_v":   va[0].item(),
        "pred_a":   va[1].item(),
    })

logger.info(f"  {len(nat_results)} naturalistic scenarios processed")
v_range = max(r["pred_v"] for r in nat_results) - min(r["pred_v"] for r in nat_results)
a_range = max(r["pred_a"] for r in nat_results) - min(r["pred_a"] for r in nat_results)
logger.info(f"  Valence spread={v_range:.3f}  Arousal spread={a_range:.3f}")


# ===========================================================================
# PHASE 5 — Longitudinal simulation (fig12 data)
# ===========================================================================

logger.info("Phase 5: longitudinal simulation")

# Simulate 200 users: track MAE by session index
N_SIM_USERS = 200
rng_sim = random.Random(99)
sim_users = rng_sim.sample(val_trajs, min(N_SIM_USERS, len(val_trajs)))

by_session_errors: dict[int, list[float]] = defaultdict(list)
all_sim_records: list[dict] = []

model.eval()
probe.eval()
with torch.no_grad():
    for traj in sim_users:
        sessions = traj["sessions"]
        arc      = traj.get("arc_name", "unknown")
        state    = torch.zeros(state_dim)
        for t, sess in enumerate(sessions):
            cid = sess["conv_id"]
            if cid not in emb_cache:
                continue
            emb   = emb_cache[cid]
            state = model(state, emb)
            va    = probe(state.unsqueeze(0)).squeeze(0)
            gt_v  = sess.get("valence", 0.0)
            gt_a  = sess.get("arousal", 0.0)
            err_v = abs(va[0].item() - gt_v)
            err_a = abs(va[1].item() - gt_a)
            if t < 15:
                by_session_errors[t].append(err_v)
            all_sim_records.append({
                "session_idx": t, "arc": arc, "warm": t >= 2,
                "err_v": err_v, "err_a": err_a,
                "pred_v": va[0].item(), "gt_v": gt_v,
            })

per_session_mae = {k: np.mean(v) for k, v in sorted(by_session_errors.items())}
per_arc_mae: dict[str, dict] = {}
for r in all_sim_records:
    if r["warm"]:
        arc = r["arc"]
        if arc not in per_arc_mae:
            per_arc_mae[arc] = {"v": [], "a": []}
        per_arc_mae[arc]["v"].append(r["err_v"])
        per_arc_mae[arc]["a"].append(r["err_a"])

logger.info(f"  {len(all_sim_records):,} session records, {len(per_session_mae)} session indices")


# ===========================================================================
# PHASE 6 — Import matplotlib and draw figures
# ===========================================================================

logger.info("Phase 6: drawing figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

from figures.visualize_step1 import (
    apply_style,
    _QUADRANT_BASE_COLORS,
    draw_circumplex_background,
    save,
)

apply_style()

# ─── Color palette ───────────────────────────────────────────────────────────
C_MODEL    = "#2980B9"
C_BASELINE = "#E67E22"
C_GOOD     = "#27AE60"
C_BAD      = "#C0392B"

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
    "anxiety_to_depression":      "#9B59B6",
    "depression_to_anxiety":      "#3498DB",
    "excitement_to_contentment":  "#E91E8C",
}


# ---------------------------------------------------------------------------
# Figure 9: World model accuracy — model vs baseline cosine sim distribution
# ---------------------------------------------------------------------------

def draw_fig9() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: overlapping histograms
    ax = axes[0]
    bins = np.linspace(-0.3, 1.0, 60)
    ax.hist(baseline_arr, bins=bins, color=C_BASELINE, alpha=0.55, density=True,
            label=f"Naive baseline  (mean={baseline_arr.mean():.3f})", zorder=2)
    ax.hist(model_arr,    bins=bins, color=C_MODEL,    alpha=0.65, density=True,
            label=f"World model     (mean={model_arr.mean():.3f})", zorder=3)
    ax.axvline(baseline_arr.mean(), color=C_BASELINE, linewidth=1.5, linestyle="--", zorder=4)
    ax.axvline(model_arr.mean(),    color=C_MODEL,    linewidth=1.5, linestyle="--", zorder=4)
    ax.set_xlabel("Cosine similarity  (state_t · e_{t+1})", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title(
        f"World Model: Next-Session Prediction Quality\n"
        f"Model vs naive baseline  ({len(model_arr):,} prediction pairs, val set)",
        pad=10,
    )
    ax.legend(fontsize=9)
    delta = model_arr.mean() - baseline_arr.mean()
    beats = (model_arr > baseline_arr).mean() * 100
    ax.text(0.97, 0.97,
            f"Delta = {delta:+.3f}\n{beats:.1f}% beats baseline",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # Right: per-arc mean model sim (sorted)
    ax2 = axes[1]
    unique_arcs = sorted(set(arc_names))
    arc_means, arc_base_means, arc_labels = [], [], []
    for arc in unique_arcs:
        idx = [i for i, a in enumerate(arc_names) if a == arc]
        arc_means.append(float(model_arr[idx].mean()))
        arc_base_means.append(float(baseline_arr[idx].mean()))
        arc_labels.append(arc.replace("_", " "))

    order = np.argsort(arc_means)[::-1]
    y_pos = np.arange(len(order))
    bar_colors = [ARC_COLORS.get(unique_arcs[i], "#888888") for i in order]

    ax2.barh(y_pos, [arc_means[i] for i in order],
             color=bar_colors, alpha=0.85, height=0.55,
             label="World model", zorder=3)
    ax2.scatter([arc_base_means[i] for i in order], y_pos,
                color="#E67E22", marker="|", s=80, linewidths=2,
                label="Naive baseline", zorder=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([arc_labels[i] for i in order], fontsize=8.5)
    ax2.set_xlabel("Mean cosine similarity", labelpad=8)
    ax2.set_title("Per-Arc World Model Accuracy", pad=10)
    ax2.axvline(model_arr.mean(), color=C_MODEL, linewidth=0.8,
                linestyle=":", alpha=0.6, label="Overall mean")
    ax2.legend(fontsize=8.5)
    ax2.invert_yaxis()
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", left=False)

    fig.suptitle("Kokoro World Model: Validated Next-Session Prediction",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=3.5)
    return fig


# ---------------------------------------------------------------------------
# Figure 10: Probe generalization — naturalistic scenarios in circumplex space
# ---------------------------------------------------------------------------

# Map expected quadrant label → rough ground-truth for display
_EXP_V_MAP = {
    "positive": +0.65, "slightly positive": +0.20,
    "neutral": 0.0, "moderate": 0.0,
    "negative": -0.65, "slightly negative": -0.20,
}
_EXP_A_MAP = {
    "high": +0.65, "moderate": +0.25,
    "low": -0.40,
}

def _quad_color(exp_quad: str) -> str:
    if exp_quad.startswith("Q1"):
        return _QUADRANT_BASE_COLORS["q1"]
    if exp_quad.startswith("Q2"):
        return _QUADRANT_BASE_COLORS["q3"]   # q2 = negative-high = distressed
    if exp_quad.startswith("Q3"):
        return _QUADRANT_BASE_COLORS["q4"]   # q3 = negative-low = depressed
    if exp_quad.startswith("Q4"):
        return _QUADRANT_BASE_COLORS["q2"]   # q4 = positive-low = content
    return "#888888"


def draw_fig10() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left: circumplex scatter of predictions
    ax = axes[0]
    draw_circumplex_background(ax, alpha=0.08)

    for r in nat_results:
        color = _quad_color(r["exp_quad"])
        ax.scatter(r["pred_v"], r["pred_a"],
                   c=color, s=80, alpha=0.85, edgecolors="white",
                   linewidths=0.8, zorder=4)
        short_label = r["label"].split("-", 1)[-1].replace("_", " ")[:14]
        ax.text(r["pred_v"] + 0.03, r["pred_a"] + 0.03,
                short_label, fontsize=6.5, color=color, zorder=5,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax.set_xlim(-1.0, 0.85)
    ax.set_ylim(-0.35, 0.65)
    ax.set_xlabel("Predicted valence", labelpad=8)
    ax.set_ylabel("Predicted arousal", labelpad=8)
    ax.set_title(
        f"Probe Predictions on 22 Naturalistic Scenarios\n"
        f"(Valence spread = {v_range:.3f}  |  vs. 0.034 original model)",
        pad=10,
    )

    # Quadrant legend
    legend_handles = [
        mpatches.Patch(color=_QUADRANT_BASE_COLORS["q1"], label="Q1: Excited (+v, +a)"),
        mpatches.Patch(color=_QUADRANT_BASE_COLORS["q3"], label="Q2: Distressed (−v, +a)"),
        mpatches.Patch(color=_QUADRANT_BASE_COLORS["q4"], label="Q3: Depressed (−v, −a)"),
        mpatches.Patch(color=_QUADRANT_BASE_COLORS["q2"], label="Q4: Content (+v, −a)"),
        mpatches.Patch(color="#888888",                   label="Ambiguous/transition"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", title="Expected quadrant")

    # Right: spread comparison bar — original vs current
    ax2 = axes[1]
    models = ["Original\n(PR = 1.4)", "Current\n(PR = 339.7)"]
    v_spreads = [0.034, v_range]
    a_spreads = [0.0,   a_range]
    x = np.arange(len(models))
    w = 0.35
    bars_v = ax2.bar(x - w/2, v_spreads, width=w, color=C_MODEL, alpha=0.85,
                     label="Valence spread", zorder=3)
    bars_a = ax2.bar(x + w/2, a_spreads, width=w, color=C_GOOD, alpha=0.85,
                     label="Arousal spread", zorder=3)
    for bar, val in zip(list(bars_v) + list(bars_a), v_spreads + a_spreads):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylabel("Prediction range on naturalistic text", labelpad=8)
    ax2.set_title(
        "Probe Generalization: Original vs Current Model\n"
        "(34× improvement in valence spread after collapse fix)",
        pad=10,
    )
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(v_spreads + a_spreads) * 1.3)

    fig.suptitle("Probe Generalization to Naturalistic Companion AI Conversations",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=3.5)
    return fig


# ---------------------------------------------------------------------------
# Figure 11: Probe r comparison across 3 training runs
# ---------------------------------------------------------------------------

def draw_fig11() -> plt.Figure:
    runs = [
        "Run 1\nJoint LN\nno VAD\n(PR = 1.4)",
        "Run 2\nSplit LN\nno VAD\n(PR = 245)",
        "Run 3\nSplit LN\n+ VAD\n(PR = 339.7)",
    ]
    val_r_v = [0.226, 0.693, 0.698]
    val_r_a = [0.191, 0.544, 0.542]
    pr_vals  = [1.4, 245, 339.7]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(runs))
    w = 0.38

    # Left: probe r bars
    ax = axes[0]
    bars_v = ax.bar(x - w/2, val_r_v, width=w, color=C_MODEL, alpha=0.85,
                    label="Valence Pearson r", zorder=3)
    bars_a = ax.bar(x + w/2, val_r_a, width=w, color=C_GOOD,  alpha=0.85,
                    label="Arousal Pearson r", zorder=3)
    for bar, val in zip(list(bars_v) + list(bars_a), val_r_v + val_r_a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=9, linespacing=1.3)
    ax.set_ylabel("Pearson r (val set)", labelpad=8)
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "Linear Probe Accuracy Across Architecture Runs\n"
        "(Same probe architecture — only transition model changed)",
        pad=10,
    )
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)

    # Annotate the 3x improvement arrow (run1 → run2)
    ax.annotate("", xy=(1 - w/2, val_r_v[1]), xytext=(0 - w/2, val_r_v[0]),
                 arrowprops=dict(arrowstyle="-|>", color=C_MODEL, lw=1.5,
                                 connectionstyle="arc3,rad=-0.2"))
    ax.text(0.5, (val_r_v[0] + val_r_v[1])/2 + 0.08,
            "+3×", fontsize=11, color=C_MODEL, ha="center", fontweight="bold")

    # Right: PR progression
    ax2 = axes[1]
    run_labels_short = ["Run 1\n(Joint LN)", "Run 2\n(Split LN)", "Run 3\n(Split LN\n+ VAD)"]
    pr_colors = [C_BAD, C_BASELINE, C_GOOD]
    bars_pr = ax2.bar(x, pr_vals, color=pr_colors, alpha=0.85, zorder=3)
    for bar, val in zip(bars_pr, pr_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_labels_short, fontsize=9, linespacing=1.3)
    ax2.set_ylabel("Participation Ratio (effective dimensions)", labelpad=8)
    ax2.set_title(
        "Effective Dimensionality of State Space\n"
        "PR = (Σλ)² / Σλ²  (max = 384)",
        pad=10,
    )
    ax2.axhline(384, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)
    ax2.text(2.4, 375, "max (384)", fontsize=8, color="#888888", ha="right")
    ax2.set_ylim(0, 430)

    fig.suptitle("Architecture Evolution: From Dimensional Collapse to 340 Effective Dimensions",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=3.5)
    return fig


# ---------------------------------------------------------------------------
# Figure 12: Longitudinal MAE learning curve + per-arc bars
# ---------------------------------------------------------------------------

def draw_fig12() -> plt.Figure:
    fig = plt.figure(figsize=(14, 6))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: MAE by session index (learning curve)
    ax = fig.add_subplot(gs[0])
    session_idxs = sorted(per_session_mae.keys())
    maes         = [per_session_mae[i] for i in session_idxs]

    # Color: red for cold-start (t<2), green for warm (t>=2) — matches diagnostic definition
    colors = [C_BAD if i < 2 else C_GOOD for i in session_idxs]
    ax.bar(session_idxs, maes, color=colors, alpha=0.8, zorder=3)
    ax.plot(session_idxs, maes, color="#333333", linewidth=1.5, marker="o",
            markersize=5, zorder=4)
    ax.axvline(1.5, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)
    ax.text(1.5, max(maes) * 0.98, "  warm-start\n  threshold",
            fontsize=8, color="#333333", va="top")
    cold_patch = mpatches.Patch(color=C_BAD, alpha=0.8, label="Cold-start (sessions 0–1)")
    warm_patch = mpatches.Patch(color=C_GOOD, alpha=0.8, label="Warm (sessions 2+)")
    ax.legend(handles=[cold_patch, warm_patch], fontsize=8.5)
    ax.set_xlabel("Session index (0 = first session)", labelpad=8)
    ax.set_ylabel("Valence MAE", labelpad=8)
    ax.set_title(
        f"Longitudinal Tracking: MAE by Session Index\n"
        f"({N_SIM_USERS} simulated users, val set trajectories)",
        pad=10,
    )
    cold_mae = np.mean([per_session_mae[i] for i in [0, 1] if i in per_session_mae])
    warm_mae = np.mean([per_session_mae[i] for i in session_idxs if i >= 2])
    ax.text(0.97, 0.97,
            f"Cold MAE: {cold_mae:.3f}\nWarm MAE: {warm_mae:.3f}\nImprovement: {(cold_mae-warm_mae)/cold_mae*100:.1f}%",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # Right: per-arc MAE bars (valence)
    ax2 = fig.add_subplot(gs[1])
    arc_names_sorted  = sorted(per_arc_mae.keys(), key=lambda a: np.mean(per_arc_mae[a]["v"]))
    arc_v_maes = [np.mean(per_arc_mae[a]["v"]) for a in arc_names_sorted]
    arc_a_maes = [np.mean(per_arc_mae[a]["a"]) for a in arc_names_sorted]
    y_pos = np.arange(len(arc_names_sorted))
    bar_colors = [ARC_COLORS.get(a, "#888888") for a in arc_names_sorted]
    ax2.barh(y_pos - 0.18, arc_v_maes, height=0.35, color=bar_colors, alpha=0.85,
             label="Valence MAE", zorder=3)
    ax2.barh(y_pos + 0.18, arc_a_maes, height=0.35, color=bar_colors, alpha=0.40,
             label="Arousal MAE", zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([a.replace("_", " ") for a in arc_names_sorted], fontsize=8.5)
    ax2.set_xlabel("MAE (warm sessions, 2+)", labelpad=8)
    ax2.set_title("Per-Arc Tracking Accuracy (Warm Sessions)", pad=10)
    ax2.axvline(warm_mae, color="#333333", linewidth=0.8, linestyle=":", alpha=0.6,
                label=f"Overall warm mean ({warm_mae:.3f})")
    ax2.legend(fontsize=8.5)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", left=False)

    fig.suptitle("Longitudinal Emotional Tracking: In-Distribution Simulation",
                 fontsize=13, fontweight="bold", y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Figure 13: PR progression + architecture comparison summary
# ---------------------------------------------------------------------------

def draw_fig13() -> plt.Figure:
    """
    A single combined summary figure: 4 metrics across 3 runs in one view.
    Good for paper / presentation use.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    run_labels = ["Run 1\nJoint LN\n(PR=1.4)", "Run 2\nSplit LN\n(PR=245)", "Run 3\nSplit LN+VAD\n(PR=339.7)"]
    x = np.arange(3)
    run_colors = [C_BAD, C_BASELINE, C_GOOD]

    # 1. PR
    pr_vals = [1.4, 245, 339.7]
    ax = axes[0, 0]
    bars = ax.bar(x, pr_vals, color=run_colors, alpha=0.85, zorder=3)
    for b, v in zip(bars, pr_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(run_labels, fontsize=8)
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Effective Dimensionality (PR)", pad=8)
    ax.set_ylim(0, 420)
    ax.axhline(384, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)

    # 2. Val loss
    val_losses = [0.5087, 0.5058, 0.5088]
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, val_losses, color=run_colors, alpha=0.85, zorder=3)
    for b, v in zip(bars2, val_losses):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(run_labels, fontsize=8)
    ax2.set_ylabel("Val cosine prediction loss")
    ax2.set_title("Validation Loss (1 − cos sim)", pad=8)
    ax2.set_ylim(0.49, 0.53)

    # 3. Probe valence r
    r_vals = [0.226, 0.693, 0.698]
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, r_vals, color=run_colors, alpha=0.85, zorder=3)
    for b, v in zip(bars3, r_vals):
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(run_labels, fontsize=8)
    ax3.set_ylabel("Pearson r")
    ax3.set_title("Linear Probe: Valence Pearson r", pad=8)
    ax3.set_ylim(0, 0.85)
    ax3.axhline(0.5, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)

    # 4. Naturalistic spread
    nat_spread_v = [0.034, None, v_range]   # run2 not measured
    ax4 = axes[1, 1]
    bar_vals = [v if v is not None else 0 for v in nat_spread_v]
    hatches  = ["", "//", ""]
    for i, (val, hatch, color) in enumerate(zip(bar_vals, hatches, run_colors)):
        b = ax4.bar(i, val, color=color, alpha=0.85 if val > 0 else 0.3,
                    hatch=hatch, zorder=3)
        label = f"{val:.3f}" if val > 0 else "not\nmeasured"
        ax4.text(i, val + 0.01, label, ha="center", va="bottom",
                 fontsize=9, fontweight="bold" if val > 0 else "normal")
    ax4.set_xticks(x); ax4.set_xticklabels(run_labels, fontsize=8)
    ax4.set_ylabel("Valence range on naturalistic text")
    ax4.set_title("Probe Generalization (naturalistic spread)", pad=8)
    ax4.set_ylim(0, 1.4)

    fig.suptitle(
        "Kokoro Architecture Evolution: Three Training Runs Compared",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(h_pad=4.0, w_pad=3.0)
    return fig


# ---------------------------------------------------------------------------
# Generate and save
# ---------------------------------------------------------------------------

print()
print("Figure 9: world model accuracy")
fig9 = draw_fig9()
save(fig9, "fig9_world_model_accuracy")
plt.close(fig9)

print("Figure 10: probe generalization")
fig10 = draw_fig10()
save(fig10, "fig10_probe_generalization")
plt.close(fig10)

print("Figure 11: probe comparison")
fig11 = draw_fig11()
save(fig11, "fig11_probe_comparison")
plt.close(fig11)

print("Figure 12: longitudinal curves")
fig12 = draw_fig12()
save(fig12, "fig12_longitudinal_curves")
plt.close(fig12)

print("Figure 13: PR progression summary")
fig13 = draw_fig13()
save(fig13, "fig13_pr_progression")
plt.close(fig13)

print("\nDone. All figures saved to figures/")
