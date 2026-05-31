"""
Step 1 visualizations: data pipeline and arc template inspection.

Produces three publication-quality figures:
  fig1_circumplex_scatter.{png,pdf}  — all 17,780 sessions in valence/arousal space
  fig2_arc_trajectories.{png,pdf}    — 4 arc types as paths through circumplex
  fig3_arc_distribution.{png,pdf}    — arc type counts in generated trajectories

Run from project root:
    python figures/visualize_step1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import numpy as np

# Make project importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prepare import EMOTION_TO_CIRCUMPLEX, load_empathetic_dialogues, LabeledSession
from data.arc_templates import ARC_TEMPLATES, ArcTemplate


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def apply_style() -> None:
    plt.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size":           10,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.titlepad":       10,
        "axes.labelsize":      11,
        "axes.labelweight":    "normal",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.8,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "legend.fontsize":     9,
        "legend.framealpha":   0.9,
        "legend.edgecolor":    "#cccccc",
        "legend.borderpad":    0.5,
        "figure.facecolor":    "white",
        "savefig.facecolor":   "white",
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "axes.grid":           True,
        "grid.alpha":          0.25,
        "grid.linewidth":      0.5,
        "grid.color":          "#999999",
    })


# ---------------------------------------------------------------------------
# Emotion color scheme — principled quadrant groupings
# ---------------------------------------------------------------------------

# Q1: positive valence, high arousal  → warm orange family
# Q2: positive valence, low arousal   → teal/green family
# Q3: negative valence, high arousal  → red/crimson family
# Q4: negative valence, low arousal   → indigo/blue family

_Q1_EMOTIONS = ["joyful", "excited", "proud", "hopeful", "impressed",
                "anticipating", "surprised", "prepared", "confident"]
_Q2_EMOTIONS = ["content", "grateful", "trusting", "caring",
                "faithful", "nostalgic", "sentimental"]
_Q3_EMOTIONS = ["apprehensive", "anxious", "afraid", "terrified",
                "angry", "furious", "annoyed", "disgusted", "jealous"]
_Q4_EMOTIONS = ["sad", "devastated", "lonely", "ashamed",
                "embarrassed", "guilty", "disappointed"]

_QUADRANT_LABELS = {
    "q1": "Excited (+ valence, + arousal)",
    "q2": "Content (+ valence, − arousal)",
    "q3": "Distressed (− valence, + arousal)",
    "q4": "Depressed (− valence, − arousal)",
}
_QUADRANT_BASE_COLORS = {
    "q1": "#D4810A",
    "q2": "#1A7A4A",
    "q3": "#C0392B",
    "q4": "#2C3E7A",
}


def _interpolated_colors(base_hex: str, n: int) -> list[str]:
    """Return n colors from white → base, skipping the very light end."""
    base = np.array(matplotlib.colors.to_rgb(base_hex))
    white = np.ones(3)
    colors = []
    for i in range(n):
        t = 0.35 + 0.65 * (i / max(n - 1, 1))   # range [0.35, 1.0]
        c = white * (1 - t) + base * t
        colors.append(matplotlib.colors.to_hex(c))
    return colors


def build_emotion_colors() -> dict[str, str]:
    """Map each of the 32 emotion labels to a distinct color."""
    colors: dict[str, str] = {}
    for group, base in zip(
        [_Q1_EMOTIONS, _Q2_EMOTIONS, _Q3_EMOTIONS, _Q4_EMOTIONS],
        [_QUADRANT_BASE_COLORS[q] for q in ["q1", "q2", "q3", "q4"]],
    ):
        for label, color in zip(group, _interpolated_colors(base, len(group))):
            colors[label] = color
    return colors


def emotion_to_quadrant(label: str) -> str:
    if label in _Q1_EMOTIONS:
        return "q1"
    if label in _Q2_EMOTIONS:
        return "q2"
    if label in _Q3_EMOTIONS:
        return "q3"
    return "q4"


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

FIGURES_DIR = Path(__file__).parent


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved {path.relative_to(FIGURES_DIR.parent)}")


# ---------------------------------------------------------------------------
# Background helper for circumplex subplots
# ---------------------------------------------------------------------------

def draw_circumplex_background(ax: plt.Axes, alpha: float = 0.07) -> None:
    """Shade the four circumplex quadrants and draw reference axes."""
    quad_colors = {
        (1, 1): _QUADRANT_BASE_COLORS["q1"],   # positive, high
        (1, -1): _QUADRANT_BASE_COLORS["q2"],  # positive, low
        (-1, 1): _QUADRANT_BASE_COLORS["q3"],  # negative, high
        (-1, -1): _QUADRANT_BASE_COLORS["q4"], # negative, low
    }
    for (sv, sa), color in quad_colors.items():
        ax.fill_between(
            [0, sv * 1.15], [0, 0], [sa * 1.15, sa * 1.15],
            color=color, alpha=alpha, zorder=0,
        )
    # Reference axes through origin
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1, alpha=0.7)
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1, alpha=0.7)


# ---------------------------------------------------------------------------
# Figure 1: Circumplex scatter of all sessions
# ---------------------------------------------------------------------------

def fig1_circumplex_scatter(sessions: list[LabeledSession]) -> plt.Figure:
    print(f"  Plotting {len(sessions)} sessions in circumplex space...")
    rng = np.random.RandomState(42)
    emotion_colors = build_emotion_colors()

    # Jitter amount: small enough that clusters are visible, large enough to
    # show density within each cluster
    JITTER = 0.045

    fig, ax = plt.subplots(figsize=(10, 9))
    draw_circumplex_background(ax, alpha=0.08)

    # Plot jittered session scatter, grouped by quadrant (so layers are clean)
    for quadrant, emotions in zip(
        ["q1", "q2", "q3", "q4"],
        [_Q1_EMOTIONS, _Q2_EMOTIONS, _Q3_EMOTIONS, _Q4_EMOTIONS],
    ):
        mask = [s.emotion_label in emotions for s in sessions]
        subset = [s for s, m in zip(sessions, mask) if m]
        if not subset:
            continue
        vs = np.array([s.valence for s in subset])
        as_ = np.array([s.arousal for s in subset])
        vs += rng.normal(0, JITTER, len(subset))
        as_ += rng.normal(0, JITTER, len(subset))

        # Color each point by its specific emotion
        point_colors = [emotion_colors[s.emotion_label] for s in subset]
        ax.scatter(
            vs, as_,
            c=point_colors, s=7, alpha=0.35, linewidths=0,
            zorder=2, rasterized=True,   # rasterize for PDF file size
        )

    # Centroid markers and labels — offset text away from origin
    for label, (v, a) in EMOTION_TO_CIRCUMPLEX.items():
        color = emotion_colors[label]
        # Centroid marker
        ax.scatter(v, a, s=60, c=color, edgecolors="white",
                   linewidths=0.8, zorder=5)

        # Offset direction: away from (0,0), then nudge slightly upward
        angle = np.arctan2(a, v)
        r_offset = 0.13
        tx = v + r_offset * np.cos(angle)
        ty = a + r_offset * np.sin(angle)

        ax.text(
            tx, ty, label,
            fontsize=7.5, color=color, ha="center", va="center",
            fontweight="semibold",
            path_effects=[
                pe.withStroke(linewidth=2, foreground="white")
            ],
            zorder=6,
        )

    # Quadrant corner labels
    quadrant_text = [
        ( 0.95,  1.08, "Excited", "right", _QUADRANT_BASE_COLORS["q1"]),
        ( 0.95, -1.05, "Content", "right", _QUADRANT_BASE_COLORS["q2"]),
        (-1.08,  1.08, "Distressed", "left",  _QUADRANT_BASE_COLORS["q3"]),
        (-1.08, -1.05, "Depressed", "left",  _QUADRANT_BASE_COLORS["q4"]),
    ]
    for x, y, text, ha, color in quadrant_text:
        ax.text(x, y, text, fontsize=10, color=color, ha=ha,
                fontweight="bold", alpha=0.65, zorder=3,
                style="italic")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Valence   (negative ← · · · → positive)", labelpad=8)
    ax.set_ylabel("Arousal   (calm ← · · · → activated)", labelpad=8)
    ax.set_title(
        f"Session Pool: {len(sessions):,} sessions across 32 emotion labels\n"
        f"(Russell circumplex space, jitter σ = {JITTER})",
        pad=12,
    )
    ax.set_aspect("equal")

    # Quadrant legend
    legend_handles = [
        mpatches.Patch(color=_QUADRANT_BASE_COLORS[q], alpha=0.75, label=_QUADRANT_LABELS[q])
        for q in ["q1", "q2", "q3", "q4"]
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              title="Circumplex quadrant", title_fontsize=8,
              framealpha=0.92)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Four arc types as paths through circumplex
# ---------------------------------------------------------------------------

def _arc_path_from_template(template: ArcTemplate) -> tuple[list[float], list[float]]:
    """Extract the sequence of zone-center (v, a) pairs from an arc template."""
    vs = [z.valence for z in template.sessions]
    as_ = [z.arousal for z in template.sessions]
    return vs, as_


def _draw_arc_on_ax(
    ax: plt.Axes,
    vs: list[float],
    as_: list[float],
    week_offsets: list[int] | None,
    title: str,
    cmap_name: str = "plasma",
) -> None:
    draw_circumplex_background(ax, alpha=0.09)
    n = len(vs)
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / (n - 1)) for i in range(n)]

    # Draw gradient path segments
    for i in range(n - 1):
        # Segment color = midpoint color
        seg_color = cmap((i + 0.5) / (n - 1))
        ax.annotate(
            "", xy=(vs[i + 1], as_[i + 1]), xytext=(vs[i], as_[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=seg_color,
                lw=1.8,
                mutation_scale=14,
                shrinkA=8,
                shrinkB=8,
            ),
            zorder=4,
        )

    # Session markers
    for i, (v, a, color) in enumerate(zip(vs, as_, colors)):
        ax.scatter(v, a, s=90, color=color, edgecolors="white",
                   linewidths=1.2, zorder=5)
        label = f"S{i}" if week_offsets is None else f"W{week_offsets[i]}"
        ax.text(v + 0.06, a + 0.06, label, fontsize=7.5, color=color,
                fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Colorbar: early → late
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Session index", fontsize=8)
    cb.set_ticks([0, n - 1])
    cb.set_ticklabels(["Start", "End"])
    cb.ax.tick_params(labelsize=7)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Valence", fontsize=9, labelpad=4)
    ax.set_ylabel("Arousal", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal")


def fig2_arc_trajectories(trajectories: list[dict]) -> plt.Figure:
    print("  Plotting arc trajectories...")
    TARGET_ARCS = [
        "gradual_decline",
        "slow_recovery",
        "acute_stress_stabilization",
        "grief_arc",
    ]
    CMAPS = ["plasma", "viridis", "inferno", "cividis"]
    TITLES = [
        "Gradual Decline",
        "Slow Recovery",
        "Acute Stress → Stabilization",
        "Grief Arc",
    ]

    # Find one representative trajectory per arc type
    arc_examples: dict[str, dict] = {}
    for traj in trajectories:
        name = traj["arc_name"]
        if name in TARGET_ARCS and name not in arc_examples:
            arc_examples[name] = traj
        if len(arc_examples) == len(TARGET_ARCS):
            break

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for ax, arc_name, cmap_name, title in zip(axes, TARGET_ARCS, CMAPS, TITLES):
        if arc_name not in arc_examples:
            # Fall back to template zone centers if no instance found
            template = next(t for t in ARC_TEMPLATES if t.name == arc_name)
            vs, as_ = _arc_path_from_template(template)
            week_offsets = None
        else:
            traj = arc_examples[arc_name]
            vs = [s["valence"] for s in traj["sessions"]]
            as_ = [s["arousal"] for s in traj["sessions"]]
            week_offsets = [s["week_offset"] for s in traj["sessions"]]

        _draw_arc_on_ax(ax, vs, as_, week_offsets, title, cmap_name)

    fig.suptitle(
        "Four Emotional Arc Types: Trajectories Through Circumplex Space\n"
        "(Actual sampled sessions from EmpatheticDialogues pool)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Arc distribution bar chart
# ---------------------------------------------------------------------------

def fig3_arc_distribution(trajectories: list[dict]) -> plt.Figure:
    print("  Plotting arc distribution...")
    from collections import Counter

    counts = Counter(t["arc_name"] for t in trajectories)
    # Sort by count descending
    sorted_arcs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_arcs)

    # Assign colors based on arc valence direction (from templates)
    def arc_trend_color(arc_name: str) -> str:
        """Color by whether the arc ends in a better or worse state."""
        template = next((t for t in ARC_TEMPLATES if t.name == arc_name), None)
        if template is None:
            return "#888888"
        start_v = template.sessions[0].valence
        end_v = template.sessions[-1].valence
        delta = end_v - start_v
        if delta > 0.2:
            return _QUADRANT_BASE_COLORS["q1"]   # improving
        elif delta < -0.2:
            return _QUADRANT_BASE_COLORS["q3"]   # declining
        else:
            return "#888888"                      # stable/oscillating

    bar_colors = [arc_trend_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="white",
                   linewidth=0.5, height=0.65)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Number of trajectories", labelpad=8)
    ax.set_title(
        f"Trajectory Distribution Across Arc Types  (n = {sum(values):,} total)",
        pad=10,
    )
    ax.set_xlim(0, max(values) * 1.15)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.invert_yaxis()

    # Legend
    legend_handles = [
        mpatches.Patch(color=_QUADRANT_BASE_COLORS["q1"], label="Net improving (+Δv > 0.2)"),
        mpatches.Patch(color=_QUADRANT_BASE_COLORS["q3"], label="Net declining (−Δv > 0.2)"),
        mpatches.Patch(color="#888888",                   label="Stable / oscillating"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    apply_style()

    print("Loading session pool...")
    sessions = load_empathetic_dialogues("train")
    print(f"  {len(sessions)} sessions loaded")

    print("Loading trajectories...")
    traj_path = Path(__file__).parent.parent / "data" / "trajectories_sample.json"
    if not traj_path.exists():
        print(f"  {traj_path} not found — generating 500 trajectories...")
        from data.construct_trajectories import construct_trajectories, save_trajectories
        trajs_obj = construct_trajectories(sessions, n_trajectories=500)
        save_trajectories(trajs_obj, traj_path)
        trajectories = [t.to_dict() for t in trajs_obj]
    else:
        with open(traj_path) as f:
            trajectories = json.load(f)
    print(f"  {len(trajectories)} trajectories loaded")

    print("\nFigure 1: Circumplex scatter")
    fig1 = fig1_circumplex_scatter(sessions)
    save(fig1, "fig1_circumplex_scatter")
    plt.close(fig1)

    print("Figure 2: Arc trajectories")
    fig2 = fig2_arc_trajectories(trajectories)
    save(fig2, "fig2_arc_trajectories")
    plt.close(fig2)

    print("Figure 3: Arc distribution")
    fig3 = fig3_arc_distribution(trajectories)
    save(fig3, "fig3_arc_distribution")
    plt.close(fig3)

    print("\nDone.")
