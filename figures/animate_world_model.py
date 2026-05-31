"""
figures/animate_world_model.py

Animated GIF showing a user's emotional state vector moving through
Russell's circumplex space over sessions — hero asset for the README.

Uses the s01 (Marcus, gradual_decline) scenario from experiments/scenarios.py.

Output:
    figures/kokoro_demo.gif   (always)
    figures/kokoro_demo.mp4   (if ffmpeg is available)

Run:
    python figures/animate_world_model.py
"""

from __future__ import annotations

# ── load heavy torch/encoder imports FIRST to avoid SSL conflict ──────────────
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

# Collect state vectors from WorldMemory before touching matplotlib
import chromadb as _chromadb
_chromadb.PersistentClient = lambda path, **kw: _chromadb.EphemeralClient()

from kokoro import WorldMemory
from experiments.scenarios import SCENARIOS

# ── pick s01 ──────────────────────────────────────────────────────────────────
scenario = next(s for s in SCENARIOS if s["scenario_id"] == "s01")
sessions   = scenario["sessions"]
n_sessions = len(sessions)

# ── run WorldMemory, snapshot state after every session ───────────────────────
tmp = tempfile.mkdtemp()

memory = WorldMemory(
    user_id     = "anim_marcus",
    db_path     = Path(tmp) / "kokoro.db",
    persist_dir = tmp,
    min_sessions = 3,
    alpha        = 0.6,
    top_k        = 3,
)

state_vecs: list[np.ndarray] = []   # (384,) after each session
valences:   list[float]      = []
arousals:   list[float]      = []
summaries:  list[str]        = []   # first user turn of each session

for sess in sessions:
    memory.update(sess)
    vec  = memory._state_store.load("anim_marcus")          # (384,)
    info = memory._state_store.get_info("anim_marcus")
    state_vecs.append(vec.copy())
    valences.append(info["valence"])
    arousals.append(info["arousal"])
    # Grab first user utterance as label
    user_turns = [t["content"] for t in sess if t["role"] == "user"]
    summaries.append((user_turns[0][:80] + "…") if user_turns else "")

# ── PCA to 2D — fit on ALL state vectors so axes are stable ──────────────────
from sklearn.decomposition import PCA  # type: ignore

X = np.stack(state_vecs)           # (n_sessions, 384)
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(X)   # (n_sessions, 2)

# Re-scale coords to [-1, 1] so they live naturally in circumplex space
# and re-interpret axes as valence-like (x) and arousal-like (y)
# by aligning PC directions with the probe's valence/arousal readings.
# Simple: flip axes so the observed valence/arousal correlation is positive.
for axis in range(2):
    corr_v = np.corrcoef(coords_2d[:, axis], valences)[0, 1]
    corr_a = np.corrcoef(coords_2d[:, axis], arousals)[0, 1]
    # axis 0 -> valence, axis 1 -> arousal; flip if anti-correlated
    if axis == 0 and corr_v < 0:
        coords_2d[:, 0] *= -1
    if axis == 1 and corr_a < 0:
        coords_2d[:, 1] *= -1

# Normalise to [-0.9, 0.9] for display
for ax in range(2):
    lo, hi = coords_2d[:, ax].min(), coords_2d[:, ax].max()
    span = hi - lo if hi > lo else 1.0
    coords_2d[:, ax] = (coords_2d[:, ax] - lo) / span * 1.6 - 0.8

# ── now import matplotlib ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

# ── figure setup ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 8, 5          # inches
DPI          = 100            # 800×500 px
FPS          = 20             # frames per second
FRAMES_PER_SESSION = 40       # 2 seconds per session at 20 fps
TRAIL_LEN    = n_sessions     # show full trail

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── circumplex background quadrants ──────────────────────────────────────────
ALPHA_BG = 0.18
ax.fill_betweenx([-1, 0], -1, 0, color="#e74c3c", alpha=ALPHA_BG)   # distressed (−V, +A) top-left
ax.fill_betweenx([0, 1],  -1, 0, color="#e74c3c", alpha=ALPHA_BG)   # distressed overlap (top-left)

# Redraw correctly: quadrant colours by (valence, arousal) sign
quad_colors = {
    "distressed": ("#e74c3c", [-1, 0], [0, 1]),    # -V, +A
    "excited":    ("#f39c12", [0, 1],  [0, 1]),    # +V, +A
    "content":    ("#2ecc71", [0, 1],  [-1, 0]),   # +V, -A
    "depressed":  ("#3498db", [-1, 0], [-1, 0]),   # -V, -A
}

ax.cla()
ax.set_facecolor("#0d1117")

for label, (color, xrange, yrange) in quad_colors.items():
    ax.fill_between(
        xrange, yrange[0], yrange[1],
        color=color, alpha=ALPHA_BG
    )

# Circumplex axes
ax.axhline(0, color="white", lw=0.6, alpha=0.3)
ax.axvline(0, color="white", lw=0.6, alpha=0.3)

# Quadrant labels
label_kw = dict(fontsize=9, alpha=0.55, ha="center", va="center",
                color="white", style="italic")
ax.text(-0.5,  0.75, "Distressed", **label_kw)
ax.text( 0.5,  0.75, "Excited",    **label_kw)
ax.text( 0.5, -0.75, "Content",    **label_kw)
ax.text(-0.5, -0.75, "Depressed",  **label_kw)

# Axis labels
ax.set_xlabel("Valence  (negative ← -> positive)", color="white",
              fontsize=9, labelpad=6)
ax.set_ylabel("Arousal  (calm ← -> activated)", color="white",
              fontsize=9, labelpad=6)
ax.tick_params(colors="white", labelsize=7)
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")

# Title
ax.set_title("Kokoro — Emotional Trajectory Memory",
             color="white", fontsize=13, fontweight="bold", pad=10)

# ── artist objects we'll update each frame ────────────────────────────────────
trail_line,  = ax.plot([], [], color="#aaaacc", lw=1.2, alpha=0.5, zorder=3)
dot,          = ax.plot([], [], "o", color="#e040fb", ms=12, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.2)

# Session marker dots (historical)
hist_scatter = ax.scatter([], [], c=[], cmap="plasma",
                          vmin=0, vmax=n_sessions - 1,
                          s=40, zorder=4, alpha=0.6, edgecolors="white",
                          linewidths=0.5)

# Counter text
counter_txt = ax.text(
    0.02, 0.97, "", transform=ax.transAxes,
    color="white", fontsize=10, fontweight="bold",
    va="top", ha="left", zorder=6,
)

# Session summary text box
summary_box = ax.text(
    0.02, 0.03, "", transform=ax.transAxes,
    color="#ccccdd", fontsize=7.5, va="bottom", ha="left",
    wrap=True, zorder=6,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
              edgecolor="#444466", alpha=0.85),
)

# Valence / arousal readout
readout_txt = ax.text(
    0.98, 0.97, "", transform=ax.transAxes,
    color="#aaaacc", fontsize=8, va="top", ha="right", zorder=6,
)

# ── interpolation helper ──────────────────────────────────────────────────────
def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


# ── animation function ────────────────────────────────────────────────────────
total_frames = n_sessions * FRAMES_PER_SESSION

def init():
    trail_line.set_data([], [])
    dot.set_data([], [])
    counter_txt.set_text("")
    summary_box.set_text("")
    readout_txt.set_text("")
    return trail_line, dot, counter_txt, summary_box, readout_txt


def animate(frame: int):
    # Which session are we transitioning INTO?
    sess_idx  = frame // FRAMES_PER_SESSION          # 0-based target session
    local_t   = (frame % FRAMES_PER_SESSION) / FRAMES_PER_SESSION  # 0->1 within session

    # Smooth ease-in-out
    t = local_t * local_t * (3 - 2 * local_t)

    # Current interpolated position
    if sess_idx == 0:
        pos = coords_2d[0]
    else:
        src = coords_2d[sess_idx - 1]
        dst = coords_2d[sess_idx]
        pos = _lerp(src, dst, t)

    # Trail: all completed sessions up to current
    n_trail = sess_idx + 1
    trail_x = coords_2d[:n_trail, 0]
    trail_y = coords_2d[:n_trail, 1]

    trail_line.set_data(trail_x, trail_y)
    dot.set_data([pos[0]], [pos[1]])

    # Historical session dots
    if n_trail > 1:
        hist_scatter.set_offsets(coords_2d[:n_trail - 1])
        hist_scatter.set_array(np.arange(n_trail - 1, dtype=float))

    # Counter
    counter_txt.set_text(f"Session {sess_idx + 1} / {n_sessions}")

    # Summary — fade in during first half of transition
    if t > 0.4:
        summary_box.set_text(f'"{summaries[sess_idx]}"')
        summary_box.set_alpha(min(1.0, (t - 0.4) / 0.3))
    else:
        summary_box.set_alpha(0)

    # Valence / arousal readout (interpolate scalar values)
    if sess_idx == 0:
        v, a = valences[0], arousals[0]
    else:
        v = valences[sess_idx - 1] + (valences[sess_idx] - valences[sess_idx - 1]) * t
        a = arousals[sess_idx - 1] + (arousals[sess_idx] - arousals[sess_idx - 1]) * t
    readout_txt.set_text(f"valence {v:+.3f}  arousal {a:+.3f}")

    return trail_line, dot, hist_scatter, counter_txt, summary_box, readout_txt


anim = FuncAnimation(
    fig, animate,
    frames=total_frames,
    init_func=init,
    interval=1000 // FPS,
    blit=True,
)

# ── save GIF ─────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
gif_path = OUT_DIR / "kokoro_demo.gif"

print(f"Saving GIF ({total_frames} frames @ {FPS}fps) -> {gif_path} …")
writer = PillowWriter(fps=FPS)
anim.save(str(gif_path), writer=writer, dpi=DPI)
print(f"  Saved: {gif_path}  ({gif_path.stat().st_size // 1024} KB)")

# ── save MP4 if ffmpeg available ──────────────────────────────────────────────
mp4_path = OUT_DIR / "kokoro_demo.mp4"
try:
    from matplotlib.animation import FFMpegWriter
    mp4_writer = FFMpegWriter(fps=FPS, bitrate=1800)
    print(f"Saving MP4 -> {mp4_path} …")
    anim.save(str(mp4_path), writer=mp4_writer, dpi=DPI)
    print(f"  Saved: {mp4_path}  ({mp4_path.stat().st_size // 1024} KB)")
except Exception:
    pass   # ffmpeg not available — silently skip

plt.close(fig)
print("\nDone.")
