"""
figures/fig0_architecture.py

Static architecture diagram of the Kokoro pipeline.
Left-to-right flowchart with two rows:
  UPDATE PATH    — what happens after each session
  INFERENCE PATH — what happens before each LLM reply

Output:
    figures/fig0_architecture.png  (200 DPI)
    figures/fig0_architecture.pdf  (vector, paper quality)

Run:
    python figures/fig0_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Palette — two colours only
# ---------------------------------------------------------------------------
C_KO    = "#2E6DBF"   # Kokoro components    (blue)
C_EXT   = "#5A5A6E"   # External             (slate)
C_EK    = "#1A4D99"   # Kokoro edge
C_EE    = "#3A3A4E"   # External edge
C_ARROW = "#444455"   # Arrow / line colour
C_ALABEL= "#555566"   # Arrow data-type label
C_CROSS = "#999AAB"   # Cross-path connector
C_DIV   = "#DDDDEE"   # Divider line
C_ROW   = "#AAAABC"   # Row label
C_BG    = "#FFFFFF"
C_TITLE = "#1A1A2E"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FW, FH = 14.0, 5.4
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BW_DEF = 1.55
BH_DEF = 0.68

def draw_box(
    x: float, y: float, label: str,
    kokoro: bool = True,
    w: float = BW_DEF, h: float = BH_DEF,
) -> None:
    fc = C_KO  if kokoro else C_EXT
    ec = C_EK  if kokoro else C_EE
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3,
    ))
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=7.4, color="white", fontweight="bold",
        multialignment="center", linespacing=1.40, zorder=4,
    )


def arrow(
    x1: float, y1: float,
    x2: float, y2: float,
    label: str = "",
    label_dy: float = 0.13,
    color: str = C_ARROW,
) -> None:
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.15,
            mutation_scale=9,
        ),
        zorder=2,
    )
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + label_dy,
            label,
            ha="center", va="bottom",
            fontsize=6.1, color=C_ALABEL, style="italic", zorder=5,
        )


def elbow_arrow(
    x_start: float, y_start: float,
    x_mid: float,   y_mid: float,
    x_end: float,   y_end: float,
    label: str = "",
    color: str = C_CROSS,
) -> None:
    """Draw an L-shaped connector: down → left → down, with arrowhead at end."""
    # Segment 1: vertical down from start to y_mid
    ax.plot([x_start, x_start], [y_start, y_mid], color=color, lw=1.0,
            zorder=2, linestyle="--", dashes=(4, 3))
    # Segment 2: horizontal to x_end at y_mid
    ax.plot([x_start, x_end], [y_mid, y_mid], color=color, lw=1.0,
            zorder=2, linestyle="--", dashes=(4, 3))
    # Segment 3: vertical down to end, with arrowhead
    ax.annotate(
        "",
        xy=(x_end, y_end), xytext=(x_end, y_mid),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.0,
            mutation_scale=8, linestyle="dashed",
        ),
        zorder=2,
    )
    if label:
        # Label on the horizontal segment
        lx = (x_start + x_end) / 2
        ly = y_mid + 0.10
        ax.text(lx, ly, label, ha="center", va="bottom",
                fontsize=5.9, color=color, style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
YU   = 3.90   # update row centre y
YI   = 1.55   # inference row centre y
YMID = (YU + YI) / 2   # 2.725

X1 = 1.15    # Session Turns / New Message
X2 = 3.35    # SessionEncoder
X3 = 5.70    # TransitionModel / MemoryStore.retrieve
X4 = 8.20    # StateStore+Probe / StateDecoder
X5 = 11.20   # MemoryStore / LLM Prompt

W_WIDE = 1.72   # wider boxes

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(
    FW / 2, FH - 0.30,
    "Kokoro  —  Pipeline Architecture",
    ha="center", va="center",
    fontsize=12, color=C_TITLE, fontweight="bold",
)

# ---------------------------------------------------------------------------
# Row labels
# ---------------------------------------------------------------------------
ax.text(0.28, YU, "UPDATE\nPATH",
        ha="center", va="center", fontsize=6.8,
        color=C_ROW, fontweight="bold", multialignment="center")
ax.text(0.28, YI, "INFERENCE\nPATH",
        ha="center", va="center", fontsize=6.8,
        color=C_ROW, fontweight="bold", multialignment="center")

# Divider
ax.axhline(y=YMID, xmin=0.04, xmax=0.98,
           color=C_DIV, lw=0.9, zorder=1)

# ---------------------------------------------------------------------------
# UPDATE PATH boxes
# ---------------------------------------------------------------------------
draw_box(X1, YU, "Session\nTurns",             kokoro=False)
draw_box(X2, YU, "Session\nEncoder",            kokoro=True)
draw_box(X3, YU, "Transition\nModel",           kokoro=True)
draw_box(X4, YU, "State Store\n+ Linear Probe", kokoro=True,  w=W_WIDE)
draw_box(X5, YU, "Memory Store\n(ChromaDB)",    kokoro=True,  w=W_WIDE)

# Update arrows
arrow(X1 + BW_DEF/2,    YU,
      X2 - BW_DEF/2,    YU,  label="turns")

arrow(X2 + BW_DEF/2,    YU,
      X3 - BW_DEF/2,    YU,  label="384-dim embedding")

arrow(X3 + BW_DEF/2,    YU,
      X4 - W_WIDE/2,    YU,  label="state vector (384-dim)")

arrow(X4 + W_WIDE/2,    YU,
      X5 - W_WIDE/2,    YU,  label="embedding  +  valence/arousal")

# ---------------------------------------------------------------------------
# INFERENCE PATH boxes
# ---------------------------------------------------------------------------
draw_box(X1, YI, "New\nMessage",               kokoro=False)
draw_box(X2, YI, "Session\nEncoder",           kokoro=True)
draw_box(X3, YI, "MemoryStore\n.retrieve()",   kokoro=True,  w=W_WIDE)
draw_box(X4, YI, "State\nDecoder",             kokoro=True)
draw_box(X5, YI, "LLM System\nPrompt",         kokoro=False, w=W_WIDE)

# Inference arrows
arrow(X1 + BW_DEF/2,   YI,
      X2 - BW_DEF/2,   YI,  label="text")

arrow(X2 + BW_DEF/2,   YI,
      X3 - W_WIDE/2,   YI,  label="query embedding (384-dim)")

arrow(X3 + W_WIDE/2,   YI,
      X4 - BW_DEF/2,   YI,  label="top-k sessions")

arrow(X4 + BW_DEF/2,   YI,
      X5 - W_WIDE/2,   YI,  label="state_summary  +  memories")

# ---------------------------------------------------------------------------
# Cross-path connection:
#   State Store + Linear Probe  ──►  MemoryStore.retrieve
#   (state vector used as emotional axis during retrieval)
# ---------------------------------------------------------------------------
elbow_arrow(
    x_start = X4,
    y_start = YU - BH_DEF/2,
    x_mid   = YMID,
    y_mid   = YMID,
    x_end   = X3,
    y_end   = YI + BH_DEF/2,
    label   = "state vec  (emotional axis)",
)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
LX, LY = 12.55, 5.05
LS = 0.26   # swatch size

ax.add_patch(FancyBboxPatch(
    (LX, LY - LS / 2), LS, LS,
    boxstyle="round,pad=0.03",
    facecolor=C_KO, edgecolor=C_EK, lw=1.0, zorder=6,
))
ax.text(LX + LS + 0.12, LY, "Kokoro component",
        va="center", fontsize=6.8, color="#333344")

ax.add_patch(FancyBboxPatch(
    (LX, LY - LS / 2 - 0.45), LS, LS,
    boxstyle="round,pad=0.03",
    facecolor=C_EXT, edgecolor=C_EE, lw=1.0, zorder=6,
))
ax.text(LX + LS + 0.12, LY - 0.45, "External",
        va="center", fontsize=6.8, color="#333344")

# Dashed swatch for cross-path line
ax.plot(
    [LX, LX + LS], [LY - 0.90, LY - 0.90],
    color=C_CROSS, lw=1.1, linestyle="--", dashes=(4, 3), zorder=6,
)
ax.text(LX + LS + 0.12, LY - 0.90, "State vector cross-path",
        va="center", fontsize=6.8, color="#333344")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout(pad=0.5)

png_path = OUT_DIR / "fig0_architecture.png"
pdf_path = OUT_DIR / "fig0_architecture.pdf"

fig.savefig(str(png_path), dpi=200, bbox_inches="tight", facecolor=C_BG)
fig.savefig(str(pdf_path),           bbox_inches="tight", facecolor=C_BG)

plt.close(fig)
print(f"Saved PNG: {png_path}  ({png_path.stat().st_size // 1024} KB)")
print(f"Saved PDF: {pdf_path}")
