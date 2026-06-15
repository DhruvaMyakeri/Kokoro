"""
figures/fig0_architecture.py

Publication-quality architecture diagram for Kokoro.

Three zones:
  TOP    — temporal state unrolling across sessions (world model framing)
  BOTTOM-LEFT  — TransitionModel internals (split LayerNorm detail)
  BOTTOM-RIGHT — VICReg loss breakdown + probe/decoder output

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_BG       = "#FAFAFA"
C_ENCODER  = "#6B3FA0"   # purple — SessionEncoder
C_MODEL    = "#1A5276"   # dark blue — TransitionModel
C_LOSS     = "#B7472A"   # deep red-orange — VICReg loss
C_PROBE    = "#1D6A39"   # dark green — LinearProbe
C_DECODER  = "#0E6655"   # teal — StateDecoder
C_SESSION  = "#4A4A5E"   # slate — external / session input
C_STATE    = "#2C3E50"   # navy — state vector
C_VICREG_S = "#E74C3C"   # sim term
C_VICREG_V = "#E67E22"   # var term
C_VICREG_C = "#8E44AD"   # cov term
C_TITLE    = "#1A1A2E"
C_PANEL    = "#ECF0F1"   # light gray panel background
C_PANEL_B  = "#BDC3C7"   # panel border
C_ARROW    = "#555566"
C_DIM      = "#7F8C8D"   # dim label text
C_WHITE    = "#FFFFFF"

FW, FH = 18.0, 9.8
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def box(ax, cx, cy, label, color, w=1.7, h=0.62, fontsize=8.2,
        subtext=None, sub_fontsize=6.8, text_color=C_WHITE, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07",
        facecolor=color, edgecolor=_darken(color, 0.6),
        linewidth=lw, zorder=4,
    ))
    label_y = cy + (0.09 if subtext else 0)
    ax.text(cx, label_y, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            multialignment="center", linespacing=1.35, zorder=5)
    if subtext:
        ax.text(cx, cy - 0.12, subtext, ha="center", va="center",
                fontsize=sub_fontsize, color=_lighten(text_color, 0.6),
                multialignment="center", zorder=5)


def _darken(hex_color: str, factor: float = 0.7) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor))


def _lighten(hex_color: str, factor: float = 1.4) -> str:
    if hex_color == C_WHITE:
        return "#DDDDDD"
    r = min(255, int(int(hex_color[1:3], 16) * factor))
    g = min(255, int(int(hex_color[3:5], 16) * factor))
    b = min(255, int(int(hex_color[5:7], 16) * factor))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def arr(ax, x1, y1, x2, y2, label="", ldy=0.12, color=C_ARROW,
        lw=1.3, fontsize=6.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=9), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + ldy
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=fontsize, color=C_DIM, style="italic", zorder=6)


def horiz_arr(ax, x1, x2, y, label="", color=C_ARROW, lw=1.4, fontsize=6.4):
    arr(ax, x1, y, x2, y, label=label, ldy=0.10, color=color, lw=lw,
        fontsize=fontsize)


def panel_bg(ax, x0, y0, x1, y1, label="", fontsize=8.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0.10",
        facecolor=C_PANEL, edgecolor=C_PANEL_B,
        linewidth=1.0, zorder=1, alpha=0.7,
    ))
    if label:
        ax.text(x0+0.18, y1-0.18, label,
                ha="left", va="top", fontsize=fontsize,
                color="#666677", fontweight="bold", zorder=2)


# ===========================================================================
# TITLE
# ===========================================================================
ax.text(FW/2, FH - 0.32,
        "Kokoro — Affective World Model for AI Companions",
        ha="center", va="center",
        fontsize=13.5, color=C_TITLE, fontweight="bold", zorder=6)
ax.text(FW/2, FH - 0.68,
        r"World model objective:  $s_{t+1} = f_\theta(s_t,\; \mathrm{enc}(\mathrm{session}_t))$"
        r"      trained with VICReg to prevent dimensional collapse",
        ha="center", va="center",
        fontsize=9.0, color="#444455", style="italic", zorder=6)

# Thin rule below title
ax.axhline(FH - 0.90, xmin=0.02, xmax=0.98, color="#CCCCDD", lw=0.9, zorder=2)


# ===========================================================================
# ZONE A — Temporal unrolling (top zone)
# ===========================================================================
panel_bg(ax, 0.30, 4.35, 17.70, FH - 1.00, label="Temporal State Evolution (3 of N sessions shown)")

# Y coordinates
Y_SESS  = FH - 1.75   # session input boxes
Y_ENC   = FH - 2.62   # encoder boxes
Y_STATE = FH - 3.55   # state vector line / transition model
Y_TGT   = FH - 4.20   # target state (for VICReg)

# X positions for 3 steps + ellipsis + inference end
X_S = [2.40, 5.90, 9.40]    # step x-centres
X_END = 13.20                 # final state / inference block

BW = 1.60    # box width
BH = 0.60    # box height
EW = 1.55    # encoder box width

# Initial zero state
ax.text(0.80, Y_STATE, r"$s_0 = \mathbf{0}$",
        ha="center", va="center",
        fontsize=8.5, color=C_STATE, fontweight="bold", zorder=5)
ax.add_patch(FancyBboxPatch(
    (0.30, Y_STATE - 0.22), 1.0, 0.44,
    boxstyle="round,pad=0.06",
    facecolor="#D5D8DC", edgecolor=C_STATE, lw=1.2, zorder=4,
))
ax.text(0.80, Y_STATE, r"$s_0 = \mathbf{0}$",
        ha="center", va="center",
        fontsize=8.0, color=C_STATE, fontweight="bold", zorder=5)

for i, xc in enumerate(X_S):
    step = i + 1

    # Session input box
    box(ax, xc, Y_SESS, f"Session {step}\n(conversation)", C_SESSION,
        w=BW, h=BH, fontsize=8.0)

    # Encoder box
    box(ax, xc, Y_ENC, "SessionEncoder\nMiniLM + VAD", C_ENCODER,
        w=EW, h=BH, fontsize=7.6)

    # Dimension label below encoder
    ax.text(xc, Y_ENC - 0.42, "387-dim emb",
            ha="center", va="top", fontsize=6.4, color=C_DIM, style="italic")

    # Session → Encoder arrow
    arr(ax, xc, Y_SESS - BH/2, xc, Y_ENC + BH/2, label="", color=C_ARROW, lw=1.2)

    # Encoder → TransitionModel (represented at state level)
    arr(ax, xc, Y_ENC - BH/2 - 0.05, xc, Y_STATE + BH/2 + 0.08,
        label="", color=C_ENCODER, lw=1.1)
    ax.text(xc + 0.12, (Y_ENC + Y_STATE)/2, f"e{step}",
            ha="left", va="center", fontsize=6.5, color=C_ENCODER, style="italic")

    # TransitionModel box
    box(ax, xc, Y_STATE, f"Transition\nModel $f_θ$", C_MODEL,
        w=BW, h=BH, fontsize=8.0)

    # State arrow from left into this TM box
    left_x = X_S[i-1] + BW/2 if i > 0 else 1.30
    horiz_arr(ax, left_x, xc - BW/2, Y_STATE,
              label=f"$s_{i}$", color=C_STATE, lw=1.5, fontsize=7.5)

    # Output state label (right side of TM box, before next arrow)
    # (will be shown as the arrow label to the right)

# Arrows between TM boxes  →  shown as state labels
# (already handled as "left→box" arrows above)

# Arrow from last TM to final state
horiz_arr(ax, X_S[-1] + BW/2, X_END - 0.55, Y_STATE,
          label=f"$s_3$  (384-dim)", color=C_STATE, lw=1.5, fontsize=7.5)

# Ellipsis between X_S[-1] and X_END
ax.text((X_S[-1] + BW/2 + X_END)/2 + 0.05, Y_STATE + 0.35,
        "…", ha="center", va="center", fontsize=14, color=C_DIM)

# ---- VICReg loss arrows (pointing DOWN from TM boxes) ----
Y_LOSS = Y_STATE - 0.70   # where the loss annotation lives
for i, xc in enumerate(X_S):
    ax.annotate("", xy=(xc, Y_LOSS + 0.08), xytext=(xc, Y_STATE - BH/2),
                arrowprops=dict(arrowstyle="-|>", color=C_LOSS, lw=1.0,
                                mutation_scale=7, linestyle="dashed"),
                zorder=3)
    loss_box_w = 1.55
    ax.add_patch(FancyBboxPatch(
        (xc - loss_box_w/2, Y_LOSS - 0.28), loss_box_w, 0.32,
        boxstyle="round,pad=0.05",
        facecolor="#FDEDEC", edgecolor=C_LOSS, lw=0.9, zorder=4, alpha=0.9,
    ))
    ax.text(xc, Y_LOSS - 0.12,
            r"$\mathcal{L}_{sim}+\mathcal{L}_{var}+\mathcal{L}_{cov}$",
            ha="center", va="center", fontsize=6.6, color=C_LOSS, zorder=5)
    if i == 0:
        ax.text(xc, Y_LOSS - 0.50, "VICReg  (training)",
                ha="center", va="top", fontsize=6.3, color=C_LOSS,
                style="italic", zorder=5)

# ---- Final state box ----
box(ax, X_END, Y_STATE, f"$s_N$\nfinal state", C_STATE,
    w=1.20, h=BH, fontsize=8.0)

# ---- Probe → Decoder column (right of unrolling) ----
X_PROBE   = 15.20
X_DECODER = 16.95
BH_SIDE   = 0.62

arr(ax, X_END + 0.60, Y_STATE, X_PROBE - 0.82, Y_STATE,
    label="384-dim", color=C_STATE, lw=1.3, fontsize=6.4)

box(ax, X_PROBE, Y_STATE, "Linear\nProbe", C_PROBE,
    w=1.42, h=BH_SIDE, fontsize=8.0)

arr(ax, X_PROBE + 0.71, Y_STATE, X_DECODER - 0.76, Y_STATE,
    label="(V, A)", color=C_PROBE, lw=1.3, fontsize=6.8)

box(ax, X_DECODER, Y_STATE, "State\nDecoder", C_DECODER,
    w=1.42, h=BH_SIDE, fontsize=8.0)

# Quadrant output below decoder
for qi, (ql, qc, qv, qa) in enumerate([
    ("Q1", "#2ECC71", "+V +A", "excited"),
    ("Q2", "#3498DB", "+V −A", "calm"),
    ("Q3", "#E67E22", "−V +A", "anxious"),
    ("Q4", "#95A5A6", "−V −A", "sad"),
]):
    qy = Y_STATE - 0.75 - qi * 0.55
    ax.add_patch(FancyBboxPatch(
        (X_DECODER - 0.68, qy - 0.20), 1.36, 0.40,
        boxstyle="round,pad=0.04",
        facecolor=qc, edgecolor=_darken(qc, 0.7), lw=0.9, zorder=4, alpha=0.85,
    ))
    ax.text(X_DECODER - 0.30, qy, f"{ql}  {qv}", ha="left", va="center",
            fontsize=7.2, color=C_WHITE, fontweight="bold", zorder=5)
    ax.text(X_DECODER + 0.52, qy, qa, ha="center", va="center",
            fontsize=6.4, color=C_WHITE, style="italic", zorder=5)

ax.annotate("", xy=(X_DECODER, Y_STATE - BH_SIDE/2),
            xytext=(X_DECODER, Y_STATE - 0.88),
            arrowprops=dict(arrowstyle="-|>", color=C_DECODER, lw=1.2,
                            mutation_scale=8), zorder=3)
ax.text(X_DECODER, Y_STATE - 0.74, "quadrant\nlabel",
        ha="center", va="top", fontsize=6.4, color=C_DECODER, style="italic")

# "INFERENCE" badge on the probe/decoder
ax.add_patch(FancyBboxPatch(
    (X_PROBE - 0.75, Y_STATE + BH_SIDE/2 + 0.08), 2.42, 0.30,
    boxstyle="round,pad=0.04",
    facecolor="#EBF5FB", edgecolor="#AED6F1", lw=0.9, zorder=4,
))
ax.text(X_PROBE + 0.46, Y_STATE + BH_SIDE/2 + 0.22,
        "INFERENCE",
        ha="center", va="center", fontsize=6.6, color="#1A5276",
        fontweight="bold", zorder=5)


# ===========================================================================
# Divider
# ===========================================================================
ax.axhline(4.25, xmin=0.02, xmax=0.98, color="#CCCCDD", lw=0.9, zorder=2)


# ===========================================================================
# ZONE B — TransitionModel detail (bottom-left)
# ===========================================================================
panel_bg(ax, 0.30, 0.25, 8.60, 4.15,
         label="TransitionModel  f_θ  (split LayerNorm architecture)")

# Y positions inside the panel
Y_IN  = 3.65   # inputs
Y_LN  = 3.05   # LayerNorm blocks
Y_CAT = 2.45   # concat
Y_MLP = 1.75   # MLP block
Y_OUT = 1.05   # output state

# --- State input (left stream) ---
XL = 2.50   # left stream x
XR = 5.30   # right stream x
XM = (XL + XR) / 2   # midpoint for concat

box(ax, XL, Y_IN, "$s_t$\nstate (384-d)", C_STATE, w=1.50, h=0.55, fontsize=8.0)
box(ax, XR, Y_IN, "$e_t$\nembedding (387-d)", C_ENCODER, w=1.62, h=0.55, fontsize=7.8)

# LN blocks
box(ax, XL, Y_LN, "LayerNorm(384)", C_MODEL, w=1.52, h=0.48, fontsize=7.6)
box(ax, XR, Y_LN, "LayerNorm(387)", C_ENCODER, w=1.52, h=0.48, fontsize=7.6)

ax.text(XL - 1.00, (Y_IN + Y_LN)/2,
        "Split LN\n(independent\nnormalization)",
        ha="center", va="center", fontsize=6.8, color="#555566",
        style="italic", multialignment="center")
ax.add_patch(mpatches.FancyArrowPatch(
    (XL - 0.55, Y_LN + 0.05), (XL - 0.76, Y_LN + 0.05),
    arrowstyle="-|>", color="#888899", lw=0.8, mutation_scale=7,
))

arr(ax, XL, Y_IN - 0.28, XL, Y_LN + 0.24, color=C_STATE, lw=1.2)
arr(ax, XR, Y_IN - 0.28, XR, Y_LN + 0.24, color=C_ENCODER, lw=1.2)

# Concat bracket
arr(ax, XL, Y_LN - 0.24, XM, Y_CAT + 0.24, color=C_MODEL, lw=1.0)
arr(ax, XR, Y_LN - 0.24, XM, Y_CAT + 0.24, color=C_MODEL, lw=1.0)

box(ax, XM, Y_CAT, "Concat  [s ‖ e]\n771-dimensional", C_MODEL,
    w=2.10, h=0.50, fontsize=7.8)

arr(ax, XM, Y_CAT - 0.25, XM, Y_MLP + 0.38, color=C_MODEL, lw=1.2)

# MLP block
box(ax, XM, Y_MLP, "3-layer MLP\n771→512→512→384", C_MODEL,
    w=2.20, h=0.68, fontsize=8.0,
    subtext="Linear→ReLU×2→Linear")

arr(ax, XM, Y_MLP - 0.34, XM, Y_OUT + 0.25, color=C_STATE, lw=1.4)

box(ax, XM, Y_OUT, "$s_{t+1}$\nnext state (384-d)", C_STATE,
    w=1.60, h=0.50, fontsize=8.0)

# No L2 norm note
ax.text(XM + 1.15, Y_OUT,
        "No L2\nnormalization\n(removed in fix)",
        ha="left", va="center", fontsize=6.3, color="#B7472A",
        style="italic", multialignment="left")


# ===========================================================================
# ZONE C — VICReg loss + metrics (bottom-right)
# ===========================================================================
panel_bg(ax, 8.80, 0.25, 17.70, 4.15,
         label="VICReg Training Objective  —  prevents dimensional collapse")

# Three loss term cards
CARDS = [
    (C_VICREG_S, "$\\mathcal{L}_{sim}$\n(Invariance)",
     "weight  λ = 25",
     r"$1 - \cos(\hat{s}_{t+1},\; s^\star_{t+1})$",
     "Cosine prediction loss:\npredicted next state ≈ target"),
    (C_VICREG_V, "$\\mathcal{L}_{var}$\n(Variance)",
     "weight  λ = 25",
     r"$\mathrm{mean}(\max(0,\;\gamma - \mathrm{std}(z)))$",
     "γ = 1.0 — forces each dim\nto have std ≥ 1 (no collapse)"),
    (C_VICREG_C, "$\\mathcal{L}_{cov}$\n(Covariance)",
     "weight  λ = 1",
     r"$\sum_{i \neq j} C_{ij}^2 / D$",
     "Penalizes correlated dims,\nforces diverse representations"),
]

XC_CARDS = [10.20, 12.90, 15.60]
YC_TOP   = 3.55
CARD_W   = 2.30
CARD_H   = 2.80

for (color, title, weight, formula, desc), xc in zip(CARDS, XC_CARDS):
    # Card background
    ax.add_patch(FancyBboxPatch(
        (xc - CARD_W/2, YC_TOP - CARD_H), CARD_W, CARD_H,
        boxstyle="round,pad=0.10",
        facecolor=C_WHITE, edgecolor=color, lw=1.6, zorder=3,
    ))
    # Color header strip
    ax.add_patch(FancyBboxPatch(
        (xc - CARD_W/2, YC_TOP - 0.55), CARD_W, 0.55,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=color, lw=0.0, zorder=4,
    ))
    # Title
    ax.text(xc, YC_TOP - 0.27, title,
            ha="center", va="center", fontsize=9.0,
            color=C_WHITE, fontweight="bold", zorder=5,
            multialignment="center", linespacing=1.2)
    # Weight badge
    ax.text(xc, YC_TOP - 0.75, weight,
            ha="center", va="center", fontsize=7.0,
            color=color, fontweight="bold", zorder=5)
    # Formula
    ax.text(xc, YC_TOP - 1.22, formula,
            ha="center", va="center", fontsize=7.8,
            color="#222233", zorder=5, multialignment="center")
    # Description
    ax.text(xc, YC_TOP - 1.88, desc,
            ha="center", va="center", fontsize=6.8,
            color="#555566", zorder=5, multialignment="center",
            linespacing=1.4)

# Total loss formula
Y_TOTAL = 0.68
ax.text((XC_CARDS[0] + XC_CARDS[-1])/2, Y_TOTAL,
        r"$\mathcal{L}_{total} = 25\cdot\mathcal{L}_{sim}"
        r" + 25\cdot\mathcal{L}_{var} + 1\cdot\mathcal{L}_{cov}$",
        ha="center", va="center",
        fontsize=9.5, color=C_LOSS, fontweight="bold", zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDEDEC",
                  edgecolor=C_LOSS, alpha=0.9))

# PR result annotation
ax.text(XC_CARDS[0] - 0.85, Y_TOTAL,
        "Participation Ratio:\n"
        "Baseline: 1.4 / 384\n"
        "Current:  339.7 / 384",
        ha="left", va="center",
        fontsize=7.2, color="#1D6A39", fontweight="bold",
        zorder=5, multialignment="left",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#EAFAF1",
                  edgecolor="#1D6A39", alpha=0.9))


# ===========================================================================
# Connecting arrow from top panel state to bottom-left model detail
# ===========================================================================
ax.annotate("", xy=(3.90, 4.22), xytext=(3.90, 4.38),
            arrowprops=dict(arrowstyle="-|>", color="#AAAABC", lw=1.0,
                            mutation_scale=7), zorder=2)
ax.text(3.90, 4.30, "detail below",
        ha="center", va="center", fontsize=5.8, color="#AAAABC")


# ===========================================================================
# Legend (top-right corner of figure)
# ===========================================================================
LX, LY0 = 0.45, 4.08
ITEMS = [
    (C_SESSION,  "Session input (external)"),
    (C_ENCODER,  "SessionEncoder (MiniLM + VAD)"),
    (C_MODEL,    "TransitionModel $f_θ$"),
    (C_STATE,    "Latent state vector $s_t$"),
    (C_LOSS,     "VICReg loss (training only)"),
    (C_PROBE,    "Linear Probe → (V, A)"),
    (C_DECODER,  "State Decoder → Q1–Q4"),
]
# Draw as a small vertical legend inside the top panel area (far left)
# Actually place at far right of top panel
LX2 = 11.80
LY2 = FH - 1.25
for i, (color, label) in enumerate(ITEMS):
    y = LY2 - i * 0.44
    ax.add_patch(FancyBboxPatch(
        (LX2, y - 0.13), 0.28, 0.26,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=_darken(color, 0.7), lw=0.8, zorder=6,
    ))
    ax.text(LX2 + 0.38, y, label,
            va="center", fontsize=6.6, color="#333344", zorder=6)


# ===========================================================================
# Save
# ===========================================================================
fig.tight_layout(pad=0.4)

png_path = OUT_DIR / "fig0_architecture.png"
pdf_path = OUT_DIR / "fig0_architecture.pdf"
fig.savefig(str(png_path), dpi=200, bbox_inches="tight", facecolor=C_BG)
fig.savefig(str(pdf_path),           bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"Saved  PNG: {png_path}  ({png_path.stat().st_size // 1024} KB)")
print(f"Saved  PDF: {pdf_path}")
