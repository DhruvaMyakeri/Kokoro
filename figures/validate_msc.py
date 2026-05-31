"""
Out-of-distribution validation: nayohan/multi_session_chat

Tests the trained Kokoro transition model on real multi-session conversations
it has NEVER seen during training.  The synthetic training data used arc-
templated EmpatheticDialogues sequences; this dataset is entirely held-out
real-world multi-session chat with no arc labelling.

Two core questions:
  1. Non-collapse check — do state vectors differ across real sessions, or
     does the model collapse all real dialogue to the same point?
  2. Coherence check — do the 20 sampled sequences trace structured paths
     in PCA space, or random noise?

Comparison metric: mean within-sequence cosine distance on MSC vs a
matched sample of synthetic training trajectories run through the same model.

Execution order is strict (same rule as visualize_step3.py):
  Phase 1 — load ML models
  Phase 2 — load data
  Phase 3 — parse MSC + batch-encode all sessions
  Phase 4 — run transition model, compute metrics, PCA
  Phase 5 — import matplotlib, draw figure

Run from project root:
    python figures/validate_msc.py

Saves:
    figures/fig_msc_validation.png  (180 dpi)
    figures/fig_msc_validation.pdf  (vector)
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
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR  = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "transition_v1_10k.pt"
SYNTH_DATA   = PROJECT_ROOT / "data" / "trajectories_10k.json"
N_PLOT_SEQS  = 20
SEED         = 42

# ===========================================================================
# PHASE 1 — Load ML models
# ===========================================================================

logger.info("Phase 1: loading ML models")

from kokoro.encoder import SessionEncoder
from kokoro.transition import TransitionModel

encoder = SessionEncoder(device="cpu")
_ = encoder.model   # trigger lazy load now, before any data loading

if not CHECKPOINT.exists():
    sys.exit(f"Checkpoint not found: {CHECKPOINT}")

ckpt   = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
cfg    = ckpt.get("model_config", {"state_dim": 384, "hidden_dim": 512})
model  = TransitionModel(**cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
device    = torch.device("cpu")
STATE_DIM = cfg.get("state_dim", 384)
logger.info(f"  TransitionModel loaded. Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# ===========================================================================
# PHASE 2 — Load data
# ===========================================================================

logger.info("Phase 2: loading data")

# --- nayohan/multi_session_chat (train split) ---
logger.info("  Loading nayohan/multi_session_chat ...")
try:
    from datasets import load_dataset
    msc_ds = load_dataset("nayohan/multi_session_chat", split="train")
    logger.info(f"  {len(msc_ds):,} rows, columns: {msc_ds.column_names}")
    first = msc_ds[0]
    for k, v in first.items():
        logger.info(f"    [{k}] ({type(v).__name__}) {str(v)[:140]}")
except Exception as exc:
    sys.exit(f"Failed to load nayohan/multi_session_chat: {exc}")

# --- Synthetic trajectories (for comparison baseline) ---
if SYNTH_DATA.exists():
    logger.info(f"  Loading synthetic trajectories from {SYNTH_DATA.name}...")
    with open(SYNTH_DATA) as fh:
        synth_all = json.load(fh)
    logger.info(f"  {len(synth_all):,} synthetic trajectories available")
else:
    logger.warning(f"  {SYNTH_DATA} not found — will skip synthetic comparison")
    synth_all = []

# ===========================================================================
# PHASE 3 — Parse MSC into multi-session sequences and batch-encode
# ===========================================================================

logger.info("Phase 3: parsing MSC and encoding sessions")

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _extract_turns_from_row(row: dict, cols: list[str]) -> list[dict]:
    """
    Try every known column layout for nayohan/multi_session_chat.
    Returns a list of {"role": "user"|"assistant", "content": str}.
    speaker1 → role "user", speaker2 → role "assistant".
    """
    turns: list[dict] = []

    def _add(speaker_hint: str, text: str) -> None:
        text = str(text).strip()
        if not text:
            return
        # Handles: "speaker1", "Speaker 1", "1", "spk1", "user", etc.
        role = "user" if "1" in str(speaker_hint) else "assistant"
        turns.append({"role": role, "content": text})

    # Format 0: "dialogue" + "speaker" are parallel lists  (nayohan/multi_session_chat)
    # dialogue = ["Hi how are you?", "I'm good!", ...], speaker = ["Speaker 1", "Speaker 2", ...]
    if "dialogue" in cols and "speaker" in cols:
        dialogues = row.get("dialogue", [])
        speakers  = row.get("speaker",  [])
        if isinstance(dialogues, list) and isinstance(speakers, list):
            for text, spk in zip(dialogues, speakers):
                _add(spk, text)
            if turns:
                return turns

    # Format A: list column where each item is a dict with speaker/text keys,
    # or a list of plain strings (all treated as speaker1 / user).
    for col in ("utterances", "conversation", "dialog", "dialogues", "dialogue", "turns"):
        if col not in cols:
            continue
        raw = row[col]
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            if isinstance(item, dict):
                speaker = str(item.get("speaker",
                              item.get("role",
                              item.get("spk", "speaker1"))))
                text    = str(item.get("text",
                             item.get("content",
                             item.get("utterance",
                             item.get("utt", "")))))
                _add(speaker, text)
            elif isinstance(item, str):
                _add("speaker1", item)
        if turns:
            return turns

    # Format B: flat speaker1 / speaker2 text columns (one utterance per row)
    for sp1_col in ("speaker1", "speaker1_utterance", "utt1"):
        for sp2_col in ("speaker2", "speaker2_utterance", "utt2"):
            if sp1_col in cols or sp2_col in cols:
                t1 = str(row.get(sp1_col, "")).strip()
                t2 = str(row.get(sp2_col, "")).strip()
                if t1:
                    _add("speaker1", t1)
                if t2:
                    _add("speaker2", t2)
                if turns:
                    return turns

    # Format C: fallback — first non-metadata list column with string items
    skip = {"dialoug_id", "dialog_id", "session_id", "id", "dataset",
            "persona1", "persona2"}   # skip persona/metadata columns
    for col in cols:
        if col in skip:
            continue
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            _add("speaker1", val)
            return turns
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    _add("speaker1", item)
            if turns:
                return turns

    return turns


def parse_msc(dataset) -> dict[str, list[dict]]:
    """
    Group dataset rows into per-dialogue session lists.

    Returns:
        dialoug_id -> sorted list of {"session_id": int, "turns": [...]}
    """
    cols       = dataset.column_names
    sequences: dict[str, list[dict]] = defaultdict(list)

    for row in dataset:
        dlg_id  = str(row.get("dialoug_id",
                      row.get("dialog_id",
                      row.get("dialogue_id", "unknown"))))
        sess_id = int(row.get("session_id", 0))
        turns   = _extract_turns_from_row(row, cols)
        if turns:
            sequences[dlg_id].append({"session_id": sess_id, "turns": turns})

    # Sort each dialogue's sessions by session_id
    for dlg_id in sequences:
        sequences[dlg_id].sort(key=lambda s: s["session_id"])

    return dict(sequences)


logger.info("  Parsing MSC rows into dialogue sequences...")
sequences = parse_msc(msc_ds)
logger.info(f"  {len(sequences):,} unique dialogue pairs found")

# Keep only multi-session sequences (need >= 2 sessions for consecutive distance)
multi_seqs = {k: v for k, v in sequences.items() if len(v) >= 2}
logger.info(f"  {len(multi_seqs):,} sequences with >= 2 sessions")

if not multi_seqs:
    logger.error(
        "No multi-session sequences found.  Check the dataset column layout — "
        "the parser printed column names above."
    )
    sys.exit(1)

sess_lengths = [len(v) for v in multi_seqs.values()]
logger.info(
    f"  Session counts: min={min(sess_lengths)}, max={max(sess_lengths)}, "
    f"mean={np.mean(sess_lengths):.2f}, median={int(np.median(sess_lengths))}"
)

# ---------------------------------------------------------------------------
# Collect unique sessions for batch encoding
# ---------------------------------------------------------------------------

unique_msc_sessions: dict[str, list[dict]] = {}   # key = "dlg_id::sess_id"
for dlg_id, sess_list in multi_seqs.items():
    for sess in sess_list:
        key = f"{dlg_id}::{sess['session_id']}"
        unique_msc_sessions[key] = sess["turns"]

logger.info(f"  {len(unique_msc_sessions):,} unique MSC sessions to encode")

# ---------------------------------------------------------------------------
# Batch encoder (same chunked approach as training/validate.py)
# ---------------------------------------------------------------------------

def batch_encode(
    sessions: dict[str, list[dict]],
    chunk_size: int = 1500,
) -> dict[str, torch.Tensor]:
    """
    sessions: key -> list of {"role", "content"} turn dicts
    Returns:  key -> (STATE_DIM,) float32 tensor (NOT L2-normalised)
    """
    from kokoro.encoder import decode_parlai_artifacts

    all_keys  = list(sessions.keys())
    n         = len(all_keys)
    cache: dict[str, torch.Tensor] = {}
    t0        = time.perf_counter()

    for start in range(0, n, chunk_size):
        chunk_keys   = all_keys[start: start + chunk_size]
        texts_all, slices, weights_all = [], [], []

        for key in chunk_keys:
            turns   = sessions[key]
            texts   = []
            weights = []
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
        for i, key in enumerate(chunk_keys):
            s, e = slices[i]
            w    = np.array(weights_all[i], dtype=np.float32)
            w   /= w.sum()
            pooled = (embs[s:e] * w[:, np.newaxis]).sum(axis=0).astype(np.float32)
            cache[key] = torch.tensor(pooled, device=device)

        done = min(start + chunk_size, n)
        if (start // chunk_size) % 5 == 0 or done == n:
            elapsed = time.perf_counter() - t0
            logger.info(f"    {done:,}/{n:,} sessions ({done/n*100:.0f}%)  [{elapsed:.1f}s]")

    logger.info(f"  Encoding complete in {time.perf_counter()-t0:.1f}s")
    return cache


logger.info("  Encoding MSC sessions...")
msc_cache = batch_encode(unique_msc_sessions)

# ===========================================================================
# PHASE 4 — Run transition model, compute metrics, PCA
# ===========================================================================

logger.info("Phase 4: running transition model and computing metrics")

# ---------------------------------------------------------------------------
# Run model on all MSC sequences
# ---------------------------------------------------------------------------

msc_states: dict[str, list[np.ndarray]] = {}  # dlg_id -> session-ordered states
all_msc_vecs: list[np.ndarray] = []            # flat list for PCA

model.eval()
with torch.no_grad():
    for dlg_id, sess_list in multi_seqs.items():
        state  = TransitionModel.initial_state(device=device)   # (384,)
        states = []
        for sess in sess_list:
            key   = f"{dlg_id}::{sess['session_id']}"
            emb   = msc_cache[key]
            state = model(state, emb)          # (384,)
            states.append(state.cpu().numpy())
        msc_states[dlg_id] = states
        all_msc_vecs.extend(states)

all_msc_arr = np.stack(all_msc_vecs, axis=0)  # (N_states, 384)
logger.info(f"  State matrix: {all_msc_arr.shape}")

# ---------------------------------------------------------------------------
# Check 1: Non-collapse  (pairwise cosine distance on a sample)
# ---------------------------------------------------------------------------

n_vecs      = len(all_msc_vecs)
sample_n    = min(n_vecs, 2000)
rng_np      = np.random.default_rng(SEED)
sample_idx  = rng_np.choice(n_vecs, size=sample_n, replace=False)
sample_arr  = all_msc_arr[sample_idx]                 # (S, 384)

# All model outputs are L2-normalised → cos_dist = 1 - dot_product
dot_mat     = sample_arr @ sample_arr.T               # (S, S)
cos_dist_mat = 1.0 - dot_mat
np.fill_diagonal(cos_dist_mat, 0.0)
triu_i, triu_j = np.triu_indices(sample_n, k=1)
global_dists   = cos_dist_mat[triu_i, triu_j]

mean_global = float(np.mean(global_dists))
std_global  = float(np.std(global_dists))
state_norms = float(np.linalg.norm(all_msc_arr, axis=1).mean())
logger.info(
    f"  Global pairwise cosine dist: mean={mean_global:.4f}, std={std_global:.4f}"
)

# ---------------------------------------------------------------------------
# Check 2: Within-sequence consecutive cosine distance  (MSC)
# ---------------------------------------------------------------------------

within_msc: list[float] = []
for dlg_id, states in msc_states.items():
    if len(states) < 2:
        continue
    dists = [
        float(1.0 - np.dot(states[i], states[i + 1]))
        for i in range(len(states) - 1)
    ]
    within_msc.append(float(np.mean(dists)))

mean_within_msc = float(np.mean(within_msc))
std_within_msc  = float(np.std(within_msc))
logger.info(
    f"  MSC within-seq cosine dist: mean={mean_within_msc:.4f}, std={std_within_msc:.4f}"
)

# ---------------------------------------------------------------------------
# Within-sequence consecutive cosine distance  (Synthetic, for comparison)
# ---------------------------------------------------------------------------

within_synth: list[float] = []
synth_cache_built = False

if synth_all:
    rng_synth   = random.Random(SEED)
    synth_sample = rng_synth.sample(synth_all, min(len(synth_all), 2000))

    # Collect unique synthetic sessions
    unique_synth: dict[str, list[dict]] = {}
    for traj in synth_sample:
        for sess in traj["sessions"]:
            cid = sess["conv_id"]
            if cid not in unique_synth:
                unique_synth[cid] = sess["turns"]

    logger.info(f"  Encoding {len(unique_synth):,} synthetic sessions for comparison...")
    synth_cache = batch_encode(unique_synth)
    synth_cache_built = True

    with torch.no_grad():
        for traj in synth_sample:
            state  = TransitionModel.initial_state(device=device)
            states = []
            for sess in traj["sessions"]:
                emb   = synth_cache[sess["conv_id"]]
                state = model(state, emb)
                states.append(state.cpu().numpy())
            if len(states) < 2:
                continue
            dists = [
                float(1.0 - np.dot(states[i], states[i + 1]))
                for i in range(len(states) - 1)
            ]
            within_synth.append(float(np.mean(dists)))

mean_within_synth = float(np.mean(within_synth)) if within_synth else float("nan")
std_within_synth  = float(np.std(within_synth))  if within_synth else float("nan")
logger.info(
    f"  Synthetic within-seq cosine dist: mean={mean_within_synth:.4f}, "
    f"std={std_within_synth:.4f}"
)

# ---------------------------------------------------------------------------
# PCA on all MSC state vectors
# ---------------------------------------------------------------------------

logger.info("  Computing PCA on MSC state vectors...")
from sklearn.decomposition import PCA

pca_2d   = PCA(n_components=2, random_state=SEED)
coords   = pca_2d.fit_transform(all_msc_arr)   # (N, 2)
var_exp  = pca_2d.explained_variance_ratio_ * 100
logger.info(f"  PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")

# Map each dialogue's states to 2-D coordinates (in same flat order)
dlg_id_order = list(msc_states.keys())
seq_coords: dict[str, np.ndarray] = {}
offset = 0
for dlg_id in dlg_id_order:
    n = len(msc_states[dlg_id])
    seq_coords[dlg_id] = coords[offset: offset + n]
    offset += n

# ---------------------------------------------------------------------------
# Select 20 sequences for path plot
# Prefer sequences with >= 3 sessions (more interesting paths)
# ---------------------------------------------------------------------------

rng_plot  = random.Random(SEED)
long_ids  = [k for k in dlg_id_order if len(msc_states[k]) >= 3]
short_ids = [k for k in dlg_id_order if len(msc_states[k]) == 2]
plot_ids  = rng_plot.sample(long_ids, min(N_PLOT_SEQS, len(long_ids)))
if len(plot_ids) < N_PLOT_SEQS and short_ids:
    needed = N_PLOT_SEQS - len(plot_ids)
    plot_ids.extend(rng_plot.sample(short_ids, min(needed, len(short_ids))))
plot_ids = plot_ids[:N_PLOT_SEQS]
logger.info(f"  Selected {len(plot_ids)} sequences for path plot")

# ===========================================================================
# PHASE 5 — matplotlib  (ALWAYS LAST — never import earlier)
# ===========================================================================

logger.info("Phase 5: drawing figure")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor("white")

# --------------------------------------------------------------------------
# Left panel: PCA scatter + 20 sequence paths
# --------------------------------------------------------------------------

ax = ax_left
ax.set_facecolor("#F8F8F8")

# Background: all state vectors  (light grey, small)
ax.scatter(
    coords[:, 0], coords[:, 1],
    c="#D5D5D5", s=5, alpha=0.4, linewidths=0, zorder=1,
)

# Colormap for the 20 trajectories
try:
    _cmap = plt.colormaps["tab20"]
except AttributeError:                      # matplotlib < 3.5 fallback
    _cmap = cm.get_cmap("tab20")

for idx, dlg_id in enumerate(plot_ids):
    path  = seq_coords[dlg_id]          # (n_sess, 2)
    color = _cmap(idx / N_PLOT_SEQS)

    # Connecting line
    ax.plot(
        path[:, 0], path[:, 1],
        color=color, linewidth=1.8, alpha=0.85, zorder=2,
        solid_capstyle="round",
    )

    # Arrows at each step to show direction (mid-segment arrows)
    for step in range(len(path) - 1):
        dx = path[step + 1, 0] - path[step, 0]
        dy = path[step + 1, 1] - path[step, 1]
        mid_x = (path[step, 0] + path[step + 1, 0]) / 2
        mid_y = (path[step, 1] + path[step + 1, 1]) / 2
        ax.annotate(
            "", xy=(mid_x + dx * 0.001, mid_y + dy * 0.001),
            xytext=(mid_x - dx * 0.001, mid_y - dy * 0.001),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=0.8,
                mutation_scale=8,
            ),
            zorder=3,
        )

    # Session markers
    # Start: large filled circle
    ax.scatter(
        [path[0, 0]], [path[0, 1]],
        c=[color], s=70, zorder=5,
        edgecolors="white", linewidths=0.8,
    )
    # Intermediate sessions: medium circles
    if len(path) > 2:
        ax.scatter(
            path[1:-1, 0], path[1:-1, 1],
            c=[color] * (len(path) - 2), s=30, zorder=5,
            edgecolors="white", linewidths=0.6,
        )
    # End: star marker
    ax.scatter(
        [path[-1, 0]], [path[-1, 1]],
        c=[color], s=130, marker="*", zorder=6,
        edgecolors="white", linewidths=0.8,
    )

    # Small index label at start point
    ax.text(
        path[0, 0], path[0, 1], str(idx + 1),
        fontsize=4.5, ha="center", va="center",
        color="white", fontweight="bold", zorder=7,
    )

ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var. exp.)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var. exp.)", fontsize=11)
ax.set_title(
    f"Real MSC State Trajectories in PCA Space\n"
    f"({len(plot_ids)} sampled sequences, n={n_vecs:,} total state vectors)",
    fontsize=11,
)
ax.grid(True, alpha=0.3, linewidth=0.5)

legend_elements = [
    Line2D([0], [0], marker="o", color="grey", markersize=8,
           label="Session start", markeredgecolor="white", linewidth=0),
    Line2D([0], [0], marker="o", color="grey", markersize=5,
           label="Intermediate session", markeredgecolor="white", linewidth=0),
    Line2D([0], [0], marker="*", color="grey", markersize=12,
           label="Final session", markeredgecolor="white", linewidth=0),
    Line2D([0], [0], color="grey", linewidth=1.8,
           label="Session progression (arrow = direction)"),
    Line2D([0], [0], marker="o", color="#D5D5D5", markersize=5,
           label="All MSC state vectors (background)", linewidth=0),
]
ax.legend(
    handles=legend_elements, fontsize=7.5, loc="upper right", framealpha=0.85,
)

# --------------------------------------------------------------------------
# Right panel: within-sequence cosine distance distributions
# --------------------------------------------------------------------------

ax2 = ax_right
ax2.set_facecolor("#F8F8F8")

all_dists = within_msc + (within_synth if within_synth else [])
if all_dists:
    bin_max = max(all_dists) * 1.05
    bins    = np.linspace(0, bin_max, 45)
else:
    bins = np.linspace(0, 0.2, 45)

ax2.hist(
    within_msc, bins=bins, density=True,
    color="#2980B9", alpha=0.72, edgecolor="white", linewidth=0.4,
    label=f"MSC real  (n={len(within_msc):,})  mean={mean_within_msc:.4f}",
)
if within_synth:
    ax2.hist(
        within_synth, bins=bins, density=True,
        color="#E67E22", alpha=0.72, edgecolor="white", linewidth=0.4,
        label=f"Synthetic  (n={len(within_synth):,})  mean={mean_within_synth:.4f}",
    )

ax2.axvline(
    mean_within_msc, color="#1A5276", linewidth=2.0,
    linestyle="--", label=f"MSC mean = {mean_within_msc:.4f}",
)
if within_synth:
    ax2.axvline(
        mean_within_synth, color="#935116", linewidth=2.0,
        linestyle="--", label=f"Synth mean = {mean_within_synth:.4f}",
    )

ax2.set_xlabel(
    "Mean consecutive cosine distance\n(within one multi-session sequence)",
    fontsize=11,
)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title(
    "State Movement Per Session Step\nReal MSC vs Synthetic Training Data",
    fontsize=11,
)
ax2.legend(fontsize=8.5, framealpha=0.85)
ax2.grid(True, alpha=0.3, linewidth=0.5)

# --------------------------------------------------------------------------
# Figure-level title and save
# --------------------------------------------------------------------------

plt.suptitle(
    "Out-of-Distribution Validation: Kokoro Transition Model on Multi-Session Chat\n"
    "(Dataset never seen during training)",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()

out_png = FIGURES_DIR / "fig_msc_validation.png"
out_pdf = FIGURES_DIR / "fig_msc_validation.pdf"
plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
logger.info(f"  Saved {out_png}")
logger.info(f"  Saved {out_pdf}")
plt.close(fig)

# ===========================================================================
# Summary table
# ===========================================================================

W = 72

def hline(char: str = "-") -> None:
    print(char * W)


print()
hline("=")
print("  OOD Validation: Kokoro Transition Model on nayohan/multi_session_chat")
hline("=")
print(f"  {'Dataset':<38} nayohan/multi_session_chat (train)")
print(f"  {'Model checkpoint':<38} {CHECKPOINT.name}")
print(f"  {'Model epoch':<38} {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
hline()

print("  DATASET STATISTICS")
print(f"  {'Total rows in train split':<38} {len(msc_ds):>10,}")
print(f"  {'Unique dialogue pairs':<38} {len(sequences):>10,}")
print(f"  {'Pairs with >= 2 sessions':<38} {len(multi_seqs):>10,}")
print(f"  {'Total sessions encoded':<38} {len(unique_msc_sessions):>10,}")
print(f"  {'Total state vectors produced':<38} {n_vecs:>10,}")
hline()

print("  SESSION COUNT DISTRIBUTION  (sessions per dialogue pair)")
for length in sorted(set(sess_lengths)):
    count = sess_lengths.count(length)
    pct   = count / len(sess_lengths) * 100
    bar   = "#" * min(int(pct / 2), 30)
    print(f"    {length} sessions: {count:>6,} ({pct:5.1f}%)  {bar}")
hline()

print("  STATE VECTOR HEALTH CHECK")
print(f"  {'Mean L2 norm (expect ~1.0)':<38} {state_norms:>10.4f}")
print(f"  {'Global pairwise cosine dist':<38}")
print(f"    {'mean':<36} {mean_global:>10.4f}")
print(f"    {'std':<36} {std_global:>10.4f}")
print(f"    {'min':<36} {float(global_dists.min()):>10.4f}")
print(f"    {'max':<36} {float(global_dists.max()):>10.4f}")
collapsed_str = "NO  (global dist > 0.005)" if mean_global > 0.005 else "YES — investigate"
print(f"  {'Collapsed?':<38} {collapsed_str:>20}")
hline()

print("  WITHIN-SEQUENCE COSINE DISTANCE  (consecutive session steps)")
print(f"  {'Method':<34} {'Mean':>8}  {'Std':>8}  {'n_seqs':>8}")
hline()
print(f"  {'MSC (real, OOD)':<34} {mean_within_msc:>8.4f}  {std_within_msc:>8.4f}  {len(within_msc):>8,}")
if within_synth:
    print(f"  {'Synthetic (training data)':<34} {mean_within_synth:>8.4f}  {std_within_synth:>8.4f}  {len(within_synth):>8,}")
    ratio = mean_within_msc / mean_within_synth if mean_within_synth > 0 else float("nan")
    print(f"  {'Ratio MSC / Synthetic':<34} {ratio:>8.4f}")
hline()

print("  PCA  (all MSC state vectors, 2 components)")
print(f"  {'PC1 variance explained':<38} {var_exp[0]:>9.1f}%")
print(f"  {'PC2 variance explained':<38} {var_exp[1]:>9.1f}%")
print(f"  {'Cumulative PC1+PC2':<38} {var_exp[0]+var_exp[1]:>9.1f}%")
hline()

print("  INTERPRETATION")
if mean_global > 0.01:
    print(f"  [PASS] State vectors are non-collapsed.")
    print(f"         Global cosine dist = {mean_global:.4f} (well above zero).")
    print(f"         The model maps real MSC sessions to distinct points in state space.")
else:
    print(f"  [WARN] State vectors may be collapsed (global dist = {mean_global:.4f}).")
    print(f"         The model may be ignoring content from real (non-synthetic) dialogue.")

if not np.isnan(mean_within_synth):
    delta = mean_within_msc - mean_within_synth
    if delta > 0.01:
        print(f"  Real MSC shows MORE movement per step ({mean_within_msc:.4f}) than")
        print(f"  synthetic training data ({mean_within_synth:.4f}). Natural conversations")
        print(f"  may contain stronger emotional transitions than synthetic arc sessions.")
    elif delta < -0.01:
        print(f"  Real MSC shows LESS movement per step ({mean_within_msc:.4f}) than")
        print(f"  synthetic training data ({mean_within_synth:.4f}). Real conversations")
        print(f"  may be more emotionally stable, or the model is less sensitive to")
        print(f"  natural (non-synthetic) dialogue patterns.")
    else:
        print(f"  MSC movement ({mean_within_msc:.4f}) is in a similar range to")
        print(f"  synthetic training data ({mean_within_synth:.4f}). The model")
        print(f"  generalises to real data without systematic over- or under-sensitivity.")

print(f"  Figure: {out_png}")
hline("=")
