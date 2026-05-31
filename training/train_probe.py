"""
Linear probe: predict valence and arousal from transition model state vectors.

A linear probe is a single Linear(384 → 2) layer with no hidden layers and no
activation.  If this simple probe achieves high Pearson r on held-out data, it
means the circumplex coordinates (valence, arousal) are linearly decodable from
the transition model's state vectors — the representation is not just arc-
discriminative (shown in Table 1) but also metrically aligned to the Russell
circumplex space.

Data construction:
  For each trajectory, sessions are run sequentially through the transition model.
  After processing session t, the resulting state vector is paired with the ground
  truth (valence_t, arousal_t) for that session.  This yields one (state, v, a)
  sample per session per trajectory — roughly 50,000 samples total from 10k trajs.

Train/val split is at the trajectory level (not the sample level) to prevent
leakage: all samples from a given trajectory appear in exactly one split.

Saves:
  checkpoints/valence_arousal_probe.pt

Run from project root:
    python -m training.train_probe
    python -m training.train_probe --help
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linear probe definition
# ---------------------------------------------------------------------------

class ValenceArousalProbe(nn.Module):
    """
    Single linear layer: state_vector (384) → [valence, arousal].

    No hidden layers, no activation, no bias unless --bias flag is set.
    By design this is the simplest possible probe — anything it learns is
    genuinely linearly decodable from the state representation.
    """

    def __init__(self, state_dim: int = 384) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., state_dim) → (..., 2)  where [:, 0]=valence, [:, 1]=arousal"""
        return self.linear(x)


# ---------------------------------------------------------------------------
# Embedding precomputation  (identical pattern to validate.py)
# ---------------------------------------------------------------------------

def precompute_embeddings(
    trajectories: list[dict],
    encoder,
    device: torch.device,
    chunk_size: int = 1500,
) -> dict[str, torch.Tensor]:
    """
    Encode all unique sessions in the trajectory set.
    Returns conv_id → (state_dim,) float32 tensor (NOT L2-normalised).
    """
    from kokoro.encoder import decode_parlai_artifacts

    unique: dict[str, list] = {}
    for traj in trajectories:
        for sess in traj["sessions"]:
            if sess["conv_id"] not in unique:
                unique[sess["conv_id"]] = sess["turns"]

    all_ids = list(unique.keys())
    n = len(all_ids)
    logger.info(f"  Encoding {n:,} unique sessions...")
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

        done = min(start + chunk_size, n)
        if (start // chunk_size) % 5 == 0 or done == n:
            logger.info(
                f"    {done:,}/{n:,} ({done/n*100:.0f}%)  [{time.perf_counter()-t0:.1f}s]"
            )

    logger.info(f"  Embedding done in {time.perf_counter()-t0:.1f}s")
    return cache


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_probe_dataset(
    trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
    transition_model: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run each trajectory through the transition model sequentially.
    After each session, collect (state_vector, valence, arousal).

    Returns:
        states:  (N_samples, 384) float32 — L2-normalised state vectors
        targets: (N_samples, 2)   float32 — [valence, arousal] in [-1, 1]
    """
    from kokoro.transition import TransitionModel

    all_states:  list[np.ndarray] = []
    all_targets: list[tuple[float, float]] = []

    transition_model.eval()
    with torch.no_grad():
        for traj in trajectories:
            state = TransitionModel.initial_state(device=device)  # (384,)
            for sess in traj["sessions"]:
                emb   = emb_cache[sess["conv_id"]]
                state = transition_model(state, emb)              # (384,)
                all_states.append(state.cpu().numpy())
                all_targets.append((float(sess["valence"]), float(sess["arousal"])))

    states_arr  = np.stack(all_states,  axis=0).astype(np.float32)   # (N, 384)
    targets_arr = np.array(all_targets, dtype=np.float32)             # (N, 2)

    return (
        torch.from_numpy(states_arr),
        torch.from_numpy(targets_arr),
    )


# ---------------------------------------------------------------------------
# Pearson r (per-output)
# ---------------------------------------------------------------------------

def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson r between two 1-D arrays."""
    if pred.std() < 1e-9 or target.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_probe(
    probe: ValenceArousalProbe,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    epochs:   int   = 100,
    lr:       float = 1e-3,
    batch_size: int = 512,
    log_every: int  = 10,
    device:   torch.device = torch.device("cpu"),
) -> dict:
    """
    Train the linear probe and return a history dict.

    Loss: MSE on [valence, arousal] simultaneously.
    Optimiser: AdamW with no weight decay (probe is tiny — regularisation
    is provided by the linear constraint itself, not explicit decay).
    """
    probe.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val,   y_val   = X_val.to(device),   y_val.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = X_train.shape[0]
    history: dict[str, list] = {
        "train_mse": [], "val_mse": [], "val_r_valence": [], "val_r_arousal": [],
    }

    probe.train()
    for epoch in range(1, epochs + 1):
        # Mini-batch SGD over training set
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            idx    = perm[start: start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = probe(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_mse = epoch_loss / max(n_batches, 1)

        # Validation
        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_mse  = float(F.mse_loss(
                torch.from_numpy(val_pred),
                torch.from_numpy(val_true),
            ).item())
            r_val    = pearson_r(val_pred[:, 0], val_true[:, 0])
            r_aro    = pearson_r(val_pred[:, 1], val_true[:, 1])

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_r_valence"].append(r_val)
        history["val_r_arousal"].append(r_aro)
        probe.train()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            logger.info(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"train_mse={train_mse:.5f}  "
                f"val_mse={val_mse:.5f}  "
                f"r_val={r_val:.4f}  "
                f"r_aro={r_aro:.4f}"
            )

    probe.eval()
    return history


# ---------------------------------------------------------------------------
# Final evaluation table
# ---------------------------------------------------------------------------

def print_results(
    probe: ValenceArousalProbe,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    history: dict,
    n_train_samples: int,
    n_val_samples:   int,
    device: torch.device,
) -> None:
    W = 66

    def hline(char: str = "-") -> None:
        print(char * W)

    probe.eval()
    with torch.no_grad():
        pred = probe(X_val.to(device)).cpu().numpy()
    true = y_val.numpy()

    v_pred, v_true = pred[:, 0], true[:, 0]
    a_pred, a_true = pred[:, 1], true[:, 1]

    r_val      = pearson_r(v_pred, v_true)
    r_aro      = pearson_r(a_pred, a_true)
    mse_val    = float(np.mean((v_pred - v_true) ** 2))
    mse_aro    = float(np.mean((a_pred - a_true) ** 2))
    mse_total  = float(np.mean((pred - true) ** 2))
    mae_val    = float(np.mean(np.abs(v_pred - v_true)))
    mae_aro    = float(np.mean(np.abs(a_pred - a_true)))

    print()
    hline("=")
    print("  Linear Probe: Valence + Arousal from State Vectors")
    hline("=")
    print(f"  {'Architecture':<36} Linear(384 -> 2), no hidden layers")
    print(f"  {'Training samples':<36} {n_train_samples:>10,}")
    print(f"  {'Validation samples':<36} {n_val_samples:>10,}")
    print(f"  {'Best val MSE (epoch)':<36} "
          f"{min(history['val_mse']):>10.5f}  "
          f"(epoch {history['val_mse'].index(min(history['val_mse'])) + 1})")
    hline()
    print(f"  {'Metric':<20} {'Valence':>12}  {'Arousal':>12}")
    hline()
    print(f"  {'Pearson r':<20} {r_val:>12.4f}  {r_aro:>12.4f}")
    print(f"  {'MSE':<20} {mse_val:>12.5f}  {mse_aro:>12.5f}")
    print(f"  {'MAE':<20} {mae_val:>12.4f}  {mae_aro:>12.4f}")
    hline()
    print(f"  {'Overall MSE':<36} {mse_total:>12.5f}")
    hline()

    # Interpretation
    print("  INTERPRETATION")
    for name, r in [("Valence", r_val), ("Arousal", r_aro)]:
        if np.isnan(r):
            print(f"  {name}: r = undefined (constant prediction — investigate)")
        elif r > 0.7:
            print(f"  {name}: r = {r:.4f}  — strong linear decodability")
        elif r > 0.4:
            print(f"  {name}: r = {r:.4f}  — moderate linear decodability")
        elif r > 0.1:
            print(f"  {name}: r = {r:.4f}  — weak linear decodability")
        else:
            print(f"  {name}: r = {r:.4f}  — essentially no linear decodability")

    # Circumplex range note
    print()
    print("  Ground truth range (valence):  "
          f"[{v_true.min():.2f}, {v_true.max():.2f}]")
    print("  Ground truth range (arousal):  "
          f"[{a_true.min():.2f}, {a_true.max():.2f}]")
    print("  Prediction range  (valence):   "
          f"[{v_pred.min():.2f}, {v_pred.max():.2f}]")
    print("  Prediction range  (arousal):   "
          f"[{a_pred.min():.2f}, {a_pred.max():.2f}]")
    hline("=")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(
    probe: ValenceArousalProbe,
    all_trajectories: list[dict],
    emb_cache: dict[str, torch.Tensor],
    transition_model: nn.Module,
    device: torch.device,
) -> None:
    """
    Given one stable_positive and one stable_negative trajectory, run them
    through the transition model and check that the probe outputs higher
    valence for the positive arc.

    This is a directional sanity check, not a performance metric.
    Uses the first matching trajectory of each arc type found in the full set.
    """
    from kokoro.transition import TransitionModel

    W = 66

    def hline(char: str = "-") -> None:
        print(char * W)

    print()
    hline("=")
    print("  SANITY CHECK: stable_positive vs stable_negative")
    hline("=")

    target_arcs = ["stable_positive", "stable_negative"]
    examples: dict[str, list[dict]] = {}
    for traj in all_trajectories:
        if traj["arc_name"] in target_arcs and traj["arc_name"] not in examples:
            examples[traj["arc_name"]] = traj["sessions"]
        if len(examples) == len(target_arcs):
            break

    for arc_name in target_arcs:
        if arc_name not in examples:
            print(f"  [{arc_name}] — no trajectory found in dataset, skipping")
            continue

        sessions = examples[arc_name]
        transition_model.eval()
        probe.eval()

        with torch.no_grad():
            state = TransitionModel.initial_state(device=device)
            for sess in sessions:
                emb   = emb_cache[sess["conv_id"]]
                state = transition_model(state, emb)
            pred_va = probe(state).cpu().numpy()

        gt_v = float(np.mean([s["valence"] for s in sessions]))
        gt_a = float(np.mean([s["arousal"] for s in sessions]))

        print(
            f"  {arc_name:<26}"
            f"  pred_val={pred_va[0]:+.3f}  pred_aro={pred_va[1]:+.3f}"
            f"  (gt_val={gt_v:+.3f}  gt_aro={gt_a:+.3f})"
        )

    hline()

    # Check direction
    if "stable_positive" in examples and "stable_negative" in examples:
        pos_sessions = examples["stable_positive"]
        neg_sessions = examples["stable_negative"]

        transition_model.eval()
        probe.eval()
        with torch.no_grad():
            def get_final_pred(sessions):
                state = TransitionModel.initial_state(device=device)
                for sess in sessions:
                    emb   = emb_cache[sess["conv_id"]]
                    state = transition_model(state, emb)
                return probe(state).cpu().numpy()

            pos_pred = get_final_pred(pos_sessions)
            neg_pred = get_final_pred(neg_sessions)

        passed = pos_pred[0] > neg_pred[0]
        print(
            f"  stable_positive valence ({pos_pred[0]:+.3f}) > "
            f"stable_negative valence ({neg_pred[0]:+.3f}):  "
            f"{'[PASS]' if passed else '[FAIL]'}"
        )
        if not passed:
            print(
                "  NOTE: Probe predicts lower valence for positive arc."
                "  This may indicate the probe hasn't converged or that the"
                "  state representation encodes valence non-linearly."
            )

    hline("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a linear probe to decode valence+arousal from state vectors"
    )
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
        "--out-probe", type=str,
        default=str(Path(__file__).parent.parent / "checkpoints" / "valence_arousal_probe.pt"),
        help="Output path for probe checkpoint",
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--val-split",  type=float, default=0.2,
                        help="Fraction of trajectories for validation")
    parser.add_argument("--log-every",  type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    data_path   = Path(args.data)
    ckpt_path   = Path(args.checkpoint)
    probe_path  = Path(args.out_probe)

    for p in [data_path, ckpt_path]:
        if not p.exists():
            sys.exit(f"File not found: {p}")
    probe_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Load transition model                                                #
    # ------------------------------------------------------------------ #
    logger.info("Loading transition model...")
    from kokoro.transition import TransitionModel

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg   = ckpt.get("model_config", {"state_dim": 384, "hidden_dim": 512})
    model = TransitionModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    state_dim = cfg.get("state_dim", 384)
    logger.info(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, state_dim={state_dim}")

    # ------------------------------------------------------------------ #
    # Load encoder                                                         #
    # ------------------------------------------------------------------ #
    logger.info("Loading session encoder...")
    from kokoro.encoder import SessionEncoder

    encoder = SessionEncoder(device="cpu")
    _ = encoder.model   # trigger lazy load

    # ------------------------------------------------------------------ #
    # Load trajectories                                                    #
    # ------------------------------------------------------------------ #
    logger.info(f"Loading trajectories from {data_path.name}...")
    with open(data_path) as fh:
        all_trajectories = json.load(fh)
    logger.info(f"  {len(all_trajectories):,} trajectories loaded")

    # Train / val split at trajectory level
    rng.shuffle(all_trajectories)
    n_val   = int(len(all_trajectories) * args.val_split)
    n_train = len(all_trajectories) - n_val
    train_trajs = all_trajectories[:n_train]
    val_trajs   = all_trajectories[n_train:]
    logger.info(f"  Split: {n_train:,} train / {n_val:,} val trajectories")

    # ------------------------------------------------------------------ #
    # Precompute embeddings                                                #
    # ------------------------------------------------------------------ #
    logger.info("Precomputing session embeddings...")
    emb_cache = precompute_embeddings(all_trajectories, encoder, device)

    # ------------------------------------------------------------------ #
    # Build probe datasets                                                 #
    # ------------------------------------------------------------------ #
    logger.info("Building probe datasets from state trajectories...")

    logger.info("  Training set...")
    X_train, y_train = build_probe_dataset(train_trajs, emb_cache, model, device)

    logger.info("  Validation set...")
    X_val, y_val     = build_probe_dataset(val_trajs,   emb_cache, model, device)

    logger.info(
        f"  Train: {X_train.shape[0]:,} samples  "
        f"Val: {X_val.shape[0]:,} samples"
    )
    logger.info(
        f"  Valence range: [{y_train[:, 0].min():.2f}, {y_train[:, 0].max():.2f}]  "
        f"Arousal range: [{y_train[:, 1].min():.2f}, {y_train[:, 1].max():.2f}]"
    )

    # ------------------------------------------------------------------ #
    # Train probe                                                          #
    # ------------------------------------------------------------------ #
    logger.info("Training linear probe...")
    probe = ValenceArousalProbe(state_dim=state_dim)
    logger.info(
        f"  Probe parameters: {sum(p.numel() for p in probe.parameters()):,}"
    )

    history = train_probe(
        probe       = probe,
        X_train     = X_train,
        y_train     = y_train,
        X_val       = X_val,
        y_val       = y_val,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        log_every   = args.log_every,
        device      = device,
    )

    # ------------------------------------------------------------------ #
    # Print results                                                        #
    # ------------------------------------------------------------------ #
    print_results(
        probe            = probe,
        X_val            = X_val,
        y_val            = y_val,
        history          = history,
        n_train_samples  = X_train.shape[0],
        n_val_samples    = X_val.shape[0],
        device           = device,
    )

    # ------------------------------------------------------------------ #
    # Save probe                                                           #
    # ------------------------------------------------------------------ #
    logger.info(f"Saving probe to {probe_path}...")
    probe.eval()
    torch.save(
        {
            "model_state_dict":  probe.state_dict(),
            "probe_config":      {"state_dim": state_dim},
            "transition_ckpt":   ckpt_path.name,
            "val_r_valence":     history["val_r_valence"][-1],
            "val_r_arousal":     history["val_r_arousal"][-1],
            "val_mse":           history["val_mse"][-1],
            "best_val_mse":      min(history["val_mse"]),
            "epochs_trained":    args.epochs,
            "history":           history,
        },
        probe_path,
    )
    logger.info(f"  Saved: {probe_path}")

    # ------------------------------------------------------------------ #
    # Sanity check                                                         #
    # ------------------------------------------------------------------ #
    sanity_check(
        probe             = probe,
        all_trajectories  = all_trajectories,
        emb_cache         = emb_cache,
        transition_model  = model,
        device            = device,
    )
