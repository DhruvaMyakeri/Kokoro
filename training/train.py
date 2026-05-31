"""
Transition model training for Kokoro.

Training objective:
  Given a trajectory [S_0, S_1, ..., S_{N-1}], encode each session to get
  embeddings [e_0, ..., e_{N-1}]. Starting from the zero initial state, run
  sessions through the transition model sequentially:

      z_0   = initial_state (zeros)
      z_{t+1} = TransitionModel(z_t, e_t)      for t = 0 ... N-2

  Loss at each step: 1 - cosine_similarity(z_{t+1}, e_{t+1})

  The model learns: "after observing session t, the updated state should point
  toward what session t+1 will feel like." This is a next-session prediction
  objective — the state encodes emotional trajectory, not just the last session.

Loss range:
  [0, 2] — 0 = perfect prediction, 1 = orthogonal (random), 2 = opposite.
  Well-trained models on this dataset should reach ~0.6–0.8 (limited by the
  noise ceiling from synthetic trajectory construction).

Training loop:
  - One optimizer step per trajectory (stochastic gradient on trajectories)
  - All session embeddings are precomputed and cached before epoch 1
  - Gradient clipping: max norm 1.0 (prevents rare exploding gradients)
  - Cosine LR annealing: warm start at lr, decays to lr/100 over training
  - Best checkpoint saved by validation loss

Run as smoke test (50 epochs on 300 sample trajectories):
    python -m training.train
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding precomputation
# ---------------------------------------------------------------------------

def precompute_embeddings(
    trajectories: list[dict[str, Any]],
    encoder,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """
    Encode every unique session referenced across all trajectories.

    Returns a dict mapping conv_id → (384,) float32 tensor on `device`.
    Sessions shared across trajectories (due to pool reuse) are encoded once.

    The encoding is done in one large batch for efficiency:
    all turn texts from all unique sessions are concatenated, encoded together,
    then split and weighted-pooled back into per-session embeddings.
    """
    from kokoro.encoder import decode_parlai_artifacts

    # Collect unique sessions
    unique: dict[str, list[dict[str, str]]] = {}
    for traj in trajectories:
        for sess in traj["sessions"]:
            if sess["conv_id"] not in unique:
                unique[sess["conv_id"]] = sess["turns"]

    logger.info(f"Precomputing embeddings for {len(unique)} unique sessions...")
    t0 = time.perf_counter()

    all_texts: list[str] = []
    conv_ids: list[str] = []
    slices: list[tuple[int, int]] = []
    weights: list[list[float]] = []

    for conv_id, turns in unique.items():
        texts_i, weights_i = [], []
        for turn in turns:
            cleaned = decode_parlai_artifacts(turn.get("content", ""))
            if cleaned:
                texts_i.append(cleaned)
                weights_i.append(2.0 if turn.get("role") == "user" else 1.0)
        if not texts_i:
            texts_i = [""]
            weights_i = [1.0]
        start = len(all_texts)
        all_texts.extend(texts_i)
        slices.append((start, start + len(texts_i)))
        weights.append(weights_i)
        conv_ids.append(conv_id)

    # Single large batch encode
    all_embs = encoder.model.encode(
        all_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )  # (total_turns, 384)

    # Weighted pool back to per-session embeddings
    cache: dict[str, torch.Tensor] = {}
    for i, conv_id in enumerate(conv_ids):
        start, end = slices[i]
        embs = all_embs[start:end]                          # (n_turns, 384)
        w = np.array(weights[i], dtype=np.float32)
        w /= w.sum()
        pooled = (embs * w[:, np.newaxis]).sum(axis=0)      # (384,)
        cache[conv_id] = torch.tensor(pooled, device=device)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"  {len(unique)} sessions encoded in {elapsed:.1f}s "
        f"({elapsed / len(unique) * 1000:.1f} ms/session)"
    )
    return cache


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def trajectory_loss(
    model,
    session_embeddings: list[torch.Tensor],
) -> torch.Tensor:
    """
    Compute mean cosine prediction loss for one trajectory.

    For a trajectory of length N, we make N-1 predictions:
      z_1 = model(z_0, e_0)  → predict e_1
      z_2 = model(z_1, e_1)  → predict e_2
      ...

    Loss at each step: 1 - cosine_similarity(z_{t+1}, normalize(e_{t+1}))

    Since the model output z_{t+1} is already L2-normalized (‖z‖ = 1),
    cosine_similarity = z_{t+1} · normalize(e_{t+1}).

    Args:
        model:             TransitionModel instance (must be in .train() or .eval()).
        session_embeddings: List of (384,) tensors, one per session in the trajectory.
                            Length must be ≥ 2.

    Returns:
        Scalar tensor, mean loss over N-1 prediction steps.
    """
    if len(session_embeddings) < 2:
        raise ValueError(
            f"Trajectory must have ≥ 2 sessions to compute a prediction loss, "
            f"got {len(session_embeddings)}"
        )

    device = session_embeddings[0].device
    state = model.initial_state(device=device)

    total_loss = torch.tensor(0.0, device=device)

    for t in range(len(session_embeddings) - 1):
        state = model(state, session_embeddings[t])
        target = F.normalize(session_embeddings[t + 1], dim=-1)
        cos_sim = (state * target).sum()
        total_loss = total_loss + (1.0 - cos_sim)

    return total_loss / (len(session_embeddings) - 1)


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def evaluate(
    model,
    trajectories: list[dict[str, Any]],
    emb_cache: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """Return mean trajectory loss over a set of trajectories (no gradients)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for traj in trajectories:
            embs = [emb_cache[s["conv_id"]] for s in traj["sessions"]]
            total += trajectory_loss(model, embs).item()
    return total / len(trajectories)


def train(
    model,
    train_trajectories: list[dict[str, Any]],
    val_trajectories: list[dict[str, Any]],
    emb_cache: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,
    log_every_n_steps: int = 10,
    checkpoint_path: Path = Path("checkpoints/transition_v1.pt"),
) -> dict[str, list[float]]:
    """
    Train the transition model.

    Args:
        model:               TransitionModel instance.
        train_trajectories:  Training split.
        val_trajectories:    Validation split.
        emb_cache:           Precomputed session embeddings.
        device:              Training device.
        epochs:              Number of full passes over train_trajectories.
        lr:                  Initial learning rate for AdamW.
        weight_decay:        L2 regularisation coefficient.
        clip_grad_norm:      Max gradient norm for clipping.
        log_every_n_steps:   Print step-level loss every N optimizer steps.
        checkpoint_path:     Where to save the best model checkpoint.

    Returns:
        Dict with "train_loss" and "val_loss" lists (one entry per epoch).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Cosine annealing: LR decays from lr → lr/100 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = math.inf
    best_epoch = -1

    n_train = len(train_trajectories)
    n_val = len(val_trajectories)
    logger.info(
        f"Training: {n_train} trajectories | Validation: {n_val} trajectories"
    )
    logger.info(
        f"Epochs: {epochs} | LR: {lr} → {lr/100:.2e} (cosine) | "
        f"Weight decay: {weight_decay}"
    )

    t_start = time.perf_counter()

    for epoch in range(epochs):
        # ----- training -----
        model.train()
        random.shuffle(train_trajectories)

        epoch_loss = 0.0
        step = 0

        for traj in train_trajectories:
            embs = [emb_cache[s["conv_id"]] for s in traj["sessions"]]
            loss = trajectory_loss(model, embs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            step += 1

            if step % log_every_n_steps == 0:
                logger.info(
                    f"  Epoch {epoch+1:>3} step {step:>4}/{n_train}  "
                    f"loss={step_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        train_loss = epoch_loss / n_train

        # ----- validation -----
        val_loss = evaluate(model, val_trajectories, emb_cache, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Epoch {epoch+1:>3}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"elapsed={elapsed:.0f}s"
        )

        # ----- checkpoint -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "model_config": {
                        "state_dim": model.state_dim,
                        "hidden_dim": model.hidden_dim,
                    },
                },
                checkpoint_path,
            )
            logger.info(f"  [best] New val_loss={val_loss:.4f} -- saved to {checkpoint_path}")

    logger.info(
        f"\nTraining complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
    )

    # Save history JSON alongside checkpoint so visualizations can load it
    history_path = checkpoint_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_loss": history["train_loss"],
                "val_loss":   history["val_loss"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "n_train": n_train,
                "n_val": n_val,
                "epochs": epochs,
                "lr": lr,
            },
            f,
            indent=2,
        )
    logger.info(f"History saved to {history_path}")

    return history


# ---------------------------------------------------------------------------
# __main__ — smoke test on 300 sample trajectories, 50 epochs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.path.insert(0, str(Path(__file__).parent.parent))
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Train Kokoro transition model")
    parser.add_argument(
        "--data", type=str,
        default=str(project_root / "data" / "trajectories_sample.json"),
        help="Path to trajectories JSON file",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(project_root / "checkpoints" / "transition_v1.pt"),
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-steps", type=int, default=50,
                        help="Log step-level loss every N steps (default: 50)")
    args = parser.parse_args()

    traj_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)

    # ----- Load trajectories -----
    if not traj_path.exists():
        logger.error(
            f"{traj_path} not found. "
            "Run: python -m data.construct_trajectories <N> [--out <path>]"
        )
        sys.exit(1)

    with open(traj_path) as f:
        trajectories = json.load(f)
    logger.info(f"Loaded {len(trajectories)} trajectories from {traj_path.name}")

    # ----- Train/val split -----
    random.seed(42)
    random.shuffle(trajectories)
    split = int(0.8 * len(trajectories))
    train_trajs = trajectories[:split]
    val_trajs = trajectories[split:]
    logger.info(f"Split: {len(train_trajs)} train / {len(val_trajs)} val")

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ----- Encoder -----
    from kokoro.encoder import SessionEncoder
    encoder = SessionEncoder()
    _ = encoder.model  # trigger model load before timing

    # ----- Precompute embeddings -----
    emb_cache = precompute_embeddings(train_trajs + val_trajs, encoder, device)

    # ----- Model -----
    from kokoro.transition import TransitionModel
    model = TransitionModel().to(device)
    logger.info(f"Model: {model.parameter_count():,} parameters")

    # ----- Train -----
    history = train(
        model=model,
        train_trajectories=train_trajs,
        val_trajectories=val_trajs,
        emb_cache=emb_cache,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        clip_grad_norm=1.0,
        log_every_n_steps=args.log_steps,
        checkpoint_path=ckpt_path,
    )

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("Training summary")
    print("=" * 60)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val = min(history["val_loss"])
    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    print(f"  Best val loss:   {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Final train loss: {final_train:.4f}")
    print(f"  Final val loss:   {final_val:.4f}")
    print(f"  Checkpoint:       {ckpt_path}")

    # Print loss curve (compact ASCII)
    print("\n  Loss curve (val, every 5 epochs):")
    for i in range(0, len(history["val_loss"]), 5):
        bar_len = int(history["val_loss"][i] * 30)
        bar = "#" * bar_len
        print(f"    Epoch {i+1:>3}: {history['val_loss'][i]:.4f}  {bar}")

    print("\nSmoke test complete.")
