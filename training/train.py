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

    # Weighted pool back to per-session embeddings, then append VAD if enabled
    use_vad = encoder.output_dim > 384
    vad_lexicon = encoder._get_vad() if use_vad else None

    cache: dict[str, torch.Tensor] = {}
    for i, conv_id in enumerate(conv_ids):
        start, end = slices[i]
        embs = all_embs[start:end]                          # (n_turns, 384)
        w = np.array(weights[i], dtype=np.float32)
        w /= w.sum()
        pooled = (embs * w[:, np.newaxis]).sum(axis=0)      # (384,)
        if use_vad:
            vad = vad_lexicon.score_turns(unique[conv_id])  # (3,)
            pooled = np.concatenate([pooled, vad])          # (387,)
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

    Used for VALIDATION only — measures prediction signal without regularization
    terms, making val_loss comparable across training configurations.

    For a trajectory of length N, we make N-1 predictions:
      z_1 = model(z_0, e_0)  → predict e_1
      z_2 = model(z_1, e_1)  → predict e_2
      ...

    Loss at each step: 1 - cosine_similarity(normalize(z_{t+1}), normalize(e_{t+1}))

    Model output is no longer L2-normalized, so we normalize explicitly here.

    Args:
        model:              TransitionModel instance (must be in .eval()).
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
        state_normed = F.normalize(state, dim=-1)
        target = F.normalize(session_embeddings[t + 1][:model.state_dim], dim=-1)
        cos_sim = (state_normed * target).sum()
        total_loss = total_loss + (1.0 - cos_sim)

    return total_loss / (len(session_embeddings) - 1)


def vicreg_loss(
    model,
    batch_session_embeddings: list[list[torch.Tensor]],
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    VICReg objective for transition model training (Bardes et al. 2022).

    Collects all predicted states from a mini-batch of trajectories into a
    matrix Z of shape (N_total, 384), then computes three terms:

      Invariance (sim): cosine similarity between normalize(z_{t+1}) and normalize(e_{t+1}).
        Prediction signal — keeps the state pointing toward the next session.
        Cosine (not MSE): only direction matters, not magnitude. MSE with unit-norm
        targets would pull z toward norm≈1, creating a soft unit-sphere constraint
        that conflicts with the variance term (unit-sphere std ≈ 0.051 vs gamma=1.0).

      Variance: mean_d[ ReLU(gamma - std(Z[:,d])) ]
        Forces each of the 384 output dimensions to maintain std >= gamma across
        the batch. This is the primary force that breaks dimensional collapse —
        dead dimensions generate gradient that pushes weights to activate them.

      Covariance: sum_{i≠j}(Cov(Z)_{ij}^2) / D
        Penalizes off-diagonal covariance between dimensions. Once the variance
        term activates multiple dimensions, this term decorrelates them —
        preventing arousal from being re-expressed as a variant of valence.

    Batch size matters for covariance quality: with batch_size=32 and ~7.7
    sessions/trajectory, Z has ~213 rows, giving a reasonable rank for the
    384×384 covariance estimate.

    Args:
        model:                    TransitionModel instance (in .train() mode).
        batch_session_embeddings: List of trajectories; each trajectory is a list
                                  of (384,) tensors. Trajectories with < 2 sessions
                                  are silently skipped.
        sim_weight:               λ — prediction term weight (default 25.0).
        var_weight:               μ — variance term weight (default 25.0).
        cov_weight:               ν — covariance term weight (default 1.0).
        gamma:                    Target std per dimension (default 1.0).

    Returns:
        Scalar loss tensor.
    """
    device = batch_session_embeddings[0][0].device

    all_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for session_embeddings in batch_session_embeddings:
        if len(session_embeddings) < 2:
            continue
        state = model.initial_state(device=device)
        for t in range(len(session_embeddings) - 1):
            state = model(state, session_embeddings[t])
            # Target is the semantic (MiniLM) portion of the next session embedding.
            # VAD dims are input signal only — state output space is always state_dim.
            target = F.normalize(session_embeddings[t + 1][:model.state_dim], dim=-1)
            all_states.append(state)
            all_targets.append(target)

    Z = torch.stack(all_states)   # (N_total, 384)
    T = torch.stack(all_targets)  # (N_total, 384)
    N, D = Z.shape

    # --- Invariance (prediction) term ---
    # Cosine, not MSE: only the direction of z matters, not its magnitude.
    # MSE with unit-norm targets (T) would pull z toward norm≈1, creating a
    # soft unit-sphere constraint that conflicts with the variance term
    # (unit-sphere std per dimension ≈ 0.051, but gamma=1.0 requires std≥1).
    z_normed = F.normalize(Z, dim=-1)
    sim_loss = (1.0 - (z_normed * T).sum(dim=-1)).mean()

    # --- Variance term ---
    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    std = torch.sqrt(Z_centered.var(dim=0) + 1e-4)   # (D,), epsilon for stability
    var_loss = torch.mean(F.relu(gamma - std))

    # --- Covariance term ---
    cov = (Z_centered.T @ Z_centered) / (N - 1)      # (D, D)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag_sq / D

    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss


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


def participation_ratio(model, val_trajectories, emb_cache, device) -> float:
    """
    Compute the participation ratio of state vectors on the validation set.

    PR = (Σλ_i)² / Σλ_i²  where λ_i are eigenvalues of the state covariance matrix.

    Interpretation: effective number of dimensions the model uses.
    Target: > 20 / 384 (up from ~1.4 with cosine-only training).
    """
    model.eval()
    states = []
    with torch.no_grad():
        for traj in val_trajectories:
            embs = [emb_cache[s["conv_id"]] for s in traj["sessions"]]
            state = model.initial_state(device=device)
            for t in range(len(embs) - 1):
                state = model(state, embs[t])
                states.append(state.cpu().numpy())
    Z = np.array(states)                      # (N, 384)
    cov = np.cov(Z.T)                         # (384, 384)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def train(
    model,
    train_trajectories: list[dict[str, Any]],
    val_trajectories: list[dict[str, Any]],
    emb_cache: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,
    batch_size: int = 32,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    log_every_n_steps: int = 10,
    checkpoint_path: Path = Path("checkpoints/transition_v1.pt"),
) -> dict[str, list[float]]:
    """
    Train the transition model with VICReg objective.

    Trajectories are grouped into mini-batches of `batch_size` before computing
    the loss. This is required for the VICReg variance and covariance terms, which
    need a sufficiently large batch of state vectors to estimate per-dimension
    statistics reliably. With batch_size=32 and ~7.7 sessions/trajectory, each
    batch yields ~213 state vectors — large enough for a stable 384×384 covariance.

    Note on epochs vs old training: each epoch now has n_train/batch_size gradient
    steps (not n_train). The default epochs=100 compensates for this; for the 10k
    dataset that gives ~25k gradient steps vs ~400k before — still sufficient for
    a ~857k parameter model and substantially faster per epoch.

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
        batch_size:          Trajectories per gradient step (VICReg batch).
        sim_weight:          VICReg λ — prediction term weight.
        var_weight:          VICReg μ — variance term weight.
        cov_weight:          VICReg ν — covariance term weight.
        gamma:               VICReg target std per output dimension.
        log_every_n_steps:   Print step-level loss every N optimizer steps.
        checkpoint_path:     Where to save the best model checkpoint.

    Returns:
        Dict with "train_loss", "val_loss", and "participation_ratio" (final).
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

    history: dict[str, Any] = {"train_loss": [], "val_loss": []}
    best_val_loss = math.inf
    best_epoch = -1

    n_train = len(train_trajectories)
    n_val = len(val_trajectories)
    n_batches = math.ceil(n_train / batch_size)
    logger.info(
        f"Training: {n_train} trajectories | Validation: {n_val} trajectories"
    )
    logger.info(
        f"Epochs: {epochs} | Batch size: {batch_size} | "
        f"{n_batches} steps/epoch | "
        f"LR: {lr} → {lr/100:.2e} (cosine) | Weight decay: {weight_decay}"
    )
    logger.info(
        f"VICReg weights — sim: {sim_weight}  var: {var_weight}  "
        f"cov: {cov_weight}  gamma: {gamma}"
    )

    t_start = time.perf_counter()

    for epoch in range(epochs):
        # ----- training -----
        model.train()
        random.shuffle(train_trajectories)

        epoch_loss = 0.0
        step = 0

        for batch_start in range(0, n_train, batch_size):
            batch = train_trajectories[batch_start : batch_start + batch_size]
            batch_embs = [
                [emb_cache[s["conv_id"]] for s in traj["sessions"]]
                for traj in batch
            ]
            loss = vicreg_loss(
                model, batch_embs,
                sim_weight=sim_weight,
                var_weight=var_weight,
                cov_weight=cov_weight,
                gamma=gamma,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            step += 1

            if step % log_every_n_steps == 0:
                logger.info(
                    f"  Epoch {epoch+1:>3} step {step:>4}/{n_batches}  "
                    f"loss={step_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        train_loss = epoch_loss / n_batches

        # ----- validation (cosine prediction loss — no regularization terms) -----
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
                        "state_dim":   model.state_dim,
                        "session_dim": model.session_dim,
                        "hidden_dim":  model.hidden_dim,
                        "l2_norm":     False,
                    },
                },
                checkpoint_path,
            )
            logger.info(f"  [best] New val_loss={val_loss:.4f} -- saved to {checkpoint_path}")

    # ----- participation ratio (computed once on final model) -----
    pr = participation_ratio(model, val_trajectories, emb_cache, device)
    history["participation_ratio"] = pr
    logger.info(f"\nParticipation ratio: {pr:.1f} / {model.state_dim}  (target > 20)")

    logger.info(
        f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
    )

    # Save history JSON alongside checkpoint so visualizations can load it
    history_path = checkpoint_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_loss": history["train_loss"],
                "val_loss":   history["val_loss"],
                "participation_ratio": pr,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "n_train": n_train,
                "n_val": n_val,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "vicreg": {
                    "sim_weight": sim_weight,
                    "var_weight": var_weight,
                    "cov_weight": cov_weight,
                    "gamma": gamma,
                },
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Trajectories per VICReg gradient step (default: 32)")
    parser.add_argument("--sim-weight", type=float, default=25.0)
    parser.add_argument("--var-weight", type=float, default=25.0)
    parser.add_argument("--cov-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=50,
                        help="Log step-level loss every N steps (default: 50)")
    parser.add_argument("--holdout-arcs", nargs="*", default=[],
                        metavar="ARC",
                        help="Arc type names to withhold from training entirely "
                             "(e.g. --holdout-arcs grief_arc post_traumatic_growth). "
                             "Withheld arcs are removed from train; val keeps all arcs "
                             "so you can measure OOD performance on the withheld types.")
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

    # ----- Arc holdout (OOD experiment) -----
    holdout_arcs: set[str] = set(args.holdout_arcs)
    if holdout_arcs:
        before = len(trajectories)
        # Val keeps ALL arcs (so we can evaluate OOD performance on withheld types)
        # Train removes withheld arcs entirely
        random.seed(42)
        random.shuffle(trajectories)
        split = int(0.8 * len(trajectories))
        train_trajs_all = trajectories[:split]
        val_trajs = trajectories[split:]
        train_trajs = [t for t in train_trajs_all if t.get("arc_name", "") not in holdout_arcs]
        removed = len(train_trajs_all) - len(train_trajs)
        logger.info(
            f"Arc holdout OOD mode: withheld arcs = {sorted(holdout_arcs)}\n"
            f"  Removed {removed} train trajectories covering withheld arcs\n"
            f"  Train: {len(train_trajs)}  |  Val: {len(val_trajs)} (all arcs, for OOD eval)"
        )
    else:
        # ----- Train/val split (standard) -----
        random.seed(42)
        random.shuffle(trajectories)
        split = int(0.8 * len(trajectories))
        train_trajs = trajectories[:split]
        val_trajs = trajectories[split:]
    logger.info(f"Split: {len(train_trajs)} train / {len(val_trajs)} val")

    # ----- Device -----
    # Always CPU — this project targets CPU-only deployment.
    # torch.cuda.is_available() is avoided intentionally: on Windows it
    # initialises the CUDA driver even when no toolkit is present, which
    # causes a deferred SIGSEGV in bash subprocesses.
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # ----- Encoder -----
    from kokoro.encoder import SessionEncoder
    encoder = SessionEncoder(use_vad_features=True)
    _ = encoder.model  # trigger model load before timing

    # ----- Precompute embeddings -----
    emb_cache = precompute_embeddings(train_trajs + val_trajs, encoder, device)

    # ----- Model -----
    from kokoro.transition import TransitionModel
    model = TransitionModel(session_dim=encoder.output_dim).to(device)
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
        batch_size=args.batch_size,
        sim_weight=args.sim_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        gamma=args.gamma,
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
    pr = history.get("participation_ratio", float("nan"))
    print(f"  Best val loss:        {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Final train loss:     {final_train:.4f}")
    print(f"  Final val loss:       {final_val:.4f}")
    print(f"  Participation ratio:  {pr:.1f} / {model.state_dim}  (target > 20)")
    print(f"  Checkpoint:           {ckpt_path}")

    # Print loss curve (compact ASCII)
    print("\n  Loss curve (val, every 5 epochs):")
    for i in range(0, len(history["val_loss"]), 5):
        bar_len = int(history["val_loss"][i] * 30)
        bar = "#" * bar_len
        print(f"    Epoch {i+1:>3}: {history['val_loss'][i]:.4f}  {bar}")

    print("\nSmoke test complete.")
