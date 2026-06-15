"""
diagnostics/state_ablation.py

Evaluate the transition model twice on the validation set:
  1. Normally — state carries trajectory history
  2. With state zeroed at every step — model sees only the current session

If the gap between the two sim losses is < 0.05, the state vector is not
doing meaningful work and the "world model" claim is unsupported. The model
would be no better than a session-only encoder at each step.

Note: a small gap is not necessarily fatal — on short trajectories (6–9
sessions) the state may not yet diverge much from session-only encoding.
The result should be interpreted alongside the separation ratio from
training/validate.py.

Run from project root:
    python -m diagnostics.state_ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.transition import TransitionModel
from kokoro.encoder import decode_parlai_artifacts

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
GAP_THRESHOLD = 0.05


def load_model(ckpt_path: Path) -> TransitionModel:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg  = ckpt.get("model_config", {"state_dim": 384, "hidden_dim": 512})
    cfg  = {k: v for k, v in cfg.items() if k in ("state_dim", "hidden_dim")}
    model = TransitionModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def build_embedding_cache(trajectories, encoder, device):
    unique = {}
    for traj in trajectories:
        for sess in traj["sessions"]:
            if sess["conv_id"] not in unique:
                unique[sess["conv_id"]] = sess["turns"]

    logger.info(f"Encoding {len(unique)} sessions...")
    all_texts, conv_ids, slices, weights_all = [], [], [], []
    for conv_id, turns in unique.items():
        texts, weights = [], []
        for turn in turns:
            cleaned = decode_parlai_artifacts(turn.get("content", ""))
            if cleaned:
                texts.append(cleaned)
                weights.append(2.0 if turn.get("role") == "user" else 1.0)
        if not texts:
            texts, weights = [""], [1.0]
        s = len(all_texts)
        all_texts.extend(texts)
        slices.append((s, s + len(texts)))
        weights_all.append(weights)
        conv_ids.append(conv_id)

    embs = encoder.model.encode(
        all_texts, batch_size=256, convert_to_numpy=True,
        show_progress_bar=False, normalize_embeddings=False,
    )
    cache = {}
    for i, conv_id in enumerate(conv_ids):
        s, e = slices[i]
        w = np.array(weights_all[i], dtype=np.float32)
        w /= w.sum()
        pooled = (embs[s:e] * w[:, np.newaxis]).sum(axis=0).astype(np.float32)
        cache[conv_id] = torch.tensor(pooled)
    return cache


def evaluate_normal(model, trajectories, emb_cache):
    """Standard evaluation — state carries trajectory history."""
    losses = []
    with torch.no_grad():
        for traj in trajectories:
            embs = [emb_cache[s["conv_id"]] for s in traj["sessions"]]
            if len(embs) < 2:
                continue
            state = model.initial_state()
            total = 0.0
            for t in range(len(embs) - 1):
                state  = model(state, embs[t])
                state_n = F.normalize(state, dim=-1)
                target  = F.normalize(embs[t + 1], dim=-1)
                total  += (1.0 - (state_n * target).sum()).item()
            losses.append(total / (len(embs) - 1))
    return float(np.mean(losses))


def evaluate_zeroed_state(model, trajectories, emb_cache):
    """Ablation — state is zeroed at every step.

    model(zeros, e_t) → z_{t+1} is purely a function of the current session
    with no accumulated history. If this matches normal evaluation, the state
    is carrying no useful information.
    """
    losses = []
    with torch.no_grad():
        for traj in trajectories:
            embs = [emb_cache[s["conv_id"]] for s in traj["sessions"]]
            if len(embs) < 2:
                continue
            total = 0.0
            for t in range(len(embs) - 1):
                # Always pass fresh zeros — no history
                state  = model.initial_state()
                state  = model(state, embs[t])
                state_n = F.normalize(state, dim=-1)
                target  = F.normalize(embs[t + 1], dim=-1)
                total  += (1.0 - (state_n * target).sum()).item()
            losses.append(total / (len(embs) - 1))
    return float(np.mean(losses))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="State ablation: history vs no-history")
    parser.add_argument("--data",       default=str(PROJECT_ROOT / "data" / "trajectories_10k.json"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "transition_v1.pt"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    for p in [Path(args.checkpoint), Path(args.data)]:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    with open(args.data) as f:
        all_trajs = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)
    n_val     = int(len(all_trajs) * args.val_fraction)
    val_trajs = all_trajs[-n_val:]
    logger.info(f"Val set: {n_val} trajectories")

    model = load_model(Path(args.checkpoint))

    from kokoro.encoder import SessionEncoder
    encoder = SessionEncoder()
    _ = encoder.model

    emb_cache = build_embedding_cache(val_trajs, encoder, torch.device("cpu"))

    logger.info("Evaluating with full state history...")
    loss_normal = evaluate_normal(model, val_trajs, emb_cache)

    logger.info("Evaluating with state zeroed at every step...")
    loss_zeroed = evaluate_zeroed_state(model, val_trajs, emb_cache)

    gap = loss_zeroed - loss_normal

    print()
    print("=" * 60)
    print("  State Ablation: does trajectory history matter?")
    print("=" * 60)
    print(f"  Normal (state carries history):  {loss_normal:.4f}")
    print(f"  Ablated (state zeroed each step): {loss_zeroed:.4f}")
    print(f"  Gap (zeroed - normal):            {gap:+.4f}")
    print()

    if gap < GAP_THRESHOLD:
        print(f"  FLAG: gap = {gap:.4f} < {GAP_THRESHOLD} threshold.")
        print(f"  The state vector is not doing meaningful work.")
        print(f"  The model is essentially session-only at each step.")
        print(f"  Consider: longer trajectories, stronger recurrence, or")
        print(f"  reviewing whether the transition model is actually using state.")
    else:
        print(f"  OK: gap = {gap:.4f} >= {GAP_THRESHOLD} threshold.")
        print(f"  The state vector carries information beyond the current session.")
        print(f"  Trajectory history is contributing to prediction quality.")

    print("=" * 60)


if __name__ == "__main__":
    main()
