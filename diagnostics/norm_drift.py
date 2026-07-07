"""
diagnostics/norm_drift.py

Roll the transition model forward on 20 MSC (multi-session chat) sequences
for 50+ steps and track ‖s_t‖ vs session number.

L2 normalization was removed from the model output (required for VICReg).
Without a replacement constraint, output norm can grow unboundedly during
inference — each step feeds the previous (unnormalized) state back as input,
which can amplify if the model learned to scale up outputs.

Flags if:
  - Mean norm at session 50 is > 10× mean norm at session 5 (unbounded growth)
  - Mean norm at session 50 is < 0.1× mean norm at session 5 (collapse to zero)

Uses nayohan/multi_session_chat (same as figures/validate_msc.py) to test on
real multi-session data not seen during training. Falls back to synthetic
trajectories if MSC is not available.

Run from project root:
    python -m diagnostics.norm_drift
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.encoder import decode_parlai_artifacts
from diagnostics.common import load_transition_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
N_SEQUENCES  = 20
N_STEPS      = 50
GROWTH_FLAG  = 10.0   # norm[50] / norm[5] > this → unbounded growth
DECAY_FLAG   = 0.1    # norm[50] / norm[5] < this → collapse to zero


def load_msc_sequences(n: int, seed: int = 42) -> list[list[dict]] | None:
    """
    Load n multi-session sequences from nayohan/multi_session_chat.
    Returns None if the dataset is not available.
    Each sequence is a list of session dicts [{role, content}].
    """
    try:
        from datasets import load_dataset
        logger.info("Loading nayohan/multi_session_chat...")
        ds = load_dataset("nayohan/multi_session_chat", split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load MSC: {e}")
        return None

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n * 3, len(ds)))

    sequences = []
    for idx in indices:
        row = ds[idx]
        # MSC format: list of sessions, each a list of utterances
        # Field name varies — try common names
        sessions_raw = row.get("sessions") or row.get("dialog") or row.get("conversations")
        if not sessions_raw or not isinstance(sessions_raw, list):
            continue
        # Convert to [{role, content}] turn format
        sessions_converted = []
        for sess in sessions_raw:
            if isinstance(sess, list):
                turns = []
                for i, utt in enumerate(sess):
                    role = "user" if i % 2 == 0 else "assistant"
                    content = utt if isinstance(utt, str) else str(utt)
                    turns.append({"role": role, "content": content})
                sessions_converted.append(turns)
        if len(sessions_converted) >= 5:
            sequences.append(sessions_converted)
        if len(sequences) >= n:
            break

    logger.info(f"Loaded {len(sequences)} MSC sequences")
    return sequences if sequences else None


def load_synthetic_sequences(
    n: int, n_steps: int, data_path: Path, seed: int = 42
) -> list[list[dict]]:
    """
    Fall back: load synthetic trajectories and pad them to n_steps sessions
    by repeating sessions with slight noise (to simulate extended use).
    """
    with open(data_path) as f:
        trajs = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(trajs)
    sequences = []
    for traj in trajs[:n]:
        sessions = [s["turns"] for s in traj["sessions"]]
        # Pad by cycling through sessions until n_steps
        while len(sessions) < n_steps:
            sessions.append(rng.choice(sessions))
        sequences.append(sessions[:n_steps])
    logger.info(f"Using {len(sequences)} synthetic sequences (padded to {n_steps} sessions)")
    return sequences


def encode_session(turns: list[dict], encoder) -> np.ndarray:
    """Encode via SessionEncoder.encode() so VAD features are appended when the
    checkpoint expects a 387-dim input (previously this hand-rolled 384-dim
    pooling and crashed VAD checkpoints at the transition model)."""
    try:
        return encoder.encode(turns)
    except ValueError:   # all-blank session
        return np.zeros(encoder.output_dim, dtype=np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Norm drift across long inference sequences")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "transition_v2.pt"))
    parser.add_argument("--data",       default=str(PROJECT_ROOT / "data" / "trajectories_10k_v2_val.json"))
    parser.add_argument("--n-sequences", type=int, default=N_SEQUENCES)
    parser.add_argument("--n-steps",     type=int, default=N_STEPS)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    model, cfg = load_transition_model(Path(args.checkpoint))
    from diagnostics.common import make_encoder
    encoder = make_encoder(cfg)

    # Try MSC first, fall back to synthetic
    sequences = load_msc_sequences(args.n_sequences, seed=args.seed)
    data_source = "MSC (real)"
    if sequences is None:
        if not Path(args.data).exists():
            sys.exit(f"Fallback data not found: {args.data}")
        sequences = load_synthetic_sequences(
            args.n_sequences, args.n_steps, Path(args.data), seed=args.seed
        )
        data_source = "synthetic (padded)"

    # Roll model forward, track norm at each step
    # norm_curves[i] = list of norms at step i across all sequences
    norm_curves: list[list[float]] = [[] for _ in range(args.n_steps)]

    logger.info(f"Rolling {len(sequences)} sequences for {args.n_steps} steps each...")
    for seq_idx, sessions in enumerate(sequences):
        state = model.initial_state()
        step  = 0
        for turns in sessions:
            if step >= args.n_steps:
                break
            emb = torch.tensor(encode_session(turns, encoder))
            with torch.no_grad():
                state = model(state, emb)
            norm_curves[step].append(state.norm().item())
            step += 1
        # If sequence is shorter than n_steps, pad with last session repeated
        while step < args.n_steps:
            with torch.no_grad():
                state = model(state, emb)
            norm_curves[step].append(state.norm().item())
            step += 1

    mean_norms = [float(np.mean(c)) for c in norm_curves if c]
    std_norms  = [float(np.std(c))  for c in norm_curves if c]

    # --- Print table ---
    print()
    print("=" * 65)
    print(f"  Norm Drift Analysis  (data: {data_source})")
    print("=" * 65)
    print(f"  Sequences: {len(sequences)}  |  Steps: {args.n_steps}")
    print()
    print(f"  {'Step':>5}  {'Mean ‖s_t‖':>12}  {'Std':>8}  {'Bar'}")
    print("-" * 65)

    display_steps = list(range(0, min(10, len(mean_norms)))) + \
                    list(range(10, len(mean_norms), max(1, len(mean_norms) // 20)))
    display_steps = sorted(set(display_steps))

    max_norm = max(mean_norms) if mean_norms else 1.0
    for i in display_steps:
        if i >= len(mean_norms):
            break
        bar_len = int(mean_norms[i] / max(max_norm, 1e-6) * 30)
        bar = "#" * bar_len
        print(f"  {i+1:>5}  {mean_norms[i]:>12.4f}  {std_norms[i]:>8.4f}  {bar}")

    print()

    # --- Diagnosis ---
    if len(mean_norms) >= 5:
        norm_early = mean_norms[4]   # step 5
        norm_late  = mean_norms[min(args.n_steps - 1, len(mean_norms) - 1)]
        ratio = norm_late / max(norm_early, 1e-9)

        print(f"  Norm at step 5:   {norm_early:.4f}")
        print(f"  Norm at step {min(args.n_steps, len(mean_norms)):>2}:  {norm_late:.4f}")
        print(f"  Ratio (late/early): {ratio:.2f}x")
        print()

        if ratio > GROWTH_FLAG:
            print(f"  FLAG: norm grew {ratio:.1f}x from step 5 to step {len(mean_norms)}.")
            print(f"  State magnitude is growing unboundedly — will cause overflow")
            print(f"  and retrieval instability on long-running users.")
            print(f"  Fix: add output clipping (torch.clamp) or soft norm constraint.")
        elif ratio < DECAY_FLAG:
            print(f"  FLAG: norm decayed to {ratio:.2f}x of early values.")
            print(f"  State is collapsing toward zero — model forgets everything.")
            print(f"  Fix: check for vanishing gradients or over-regularization.")
        else:
            print(f"  OK: norm ratio = {ratio:.2f}x (within [{DECAY_FLAG}x, {GROWTH_FLAG}x]).")
            print(f"  State magnitude is stable across long inference sequences.")
    else:
        print(f"  Not enough steps to evaluate stability ({len(mean_norms)} steps recorded).")

    print("=" * 65)


if __name__ == "__main__":
    main()
