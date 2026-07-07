"""
diagnostics/common.py — shared, checkpoint-faithful loading for all diagnostics.

Why this exists
---------------
Four diagnostics (topic_leakage, state_ablation, norm_drift, per_arc_val_loss)
previously rebuilt the transition model with

    cfg = {k: v for k, v in cfg.items() if k in ("state_dim", "hidden_dim")}

which silently DROPS session_dim. Any checkpoint trained with VAD features
(session_dim=387 — including the current production checkpoint transition_v1.pt)
then fails to load: the reconstructed emb_norm is LayerNorm(384) but the stored
weights are shape (387,). The same scripts also instantiated
SessionEncoder() without VAD features, so even if loading had succeeded the
inputs would have been 384-dim against a 387-dim model. In short: the entire
monitoring suite was broken for the model it was supposed to monitor.

This module gives every diagnostic one correct path:

    from diagnostics.common import load_transition_model, make_encoder, build_embedding_cache

    model, cfg = load_transition_model(ckpt_path)     # honours session_dim
    encoder    = make_encoder(cfg)                    # VAD on iff checkpoint used it
    cache      = build_embedding_cache(trajs, encoder)  # (session_dim,) tensors

When comparing state vectors (state_dim = 384) against session embeddings that
may be 387-dim, always slice the embedding: ``emb[:model.state_dim]``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_transition_model(ckpt_path: str | Path):
    """Load a transition-model checkpoint (GRU or legacy MLP, auto-detected),
    preserving session_dim. Returns (model, model_config)."""
    from kokoro.transition import load_transition_checkpoint

    ckpt_path = Path(ckpt_path)
    model, cfg = load_transition_checkpoint(ckpt_path)
    logger.info(
        f"Loaded {ckpt_path.name}: arch={cfg.get('arch', 'mlp')}, "
        f"session_dim={model.session_dim}"
    )
    return model, cfg


def make_encoder(model_config: dict):
    """Return a SessionEncoder whose output dim matches the checkpoint."""
    from kokoro.encoder import SessionEncoder

    use_vad = model_config.get("session_dim", 384) > 384
    # device pinned to CPU: on this project's target environment, letting
    # sentence-transformers auto-detect CUDA inside bash subprocesses causes
    # deferred segfaults (see the device note in training/train.py).
    encoder = SessionEncoder(use_vad_features=use_vad, device="cpu")
    _ = encoder.model  # trigger lazy load
    return encoder


def build_embedding_cache(
    trajectories: list[dict],
    encoder,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Encode every unique session in *trajectories*.

    Returns conv_id -> (encoder.output_dim,) float32 tensor. Appends VAD
    features when the encoder was built with use_vad_features=True, matching
    the training-time preprocessing in training/train.py exactly.
    """
    from kokoro.encoder import decode_parlai_artifacts

    unique: dict[str, list] = {}
    for traj in trajectories:
        for sess in traj["sessions"]:
            if sess["conv_id"] not in unique:
                unique[sess["conv_id"]] = sess["turns"]

    logger.info(f"Encoding {len(unique)} unique sessions...")
    t0 = time.perf_counter()

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
        all_texts, batch_size=batch_size, convert_to_numpy=True,
        show_progress_bar=False, normalize_embeddings=False,
    )

    use_vad = encoder.output_dim > 384
    vad_lexicon = encoder._get_vad() if use_vad else None

    cache: dict[str, torch.Tensor] = {}
    for i, conv_id in enumerate(conv_ids):
        s, e = slices[i]
        w = np.array(weights_all[i], dtype=np.float32)
        w /= w.sum()
        pooled = (embs[s:e] * w[:, np.newaxis]).sum(axis=0).astype(np.float32)
        if use_vad:
            vad = vad_lexicon.score_turns(unique[conv_id])
            pooled = np.concatenate([pooled, vad])
        cache[conv_id] = torch.tensor(pooled)

    logger.info(f"  done in {time.perf_counter() - t0:.1f}s")
    return cache
