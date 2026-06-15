# Kokoro — Development Report

**Project:** Emotional trajectory memory for AI companions  
**Model:** Transition model (MLP) mapping (state, session_embedding) → next_state  
**State space:** 384-dimensional, initialized to zero vector per user  
**Date:** June 2026

---

## 1. The Problem We Started With

The transition model had severe **dimensional collapse**. The 384-dimensional state space was effectively 1-dimensional — it tracked only valence polarity (happy vs. sad) and ignored arousal entirely.

**Evidence:**
- Participation Ratio = **1.4 / 384** (effective dimensionality)
- Decoder thresholds: `_VALENCE_POS = -0.121`, `_VALENCE_NEG = -0.133` — both negative, only 0.012 apart. All states were clustering in a tiny negative-valence band.
- Probe Pearson r for arousal ≈ 0 — the model had no arousal signal whatsoever

**Root causes identified (in order of severity):**

1. **Cosine prediction objective** — minimizing `1 - cos(z, target)` does not penalize variance collapse. A model can minimize this loss by outputting the same vector for every input while being "directionally correct."

2. **L2 normalization on output** — the transition model was normalizing its output to the unit hypersphere. VICReg's variance term requires per-dimension std ≥ γ (typically 1.0), but unit-sphere vectors have per-dim std ≈ 0.051 — a 20× conflict that made the variance term impossible to satisfy.

3. **Joint LayerNorm on concatenated input** — a single `LayerNorm(768)` was applied to `[state ‖ session_emb]`. After VICReg training, state vectors have norm ~14–19 while session embeddings (MiniLM) have norm ~1. The joint norm let the state dominate the input, making session content nearly invisible.

4. **Missing arousal-primary training data** — all arc templates moved valence and arousal together (Q1→Q2, Q3→Q4). There were no arcs where valence stays flat while arousal moves independently. The model had no training signal to distinguish the two axes.

5. **Bug in `depression_to_anxiety` arc** — a waypoint `unsettled(-0.1, +0.2)` introduced a +0.6 valence swing correlated with the arousal rise, re-entangling the axes even in the one arc meant to separate them.

---

## 2. What We Changed

### 2.1 VICReg Loss (`training/train.py`)

Replaced the cosine prediction objective with **VICReg** (Bardes et al. 2022):

```
L = 25·L_sim + 25·L_var + 1·L_cov
```

- **L_sim** (invariance): cosine similarity between predicted state z and target state t — `(1 - cos(z_normed, t)).mean()`  
  *(Note: originally implemented as MSE, which reintroduced a soft unit-sphere constraint. Fixed to cosine.)*
- **L_var** (variance): `mean(relu(γ - std(z, dim=0)))` with γ=1.0 — forces each dimension to have std ≥ 1
- **L_cov** (covariance): penalizes off-diagonal covariance — `Σ_off² / D` — prevents dimensions from collapsing together

Training batches 32 trajectories simultaneously, giving ~214 state vectors per gradient step for the variance and covariance terms.

### 2.2 Removed L2 Normalization (`kokoro/transition.py`)

Removed the unit-sphere normalization from the model output. The output is now a raw 384-dim vector with unconstrained magnitude. Downstream components that need unit vectors (retrieval cosine ops) normalize independently.

### 2.3 Split LayerNorm (`kokoro/transition.py`)

Replaced the joint `LayerNorm(768)` on the concatenated input with two independent norms:

```python
# Before
self.net = nn.Sequential(
    nn.LayerNorm(state_dim * 2),   # joint — let high-norm state dominate
    nn.Linear(state_dim * 2, hidden_dim),
    ...
)

# After
self.state_norm = nn.LayerNorm(state_dim)
self.emb_norm   = nn.LayerNorm(state_dim)

def forward(self, state, session_emb):
    s = self.state_norm(state)
    e = self.emb_norm(session_emb)
    return self.net(torch.cat([s, e], dim=-1))
```

This gives state and session embedding equal weight in the input regardless of their absolute magnitudes.

### 2.4 Arousal-Primary Arc Templates (`data/arc_templates.py`)

Added three new arc templates where valence stays flat while arousal moves independently:

| Arc | Valence | Arousal | Description |
|-----|---------|---------|-------------|
| `excitement_to_contentment` | high positive (flat) | drops Q1→Q2 | Energy winds down after a peak event |
| `anxiety_to_depression` | negative (flat) | drops Q3→Q4 | Burnout — high-arousal distress → low-energy withdrawal |
| `depression_to_anxiety` | negative (flat) | rises Q4→Q3 | Activation of existing low mood into anxiety |

**Bug fixed in `depression_to_anxiety`:** the original waypoint `unsettled(-0.1, +0.2)` introduced a valence swing from -0.6 to -0.1 (+0.5 correlated with arousal rise). Replaced with `anxious(-0.4, +0.5)` to keep valence ≤ -0.4 throughout.

### 2.5 VAD Lexicon Features (`kokoro/vad.py`, `kokoro/encoder.py`)

Added an optional explicit arousal channel via Warriner/NRC-VAD lexicon scores:

- `kokoro/vad.py` — `VADLexicon` class with ~145-word subset, `score_turns()` returns (valence, arousal, dominance) float32
- `kokoro/encoder.py` — `SessionEncoder(use_vad_features=True)` appends 3-dim VAD vector to the 384-dim MiniLM embedding → 387-dim output
- VAD arousal separation gap: **1.432** (vs MiniLM's 0.112) — explicit arousal signal is ~13× stronger than what MiniLM encodes implicitly

**Dimension mismatch to resolve before training with VAD:** the transition model's `emb_norm` is `LayerNorm(384)`. Enabling VAD produces 387-dim encoder output. A projection layer (387→384) or resizing `emb_norm` and the first `Linear` to 387 will be needed — the checkpoint is incompatible and will require a full retrain.

### 2.6 Diagnostic Scripts (`diagnostics/`)

Four diagnostic tools written to monitor model health:

| Script | What it checks | Failure threshold |
|--------|---------------|-------------------|
| `per_arc_val_loss.py` | Per-arc cosine sim loss | Arousal arcs >0.05 above valence arc mean |
| `state_ablation.py` | Normal vs zeroed-state forward pass | Gap <0.05 (state has no effect) |
| `topic_leakage.py` | Spearman r between emotional/semantic retrieval axes | r >0.8 (axes not separated) |
| `norm_drift.py` | ‖s_t‖ over 50+ steps | Growth >10× or decay <0.1× |

---

## 3. Training

### Run 1 — VICReg with MSE sim term (epochs 1–36)
- Val loss stuck at **0.9609** for 36 epochs
- Root cause: MSE(z, unit_norm_target) pulls z toward norm≈1, creating the same soft unit-sphere constraint that L2 norm removal was meant to fix. The variance term (requires std≥1) and the MSE term (requires norm≈1 → std≈0.051) were in direct conflict.

### Run 2 — VICReg with cosine sim term, old joint-LN architecture (epochs 37–51)
- Val loss jumped to **0.5059** at epoch 37 (immediate improvement after cosine fix)
- Plateaued there for all remaining epochs
- Checkpoint saved: epoch 51, val_loss=0.5058
- PR computed: **22.1 / 384** — collapse fixed, but state space not fully utilizing split-LN benefits

### Run 3 — VICReg with cosine sim term, new split-LN architecture, fixed arc data
- Val loss reached **0.5087** at epoch 25 (best), then plateaued
- Checkpoint saved: epoch 25, val_loss=0.5087
- PR computed: **245.3 / 384** — 175× improvement over baseline

**Note on the val loss paradox:** Run 3 has a slightly *higher* val loss than Run 2 (0.5087 vs 0.5058) despite a massively better PR and probe scores. This is not a contradiction. The cosine val loss only measures directional prediction accuracy — whether the state points toward the next session. It does not measure how spread the representation is. The split LayerNorm gave the model better representational structure (PR: 22→245, valence r: 0.23→0.69) while the prediction ceiling stayed the same because it is set by data quality, not architecture. Val loss is a collapse detector here, not the headline metric; the probe scores are.

---

## 4. Results

### Linear Probe — headline metric

| Stage | Valence r | Arousal r | Sanity check¹ |
|-------|-----------|-----------|--------------|
| Baseline | ~0 | ~0 | identical predictions |
| VICReg + joint LN | 0.23 | 0.19 | identical predictions |
| VICReg + split LN | **0.69** | **0.54** | +0.350 vs −0.510 ✓ |

¹ *Sanity check: a stable-positive arc and a stable-negative arc are run through the model and probe independently. The check passes when the positive arc's predicted valence exceeds the negative arc's. Identical predictions (old checkpoints) indicate the probe cannot distinguish polar-opposite emotional trajectories.*

### Participation Ratio — collapse detector

| Stage | PR | Top-1 variance |
|-------|----|----------------|
| Baseline (cosine loss, L2 norm) | 1.4 / 384 | ~100% in dim-1 |
| VICReg + joint LayerNorm (epoch 51) | 22.1 / 384 | 12.6% |
| VICReg + split LayerNorm (epoch 25) | **245.3 / 384** | **1.0%** |

PR = (Σλ_i)²/Σλ_i² measures how evenly variance is distributed across the 384 output dimensions. A value near 1 means one dimension dominates (collapsed); a value near 384 means all dimensions carry equal variance. Note: PR measures spread, not usefulness — the covariance loss term actively pushes dimensions apart, so high PR is partly the loss doing what it's told. The probe scores above are the evidence that the spread is meaningful.

### Decoder Thresholds (recalibrated from val-set probe distribution)

| Threshold | Old value | New value | Basis |
|-----------|-----------|-----------|-------|
| `_VALENCE_POS` | −0.121 | **−0.005** | 67th percentile of predictions |
| `_VALENCE_NEG` | −0.133 | **−0.367** | 33rd percentile of predictions |
| `_AROUSAL_HIGH` | +0.300 | **+0.302** | 75th percentile of predictions |
| `_AROUSAL_LOW` | −0.100 | **+0.024** | 25th percentile of predictions |

The old valence thresholds (both negative, 0.012 apart) confirm how completely collapsed the original state space was. **Any retrain invalidates these thresholds** — they are percentiles of the current checkpoint's prediction distribution. The recalibration script is the inline computation in Section 3; it should be run as a standard post-training step rather than treated as fixed constants.

---

## 5. Known Limitations

### Arousal coverage gap — structural, not just a bug fix

Ground truth arousal in the training data only spans [−0.30, 0.80]. This is a structural limit of the EmpatheticDialogues dataset: the deepest Q4 emotion labels (`devastated`, `sad`, `lonely`) have arousal values of −0.20 to −0.30. Arc templates that specify lower-arousal zones (e.g., `exhausted` at −0.6, `depressed` at −0.5) find no direct matches in the session pool and fall back to a tolerance boost (+0.15), which maps them to `sad`/`devastated` sessions — sessions with actual arousal around −0.25.

The `depression_to_anxiety` arc bug was correctly fixed (valence kept ≤ −0.4 throughout), but this doesn't change the dataset ceiling. The ground truth arousal floor will remain near −0.30 until the session pool is supplemented with genuinely low-energy negative content. Arousal r=0.54 reflects this; the model has learned to separate axes but is limited by what the training distribution covers.

### Synthetic data ceiling
Val loss plateaued at ~0.508 across all three training runs. This is a data ceiling — the model has learned everything the synthetic arc templates can teach. Real multi-session user histories would provide more varied trajectory signal.

### Probe linearity assumption
The valence/arousal probe is a single linear layer (384→2). r=0.69/0.54 means the circumplex coordinates are linearly decodable from the state, which is the right property. A nonlinear probe would likely score higher but would test representational richness rather than decodability.

---

## 6. File Inventory

### Modified
| File | Change |
|------|--------|
| `kokoro/transition.py` | Split LayerNorm, removed L2 norm, 2→3 linear layers; fixed stale self-test assertions |
| `kokoro/encoder.py` | Optional VAD features (use_vad_features param) |
| `kokoro/decoder.py` | Recalibrated thresholds, removed "L2-normalised" requirement |
| `training/train.py` | VICReg loss, participation_ratio(), cosine sim term, CPU-only fix |
| `training/train_probe.py` | Correct checkpoint path, architecture auto-detection, initial_state fix |
| `data/arc_templates.py` | 3 arousal-primary arcs, fixed depression_to_anxiety waypoint |

### Added
| File | Purpose |
|------|---------|
| `kokoro/vad.py` | Warriner/NRC-VAD lexicon, explicit arousal channel |
| `diagnostics/per_arc_val_loss.py` | Per-arc validation loss monitor |
| `diagnostics/state_ablation.py` | State contribution ablation |
| `diagnostics/topic_leakage.py` | Emotional/semantic axis separation |
| `diagnostics/norm_drift.py` | State vector norm stability |

### Checkpoints
| File | Contents |
|------|---------|
| `checkpoints/transition_v1.pt` | Epoch 25, val_loss=0.5087, split-LN architecture |
| `checkpoints/valence_arousal_probe.pt` | Linear probe, valence r=0.69, arousal r=0.54 |

---

## 7. Next Steps (prioritised)

1. **VAD features in training** — cheapest available win. `SessionEncoder(use_vad_features=True)` gives the model a 13× stronger arousal signal (VAD gap 1.432 vs MiniLM 0.112). Requires: add a projection layer (387→384) or resize `emb_norm` + first `Linear` to accept 387-dim input, then retrain from scratch. Likely to improve arousal r meaningfully without touching the data pipeline.

2. **Automate decoder threshold recalibration** — the current thresholds are hardcoded percentiles of one checkpoint's prediction distribution. Any retrain invalidates them. Add a post-training step that rolls the val set through the new model+probe and writes the 33/67/25/75 percentiles directly to `decoder.py` constants.

3. **Supplement Q4 data** — to push arousal below −0.30, the session pool needs low-energy negative content that EmpatheticDialogues doesn't have. Options: add a second dataset (DailyDialog has more neutral/fatigued content), or generate synthetic conversation text for `exhausted`/`depressed` zones.

4. **Full retrain with LSTM** — the docstring in `transition.py` has the upgrade path. When real multi-session user data is available, swap the MLP for LSTM. The public API is unchanged.

5. **Real data** — EmpatheticDialogues is a single-session dataset. Real multi-session histories would let the model learn genuine longitudinal trajectory dynamics rather than synthetic arc patterns.
