# Kokoro Figure Validation Guide

This document describes what each figure should show. Use it alongside the images to verify correctness.

---

## fig0_architecture.png

**What it is:** System pipeline diagram showing the three main components.

**Should show:**
- Session Encoder → Transition Model → State Store + downstream outputs (Linear Probe / State Decoder)
- 384-dim state vector labeled somewhere
- Flow from raw conversation turns through to emotional state output

**Notes / known gaps:**
- Does NOT annotate VICReg loss internals (sim + var + cov) — that's a pipeline overview, not a model architecture diagram
- Does NOT show split-vs-joint LayerNorm detail — that is an internal model implementation detail not depicted here
- No Q1–Q4 quadrant mapping — that is in the decoder, not shown at this level

---

## fig1_circumplex_scatter.png

**What it is:** Valence-Arousal circumplex showing the ground-truth distribution of sessions in the training data (EmpatheticDialogues).

**Should show:**
- 2D scatter, Valence on x-axis, Arousal on y-axis
- Points colored/labeled by emotion category
- Quadrant labels: Q1 (high V, high A), Q2 (high V, low A), Q3 (low V, high A), Q4 (low V, low A)
- Arousal floor around −0.3 to −0.4 — deep Q4 is sparsely populated (dataset ceiling, known limitation)
- No single collapsed blob — spread across multiple quadrants

**Red flags:**
- All points in one cluster
- Arousal extending well below −0.5 (would mean the known limitation was resolved)

---

## fig2_arc_trajectories.png

**What it is:** Circumplex paths for four representative arc types. Intentionally shows only 4 arcs.

**Should show:**
- 2×2 grid, one arc per panel: Gradual Decline, Slow Recovery, Acute Stress → Stabilization, Grief Arc
- Each panel shows waypoints as a path through V-A space, colored by time progression
- Arrows or markers indicating direction of movement

**Notes / known gaps:**
- By design shows 4 of 14 arcs — the 3 arousal-primary arcs (excitement_to_contentment, anxiety_to_depression, depression_to_anxiety) are NOT shown here. Their flat-valence / arousal-axis movement cannot be verified from this figure.
- This is a design choice (2×2 grid), not a bug.

**Red flags:**
- Any of the 4 arcs shown moving in clearly wrong directions (e.g., Gradual Decline going upward in valence)

---

## fig3_arc_distribution.png

**What it is:** Bar chart of trajectory counts per arc type across the full training dataset.

**Should show:**
- All **14 arc types** including the 3 arousal-primary arcs: `excitement_to_contentment`, `anxiety_to_depression`, `depression_to_anxiety`
- Total n = **10,000** trajectories
- Broadly balanced counts (roughly 460–1,023 per arc, no 10× imbalance)
- A mean/reference line

**Red flags:**
- Only 11 arcs shown (missing the 3 arousal-primary arcs)
- Total n ≈ 300 (old sample file was used instead of the 10k file)

---

## fig4_tsne_embeddings.png

**What it is:** t-SNE of raw session embeddings (MiniLM), colored by arc type or valence.

**Should show:**
- 2D t-SNE scatter with weak structure — overlap expected for raw embeddings
- No perfect arc separation (would indicate data leakage)
- Some gradient if colored by valence

---

## fig5_valence_correlation.png

**What it is:** PC1 of raw MiniLM session embeddings correlated against ground-truth valence. This is an **encoder structure analysis**, NOT the trained probe output.

**Should show:**
- Scatter of PC1 vs ground-truth valence with Pearson r ≈ **0.477**
- A PCA variance-explained panel beside it showing how much variance PC1 captures
- This confirms the raw MiniLM embeddings already encode valence signal implicitly (r=0.477 before any transition model training)

**Notes — do NOT confuse with probe numbers:**
- The trained probe achieves valence r = **0.698**, arousal r = **0.542** — those numbers appear in **fig11** and **fig13**, not here
- r = 0.477 here is correct and expected for raw embeddings; it is NOT a regression from the probe's 0.698

**Red flags:**
- r > 0.7 (would be suspiciously high for raw embeddings)
- Figure described as "probe performance" (it is encoder structure analysis)

---

## fig6_loss_curves.png

**What it is:** Training curves comparing old vs current architecture.

**Should show:**
- **Left panel (Old Architecture — Joint LayerNorm, no VICReg, no VAD):**
  - Train loss decreasing, val loss diverging (overfitting)
  - "Best val 0.503 (epoch 4)" annotated
  - Both curves on same y-scale (~0.40–0.65)

- **Right panel (Current Architecture — Split LayerNorm + VICReg + VAD):**
  - **Dual y-axes** — train and val are on different scales (intentional)
  - Left y-axis: val cosine loss ~0.509–0.515 (red dashed, nearly flat)
  - Right y-axis: total VICReg train loss ~14.7–17.8 (blue, decreasing)
  - "Best val 0.509 (epoch 24)" annotated
  - "Final PR = 340/384" green box (bottom right)

**Why dual y-axis on right:** VICReg total train loss includes regularization terms (std + cov), pushing it to 14–18. Validation only measures cosine loss (~0.51). They are different quantities and cannot share an axis.

**Red flags:**
- Right panel with a single y-axis and train curve invisible/flat (means the bug was re-introduced)
- PR annotation missing from right panel

---

## fig7_state_trajectories.png

**What it is:** Learned state trajectories through PCA-reduced state space.

**Should show:**
- PC1 vs PC2 of state vectors evolving over sessions
- 3 arcs: gradual_decline (red), slow_recovery (green), grief_arc (dark)
- Trajectories in distinguishable directions with temporal color gradient
- PC1 variance explained should be **well under 80%** (e.g., 6–7%) — high PC1 dominance would indicate collapse

**Red flags:**
- PC1 explaining >80% of variance (old model collapsed at ~82.6%)
- All three trajectories converging to same point

---

## fig8_arc_separation.png

**What it is:** Arc separation analysis in the current model's state space (PR=339.7).

**Should show:**
- PCA scatter with all 14 arcs, heavy overlap between clusters
- PC1 ≈ 1.3%, PC2 ≈ 1.2% variance explained (labeled on axes)
- PR = 339.7 in title or annotation
- A note explaining why low separation is expected (concentration of measure in 340 effective dimensions)

**Notes — numbers NOT visible in figure:**
- The specific sep ratio 1.0085 (model) vs 0.9994 (shuffled-label control) are in the diagnostic output but not annotated on this figure
- "Only -0.9% collapse vs shuffled" is the key result but lives in PIPELINE.md / diagnostics, not the figure itself

**Red flags:**
- Large, visually obvious arc clusters with empty space between them (would indicate collapse to a low-dimensional space)
- PC1 >10% variance explained

---

## fig9_world_model_accuracy.png

**What it is:** World model prediction accuracy distribution.

**Should show:**
- Distribution(s) of cosine similarity between predicted and actual next state
- World model mean ≈ **0.491**, naive baseline mean ≈ **0.257**
- ~97% of predictions beat the naive baseline
- Per-arc accuracy bar chart

**Notes:**
- Figure shows one baseline (naive), not two. Guide originally expected last-session + EWMA — one baseline is fine.

**Red flags:**
- Model distribution to the left of (or equal to) baseline
- All cosine sims near 0 (model predicts orthogonal states)

---

## fig10_probe_generalization.png

**What it is:** Probe applied to 22 naturalistic companion AI scenarios.

**Should show:**
- Circumplex scatter of 22 scenario predictions
- Q1 scenarios (excited, happy) in upper-right, Q4 scenarios (grief, exhausted) in lower-left
- Valence spread ≈ **1.147** (annotated)
- Comparison bar: original model spread 0.034 → current 1.147 (34× improvement)
- Arousal visibly compressed (most points near/above 0) — known data ceiling, not a bug

**Red flags:**
- All 22 points clustered in a tiny region (spread < 0.3)
- Q1 scenarios predicting lower valence than Q4 scenarios

---

## fig11_probe_comparison.png

**What it is:** Probe r values across all 3 training runs, plus PR progression.

**Should show:**
- Grouped bars for 3 runs: valence r and arousal r per run
- Run 1 (PR=1): valence r ≈ **0.226**, arousal r ≈ small positive (NOT ~0 — even collapsed state retains some valence signal)
- Run 2 (PR=245): intermediate
- Run 3 (PR=339.7): valence r ≈ **0.698**, arousal r ≈ **0.542**
- Arousal r never exceeds valence r (arousal harder to learn)
- PR bar chart showing monotonic increase 1 → 245 → 340

**Red flags:**
- Run 3 lower than Run 2 (regression)
- PR not showing clear 100–300× jump from baseline

---

## fig12_longitudinal_curves.png

**What it is:** Simulated longitudinal tracking — MAE over session index + per-arc breakdown.

**Should show:**
- **Left panel (MAE by session):**
  - Sessions 0–1 colored red (cold-start), sessions 2+ green (warm) — boundary at session **1.5**
  - MAE dips at sessions 3–4 (down to ~0.24–0.26), then climbs back at sessions 5–8 (~0.30–0.38)
  - Cold MAE annotation ≈ **0.310**, Warm MAE ≈ **0.309**, Improvement ≈ **~0%**
  - This "flat" result is correct — the per-slot averaging gives equal weight to sparse late sessions

- **Right panel (per-arc MAE):**
  - All 14 arcs present
  - Best: anxiety_to_depression (~0.208), gradual_decline (~0.231)
  - Worst: excitement_to_contentment (~0.357), stable_negative (~0.337)
  - A vertical reference line at the warm mean

**Important: discrepancy with diagnostic (+11.2%)**

The diagnostic script reports cold=0.314, warm=0.278, +11.2%. The figure shows ~0%. Both are correct but measure different things:

- **Figure** (per-slot mean): averages the per-session-index MAEs. Each session slot (0, 1, 2, … 8) gets equal weight regardless of how many trajectories contributed to it. Sessions 7–8 have few data points but high MAE, pulling the warm mean up.
- **Diagnostic** (per-record mean): averages across all individual session records (314 cold records vs 450+ warm records). Late sessions have proportionally less weight. This gives +11.2%.

Neither is "wrong" — they ask different questions. The per-slot figure is more honest about model behavior at a specific session count; the per-record diagnostic is more representative of average user experience across the distribution.

**What the late-session MAE rise means:**

Sessions 7–8 MAE (~0.32–0.38) rising above sessions 3–4 (~0.24–0.26) is a real finding, not a bug:
- Very few val-set trajectories reach session 7–8, so that sample is biased toward unusual/hard arcs
- The model's state representation may accumulate prediction error over many steps (compounding drift)
- In real deployment, users with 7+ sessions may experience degraded emotional tracking

This is a meaningful caveat for real-world deployment: the warm-up benefit is real for sessions 2–5 but does not hold for very long user histories.

**Red flags:**
- Threshold line at 2.5 (old bug, now fixed to 1.5)
- Legend saying "sessions 0–2" for cold (fixed to "sessions 0–1")
- All 14 arcs missing from right panel

**Note on "10/10 arc-change detections":** This result (mean lag 0.7 sessions, all 10 transitions detected) comes from the diagnostic script output, not this figure. The figure does not visualize it.

---

## fig13_pr_progression.png

**What it is:** 4-panel summary across the 3 architecture runs.

**Should show:**
- Panel 1 (PR): 1.4 → 245 → 340 — massive jump (show log scale or just annotate clearly)
- Panel 2 (Val loss): ~0.503 → 0.506 → 0.509 — nearly flat (data ceiling, not regression)
- Panel 3 (Probe valence r): 0.226 → 0.693 → 0.698 — monotonic improvement
- Panel 4 (Naturalistic spread): 0.034 → (not measured) → 1.147

**Red flags:**
- Val loss panel showing a large downward trend (should be flat — large drop would indicate something else changed)
- PR panel where the 1.4 bar is invisible at linear scale without annotation

---

## fig_msc_validation.png

**What it is:** Multi-Session Chat (MSC) distribution-match validation. Tests whether the model's state dynamics on real MSC conversations match the synthetic training distribution.

**Should show:**
- State movement per step: real MSC mean ≈ 0.0179 vs synthetic training mean ≈ 0.0149
- Real is slightly higher but close — distributions broadly match
- Possibly a PCA trajectory plot of real MSC sessions

**Important note — PC1 on OOD data:**
- The left PCA panel shows PC1 ≈ 71.4% variance on MSC trajectories — much higher than in-distribution (PC1 ≈ 1.3–6.7% in fig7/fig8)
- This means on OOD real conversations, state vectors collapse onto a single dominant drift direction
- The model generalizes at the aggregate distribution level (similar step sizes) but loses multi-dimensional structure off-distribution

**This is NOT a zero-state ablation** — the guide originally described it as "state-informed vs zeroed-state comparison." The actual figure is a distribution-match check. The ablation result (state contributes 11.2% MAE improvement) is in fig12, not here.

**Red flags:**
- State movement mean on MSC >> 0.05 (states jumping around wildly on real data)
- State movement mean on MSC ≈ 0 (model producing static states on real conversations)

---

## Summary Table

| Filename | Key numbers to verify | Source |
|----------|----------------------|--------|
| fig3_arc_distribution.png | n = 10,000; 14 arc types | visualize_step1.py (10k file) |
| fig5_valence_correlation.png | PC1 vs valence r = **0.477** (raw MiniLM, NOT probe) | visualize_step2.py |
| fig6_loss_curves.png | dual y-axis right panel; best val 0.509 ep 24; PR = 340/384 | visualize_step3.py |
| fig8_arc_separation.png | PC1 = 1.3%, PR = 339.7; heavy overlap (sep ratio in diagnostics, not on figure) | visualize_step3.py |
| fig10_probe_generalization.png | valence spread 1.147 vs 0.034 original | visualize_step4.py |
| fig11_probe_comparison.png | valence r: 0.226 → 0.698; arousal r: 0.542 | visualize_step4.py |
| fig12_longitudinal_curves.png | threshold at session **1.5**; figure shows ~0% per-slot; diagnostic +11.2% per-record (different averaging) | visualize_step4.py |
| fig13_pr_progression.png | PR: 1.4 → 245 → 340; val loss flat | visualize_step4.py |
| fig_msc_validation.png | MSC step 0.0179 ≈ synthetic 0.0149; PC1=71.4% on OOD | validate_msc.py |
