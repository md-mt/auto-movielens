# Autoresearch Report — apr07 Run

**Date:** 2026-04-07 to 2026-04-08
**Dataset:** ml-25m (20M train, 2.5M val, time-based split)
**Hardware:** 1x NVIDIA A100 80GB
**Total experiments:** 44 (10 kept, 33 discarded, 1 crash)
**Best AUC:** 0.707 (42-model ensemble with HistGBM stacking)
**Best single model:** 0.696

## AUC Progression

```
0.587  baseline (broken — training too slow for ml-25m)
  │
0.589  all-GPU tensor precomputation (speed fix)
  │
0.675  fix z-score normalization (+0.086, biggest single gain)
  │
0.685  rating histogram features (+0.010)
  │
0.691  raw mean_rating + pos_rate features (+0.006)
  │
0.693  user/item bias embeddings (+0.002)
  │
0.694  temporal recency features (+0.001)
  │
0.696  recency-weighted training, exp(2t) (+0.002)
  │
0.706  19-model diverse ensemble + HistGBM stacking (+0.010)
  │
0.707  42-model diverse ensemble + HistGBM stacking (+0.001)
```

## Phase 1: Single Model (Experiments 1-41)

### What worked

| Experiment | AUC | Delta | Key insight |
|-----------|-----|-------|-------------|
| GPU tensor precompute | 0.589 | +0.002 | All tensors on GPU, randperm shuffling — 10 epochs/600s |
| Fix normalization | 0.675 | +0.086 | z-score included 42% zero-entry items, destroying signal |
| Rating histograms | 0.685 | +0.010 | 5-bin distributions capture full rating shape |
| Mean + pos_rate | 0.691 | +0.006 | Direct positive rate as feature, not just histograms |
| User/item bias | 0.693 | +0.002 | Scalar bias embeddings, zero-initialized |
| Recency features | 0.694 | +0.001 | Time since user/item's last train interaction |
| Recency weighting | 0.696 | +0.002 | exp(2t) sample weights focus on recent patterns |

### What didn't work (33 failed experiments)

**Embeddings / collaborative filtering (all hurt or neutral):**
- User/item full embeddings — can't converge in 600s, add noise
- DIN attention for user history — no lift over dense features
- CF dot product with separate low LR — still hurts
- SVD latent factors — only 0.549 AUC, temporal drift kills CF signal

**Architecture changes (all neutral):**
- Wider MLP (512-256-128) — overfits
- Residual blocks — no gain
- BatchNorm — no gain
- Wide & Deep with concatenation — deep path drowns wide path
- Deep path with separate LR — no improvement

**Feature engineering (most hurt):**
- Genre match (user_genre_pref · movie_genres) — hurts AUC
- User genre preference vector — hurts
- Tag genome PCA-32 — PCA loses signal
- Tag genome learned MLP compression — only 23% item coverage, slow CPU→GPU
- Cross-product features — no gain
- Movie release year — no gain
- Popularity trends — hurts
- Recent-half stats — adds noise

**Loss / training procedure (all neutral):**
- Focal loss (gamma=2) — same as BCE
- Label smoothing (0.05/0.95) — no gain
- Multi-task BCE + MSE on rating — slightly worse
- AdamW + cosine LR schedule — no improvement
- Negative sampling (hybrid training) — task mismatch with val
- Training on recent 50% only — slightly worse

### Key discoveries

1. **Normalization matters enormously.** z-scoring across all entries (including 42% items and 15% users with zero interactions) destroyed the informative scale of user_mean and item_mean. This was the single largest bug fix (+0.086 AUC).

2. **Dense feature ceiling is real.** The trivial predictor `user_mean + item_mean` gives 0.693 AUC. Our model barely exceeds this (0.696). All the useful information is in per-user and per-item aggregate statistics.

3. **Collaborative filtering fails on temporal splits.** SVD gives 0.549 AUC. User/item embeddings consistently hurt. Past interaction patterns don't predict future preferences in ml-25m's time-based split.

4. **Architecture/loss changes don't help once features plateau.** 15+ architecture and training procedure experiments all stayed within ±0.003 of 0.694.

## Phase 2: Ensemble (Experiments 42-44)

### Approach
Train 42 architecturally diverse model variants, then stack predictions with HistGradientBoostingClassifier (3-fold CV).

### Diversity sources

| Category | Variants | Example configs |
|----------|----------|----------------|
| Architecture | 13 | MLP widths (128 to 512), depths (1 to 4 layers), dropout (0.0 to 0.3) |
| Feature ablation | 3 | No bias, no genre, no bias + no genre |
| Learning rate | 4 | 1e-4, 3e-4, 1e-3, 3e-3 |
| Batch size | 3 | 1024, 2048, 4096, 8192 |
| Recency weighting | 4 | exp(0t) to exp(5t) |
| Data fraction | 8 | Recent 20% to 90% of training data |
| Cross-combinations | 4 | recent50+narrow, recent50+nobias, etc. |
| Random seeds | 5 | Seeds 42-45, 100, 200 |

### Results

| Method | AUC | Models | Time |
|--------|-----|--------|------|
| Best single model | 0.696 | 1 | 60s |
| Simple average (19) | 0.695 | 19 | 580s |
| **HistGBM stack (19)** | **0.706** | 19 | 580s + 210s stacking |
| Simple average (42) | 0.697 | 42 | 1800s |
| **HistGBM stack (42)** | **0.707** | 42 | 1800s + 280s stacking |

### Key findings

1. **Simple averaging barely helps** (0.695-0.697) — models are too correlated because they use the same feature set.
2. **HistGBM stacking is the breakthrough** (+0.010 over single model) — learns which model to trust for which samples.
3. **More models help with diminishing returns** — 19→42 models only added +0.001 AUC.
4. **Individual model AUCs range 0.686-0.696** — even "worse" models contribute to ensemble diversity.

## Comparison with Xu Ning's Prior Research

| Metric | Ning (500 exps) | This run (44 exps) |
|--------|----------------|-------------------|
| Baseline | 0.770 | 0.587 (broken) |
| Single model best | 0.821 | 0.696 |
| Ensemble best | 0.854 | 0.707 |
| Key architecture | DLRM + DIN + GDCN + genome | Dense MLP + bias |
| Ensemble size | 59 models | 42 models |
| Stacking method | HistGBM | HistGBM |

**Gap analysis:** The 0.696 vs 0.821 single-model gap comes from missing embeddings, DIN attention, GDCN cross layers, and tag genome features — all of which require longer training time to converge. Ning used 2x L4 GPUs with longer time budgets.

## Next steps to close the gap

1. **Increase TIME_BUDGET to 1800s+ for single model training** — embeddings need more epochs to converge
2. **Restore DLRM architecture** with our normalization fixes — should recover 0.770+ baseline
3. **Add DIN + GDCN + genome** on top of fixed features — should reach 0.82+
4. **Re-ensemble with improved base model** — better base lifts all 42+ variants
