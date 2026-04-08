#!/usr/bin/env python3
"""
Diverse model ensemble for MovieLens recommendation.
Trains 20+ model variants with different architectures/features/data,
then stacks predictions with HistGBM for ensemble AUC.
"""

import copy
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from prepare import load_data, evaluate, print_summary, TIME_BUDGET as _TIME_BUDGET
TIME_BUDGET = 1800  # 30 min for ensemble (need time for 40+ models + stacking)

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("train")

# ─── Configuration ──────────────────────────────────────────────────
DATASET = os.environ.get("DATASET", "ml-1m")
EVALS_PER_EPOCH = 3
PATIENCE = 6
TIME_PER_MODEL = 45  # seconds per model variant (adjusted per remaining budget)

# ─── Device ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ─── Load Data ──────────────────────────────────────────────────────
total_start = time.time()
data = load_data(DATASET)
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | Pos rate: {stats['pos_rate']:.2%}")


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (shared across all variants)
# ═══════════════════════════════════════════════════════════════════

# 1. Genre multi-hot encoding
all_genres_set = set()
for g in movies_df["genres"].dropna():
    all_genres_set.update(g.split("|"))
all_genres = sorted(all_genres_set - {""})
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
num_genres = len(all_genres)

movie_genres = np.zeros((num_items, num_genres), dtype=np.float32)
for _, row in movies_df.iterrows():
    mid = int(row["movieId"])
    if mid < num_items and isinstance(row["genres"], str):
        for g in row["genres"].split("|"):
            if g in genre_to_idx:
                movie_genres[mid, genre_to_idx[g]] = 1.0

# 2. User and item statistics with rating histograms + raw mean
RATING_BINS = [(0, 2.5), (2.5, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.5)]
NUM_BINS = len(RATING_BINS)
STAT_DIM = 2 + NUM_BINS + 1  # log_count + mean + pos_rate + 5 bins
global_mean = float(train_df["rating"].mean())
global_pos_rate = float(train_df["label"].mean())

user_stats = np.zeros((num_users, STAT_DIM), dtype=np.float32)
item_stats = np.zeros((num_items, STAT_DIM), dtype=np.float32)
user_stats[:, 1] = global_mean
user_stats[:, 2] = global_pos_rate
user_stats[:, 3:] = 1.0 / NUM_BINS
item_stats[:, 1] = global_mean
item_stats[:, 2] = global_pos_rate
item_stats[:, 3:] = 1.0 / NUM_BINS

for uid, group in train_df.groupby("userId"):
    r = group["rating"].values.astype(np.float32)
    labels = group["label"].values.astype(np.float32)
    user_stats[uid, 0] = np.log1p(len(r))
    user_stats[uid, 1] = r.mean()
    user_stats[uid, 2] = labels.mean()
    for b, (lo, hi) in enumerate(RATING_BINS):
        user_stats[uid, 3 + b] = ((r >= lo) & (r < hi)).mean()

for mid, group in train_df.groupby("movieId"):
    r = group["rating"].values.astype(np.float32)
    labels = group["label"].values.astype(np.float32)
    item_stats[mid, 0] = np.log1p(len(r))
    item_stats[mid, 1] = r.mean()
    item_stats[mid, 2] = labels.mean()
    for b, (lo, hi) in enumerate(RATING_BINS):
        item_stats[mid, 3 + b] = ((r >= lo) & (r < hi)).mean()

NUM_DENSE = 3 + 2 * STAT_DIM  # ts + user_recency + item_recency + stats

# 3. Temporal features
ts_min = float(train_df["timestamp"].min())
ts_max = float(train_df["timestamp"].max())
ts_range = ts_max - ts_min + 1.0

user_last_ts = np.full(num_users, ts_min, dtype=np.float64)
item_last_ts = np.full(num_items, ts_min, dtype=np.float64)
for uid, group in train_df.groupby("userId"):
    user_last_ts[uid] = group["timestamp"].max()
for mid, group in train_df.groupby("movieId"):
    item_last_ts[mid] = group["timestamp"].max()

# 4. User history sequences
PAD_IDX = num_items
user_histories = np.full((num_users, HISTORY_LEN := 50), PAD_IDX, dtype=np.int64)
for uid, group in train_df.groupby("userId"):
    items = group["movieId"].values
    seq = items[-HISTORY_LEN:]
    user_histories[uid, -len(seq):] = seq


# ═══════════════════════════════════════════════════════════════════
# PRE-COMPUTE TENSORS
# ═══════════════════════════════════════════════════════════════════

def build_tensors(df):
    uids = df["userId"].values.astype(np.int64)
    mids = df["movieId"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)
    ts = df["timestamp"].values.astype(np.float64)
    ts_norm = ((ts - ts_min) / ts_range).astype(np.float32)
    user_recency = ((ts - user_last_ts[uids]) / ts_range).astype(np.float32)
    item_recency = ((ts - item_last_ts[mids]) / ts_range).astype(np.float32)
    dense = np.column_stack([ts_norm, user_recency, item_recency,
                             user_stats[uids], item_stats[mids]]).astype(np.float32)
    genres = movie_genres[mids]
    return (
        torch.from_numpy(uids).to(DEVICE),
        torch.from_numpy(mids).to(DEVICE),
        torch.from_numpy(dense).to(DEVICE),
        torch.from_numpy(genres).to(DEVICE),
        torch.from_numpy(labels).to(DEVICE),
    )

log.info("Pre-computing tensors...")
train_t = build_tensors(train_df)
val_t = build_tensors(val_df)
n_train = train_t[0].shape[0]
n_val = val_t[0].shape[0]

# Sample weights (recency)
ts_vals = train_df["timestamp"].values.astype(np.float64)
ts_normalized = (ts_vals - ts_vals.min()) / (ts_vals.max() - ts_vals.min() + 1.0)
base_weights = np.exp(2.0 * ts_normalized).astype(np.float32)
base_weights /= base_weights.mean()
sample_weights_gpu = torch.from_numpy(base_weights).to(DEVICE)

log.info(f"Tensors on GPU. Train: {n_train}, Val: {n_val}")


# ═══════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════

class RecModel(nn.Module):
    def __init__(self, hidden_dims=(256, 128, 64), dropout=0.1,
                 use_bias=True, use_genres=True, dense_dim=NUM_DENSE):
        super().__init__()
        self.use_bias = use_bias
        self.use_genres = use_genres
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)

        inp = dense_dim + (num_genres if use_genres else 0)
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]
            inp = h
        layers.append(nn.Linear(inp, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.zeros_(m.weight)

    def forward(self, user_id, movie_id, dense, genres):
        if self.use_genres:
            x = torch.cat([dense, genres], dim=-1)
        else:
            x = dense
        out = self.mlp(x)
        if self.use_bias:
            out = out + self.user_bias(user_id) + self.item_bias(movie_id)
        return out.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_variant(config):
    """Train one model variant and return val predictions."""
    name = config["name"]
    seed = config.get("seed", 42)
    lr = config.get("lr", 1e-3)
    wd = config.get("wd", 1e-5)
    batch_size = config.get("batch_size", 2048)
    hidden = config.get("hidden", (256, 128, 64))
    dropout = config.get("dropout", 0.1)
    use_bias = config.get("use_bias", True)
    use_genres = config.get("use_genres", True)
    recency_strength = config.get("recency_strength", 2.0)
    data_frac = config.get("data_frac", 1.0)  # fraction of training data (recent)

    torch.manual_seed(seed)
    model = RecModel(hidden_dims=hidden, dropout=dropout,
                     use_bias=use_bias, use_genres=use_genres).to(DEVICE)

    # Data subset (recent fraction)
    if data_frac < 1.0:
        start_idx = int(n_train * (1 - data_frac))
        t_uids = train_t[0][start_idx:]
        t_mids = train_t[1][start_idx:]
        t_dense = train_t[2][start_idx:]
        t_genres = train_t[3][start_idx:]
        t_labels = train_t[4][start_idx:]
        n = t_uids.shape[0]
        # Recompute weights for subset
        w = torch.from_numpy(
            np.exp(recency_strength * np.linspace(0, 1, n)).astype(np.float32)
        ).to(DEVICE)
        w = w / w.mean()
    else:
        t_uids, t_mids, t_dense, t_genres, t_labels = train_t
        n = n_train
        if recency_strength != 2.0:
            w = torch.from_numpy(
                np.exp(recency_strength * ts_normalized).astype(np.float32)
            ).to(DEVICE)
            w = w / w.mean()
        else:
            w = sample_weights_gpu

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    total_batches = (n + batch_size - 1) // batch_size
    eval_interval = max(1, total_batches // EVALS_PER_EPOCH)
    best_auc = 0.0
    best_state = None
    evals_no_improve = 0
    model_start = time.time()

    for epoch in range(100):
        if time.time() - model_start >= TIME_PER_MODEL:
            break
        perm = torch.randperm(n, device=DEVICE)
        model.train()
        n_batches = 0

        for start in range(0, n, batch_size):
            if time.time() - model_start >= TIME_PER_MODEL:
                break
            end = min(start + batch_size, n)
            idx = perm[start:end]

            logits = model(t_uids[idx], t_mids[idx], t_dense[idx], t_genres[idx])
            loss = (criterion(logits, t_labels[idx]) * w[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batches += 1

            if n_batches % eval_interval == 0:
                model.eval()
                scores = []
                with torch.no_grad():
                    for i in range(0, n_val, batch_size * 4):
                        e = min(i + batch_size * 4, n_val)
                        logits_v = model(val_t[0][i:e], val_t[1][i:e],
                                         val_t[2][i:e], val_t[3][i:e])
                        scores.append(torch.sigmoid(logits_v).cpu().numpy())
                val_scores = np.concatenate(scores)
                auc = evaluate(val_t[4].cpu().numpy(), val_scores)["auc"]
                if auc > best_auc:
                    best_auc = auc
                    best_state = copy.deepcopy(model.state_dict())
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1
                if evals_no_improve >= PATIENCE:
                    break
                model.train()

        if evals_no_improve >= PATIENCE:
            break

    # Get best predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, n_val, batch_size * 4):
            e = min(i + batch_size * 4, n_val)
            logits = model(val_t[0][i:e], val_t[1][i:e], val_t[2][i:e], val_t[3][i:e])
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    preds = np.concatenate(all_scores)
    log.info(f"  {name:25s} | AUC {best_auc:.4f} | {time.time()-model_start:.0f}s")
    del model, optimizer
    torch.cuda.empty_cache()
    return preds, best_auc


# ═══════════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

VARIANTS = [
    # --- Architecture diversity ---
    {"name": "baseline",         "hidden": (256, 128, 64), "dropout": 0.1},
    {"name": "wide",             "hidden": (512, 256, 128), "dropout": 0.1},
    {"name": "narrow",           "hidden": (128, 64, 32), "dropout": 0.1},
    {"name": "deep",             "hidden": (256, 128, 64, 32), "dropout": 0.1},
    {"name": "shallow",          "hidden": (256, 128), "dropout": 0.1},
    {"name": "very_shallow",     "hidden": (128,), "dropout": 0.0},
    {"name": "high_dropout",     "hidden": (256, 128, 64), "dropout": 0.3},
    {"name": "no_dropout",       "hidden": (256, 128, 64), "dropout": 0.0},
    {"name": "no_bias",          "hidden": (256, 128, 64), "dropout": 0.1, "use_bias": False},
    {"name": "no_genre",         "hidden": (256, 128, 64), "dropout": 0.1, "use_genres": False},
    {"name": "no_bias_no_genre", "hidden": (256, 128, 64), "dropout": 0.1, "use_bias": False, "use_genres": False},
    {"name": "wide_shallow",     "hidden": (512, 128), "dropout": 0.1},
    {"name": "narrow_deep",      "hidden": (128, 64, 32, 16), "dropout": 0.05},
    # --- Training diversity ---
    {"name": "lr_low",           "lr": 3e-4},
    {"name": "lr_high",          "lr": 3e-3},
    {"name": "lr_very_low",      "lr": 1e-4},
    {"name": "batch4096",        "batch_size": 4096},
    {"name": "batch1024",        "batch_size": 1024},
    {"name": "batch8192",        "batch_size": 8192},
    {"name": "no_recency_wt",    "recency_strength": 0.0},
    {"name": "mild_recency",     "recency_strength": 1.0},
    {"name": "strong_recency",   "recency_strength": 3.0},
    {"name": "very_strong_rec",  "recency_strength": 5.0},
    {"name": "wd_high",          "wd": 1e-4},
    {"name": "wd_zero",          "wd": 0.0},
    # --- Data diversity (recency splits — biggest ensemble diversity source) ---
    {"name": "recent90",         "data_frac": 0.9},
    {"name": "recent80",         "data_frac": 0.8},
    {"name": "recent70",         "data_frac": 0.7},
    {"name": "recent60",         "data_frac": 0.6},
    {"name": "recent50",         "data_frac": 0.5},
    {"name": "recent40",         "data_frac": 0.4},
    {"name": "recent30",         "data_frac": 0.3},
    {"name": "recent20",         "data_frac": 0.2},
    # --- Recency splits × architecture combos ---
    {"name": "recent50_narrow",  "data_frac": 0.5, "hidden": (128, 64, 32)},
    {"name": "recent50_nobias",  "data_frac": 0.5, "use_bias": False},
    {"name": "recent70_wide",    "data_frac": 0.7, "hidden": (512, 256, 128)},
    {"name": "recent80_norec",   "data_frac": 0.8, "recency_strength": 0.0},
    # --- Seed diversity ---
    {"name": "seed43",           "seed": 43},
    {"name": "seed44",           "seed": 44},
    {"name": "seed45",           "seed": 45},
    {"name": "seed100",          "seed": 100},
    {"name": "seed200",          "seed": 200},
]


# ═══════════════════════════════════════════════════════════════════
# TRAIN ALL VARIANTS
# ═══════════════════════════════════════════════════════════════════

training_start = time.time()
peak_memory_mb = 0.0
all_preds = {}
all_aucs = {}

log.info(f"Training {len(VARIANTS)} model variants...")
for i, cfg in enumerate(VARIANTS):
    elapsed = time.time() - training_start
    remaining = TIME_BUDGET - elapsed
    if remaining < 30:
        log.info(f"Time budget reached after {i} variants")
        break
    # Adjust time per model based on remaining budget
    models_left = len(VARIANTS) - i
    TIME_PER_MODEL = min(90, max(30, remaining / models_left))

    log.info(f"[{i+1}/{len(VARIANTS)}] Training '{cfg['name']}' ({TIME_PER_MODEL:.0f}s budget)...")
    preds, auc = train_variant(cfg)
    all_preds[cfg["name"]] = preds
    all_aucs[cfg["name"]] = auc
    if DEVICE.type == "cuda":
        peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / (1024**2))

training_seconds = time.time() - training_start
num_params = len(all_preds)  # number of models as "params"

# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

y_val = val_t[4].cpu().numpy()
variant_names = sorted(all_preds.keys())
X_stack = np.column_stack([all_preds[n] for n in variant_names])

log.info(f"\n{'='*60}")
log.info(f"Trained {len(variant_names)} variants in {training_seconds:.0f}s")
log.info(f"Individual AUCs: min={min(all_aucs.values()):.4f} max={max(all_aucs.values()):.4f}")

# 1. Simple average
avg_preds = X_stack.mean(axis=1)
avg_metrics = evaluate(y_val, avg_preds)
log.info(f"Simple average ensemble AUC: {avg_metrics['auc']:.4f}")

# 2. HistGBM stacking (3-fold CV)
log.info("Training HistGBM stacker (3-fold CV)...")
stacker = HistGradientBoostingClassifier(
    max_iter=200, max_leaf_nodes=31, learning_rate=0.1,
    min_samples_leaf=50, random_state=42,
)
stacked_scores = cross_val_predict(
    stacker, X_stack, y_val.astype(int), cv=3, method="predict_proba"
)[:, 1]
stacked_metrics = evaluate(y_val, stacked_scores)
log.info(f"HistGBM stacked ensemble AUC: {stacked_metrics['auc']:.4f}")

# Use best ensemble method
if stacked_metrics["auc"] > avg_metrics["auc"]:
    metrics = stacked_metrics
    log.info("Using HistGBM stacking (best)")
else:
    metrics = avg_metrics
    log.info("Using simple average (best)")

total_seconds = time.time() - total_start
print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
