#!/usr/bin/env python3
"""
DLRM baseline for MovieLens pointwise recommendation.
This is the file you modify for experimentation.

Features:
  - Sparse: userId, movieId, genre (multi-hot), user history sequence
  - Dense: normalized timestamp, user stats (count, mean, std), item stats (count, mean, std)

Model: DLRM with bottom MLP, embedding interaction (pairwise dots), top MLP.
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
from prepare import load_data, evaluate, print_summary, TIME_BUDGET

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
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM = 16
HISTORY_LEN = 50
NUM_DENSE = 7  # 1 timestamp + 3 user stats + 3 item stats
EVALS_PER_EPOCH = 3  # sub-epoch evaluation
PATIENCE = 6  # early stop after N evals with no improvement

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
# FEATURE ENGINEERING
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

# 2. User and item statistics (computed on train only)
user_stats = np.zeros((num_users, 3), dtype=np.float32)
item_stats = np.zeros((num_items, 3), dtype=np.float32)

for uid, group in train_df.groupby("userId"):
    r = group["rating"].values.astype(np.float32)
    user_stats[uid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for mid, group in train_df.groupby("movieId"):
    r = group["rating"].values.astype(np.float32)
    item_stats[mid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for arr in [user_stats, item_stats]:
    for col in range(arr.shape[1]):
        mu = arr[:, col].mean()
        sigma = arr[:, col].std() + 1e-8
        arr[:, col] = (arr[:, col] - mu) / sigma

# 3. User history sequences
PAD_IDX = num_items
user_histories = np.full((num_users, HISTORY_LEN), PAD_IDX, dtype=np.int64)
for uid, group in train_df.groupby("userId"):
    items = group["movieId"].values
    seq = items[-HISTORY_LEN:]
    user_histories[uid, -len(seq):] = seq

# 4. Timestamp normalization
ts_min = float(train_df["timestamp"].min())
ts_range = float(train_df["timestamp"].max() - ts_min) + 1.0


# ═══════════════════════════════════════════════════════════════════
# PRE-COMPUTE & MOVE ALL TENSORS TO GPU
# ═══════════════════════════════════════════════════════════════════

def build_tensors(df):
    uids = df["userId"].values.astype(np.int64)
    mids = df["movieId"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)
    ts_norm = ((df["timestamp"].values - ts_min) / ts_range).astype(np.float32)
    dense = np.column_stack([ts_norm, user_stats[uids], item_stats[mids]]).astype(np.float32)
    hist = user_histories[uids]
    genres = movie_genres[mids]
    return (
        torch.from_numpy(uids).to(DEVICE),
        torch.from_numpy(mids).to(DEVICE),
        torch.from_numpy(dense).to(DEVICE),
        torch.from_numpy(hist).to(DEVICE),
        torch.from_numpy(genres).to(DEVICE),
        torch.from_numpy(labels).to(DEVICE),
    )

log.info("Pre-computing and moving tensors to GPU...")
train_t = build_tensors(train_df)
val_t = build_tensors(val_df)
n_train = train_t[0].shape[0]
n_val = val_t[0].shape[0]
log.info(f"Tensors on GPU. Train: {n_train}, Val: {n_val}")


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class DLRM(nn.Module):
    def __init__(self):
        super().__init__()
        D = EMBED_DIM
        self.user_embed = nn.Embedding(num_users, D)
        self.item_embed = nn.Embedding(num_items, D)
        self.hist_embed = nn.Embedding(num_items + 1, D, padding_idx=PAD_IDX)
        self.bottom_mlp = nn.Sequential(
            nn.Linear(NUM_DENSE, 64), nn.ReLU(),
            nn.Linear(64, D), nn.ReLU(),
        )
        self.genre_proj = nn.Sequential(nn.Linear(num_genres, D), nn.ReLU())
        self.top_mlp = nn.Sequential(
            nn.Linear(10 + 5 * D, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, user_id, movie_id, dense, history, genres):
        user_e = self.user_embed(user_id)
        item_e = self.item_embed(movie_id)
        hist_e = self.hist_embed(history)
        mask = (history != PAD_IDX).unsqueeze(-1).float()
        hist_e = (hist_e * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        dense_e = self.bottom_mlp(dense)
        genre_e = self.genre_proj(genres)

        vecs = [user_e, item_e, hist_e, dense_e, genre_e]
        dots = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dots.append((vecs[i] * vecs[j]).sum(dim=-1, keepdim=True))

        out = torch.cat(dots + vecs, dim=-1)
        return self.top_mlp(out).squeeze(-1)


model = DLRM().to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
log.info(f"Parameters: {num_params / 1e6:.1f}M | Genres: {num_genres}")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_eval():
    model.eval()
    all_scores = []
    for i in range(0, n_val, BATCH_SIZE * 4):
        end = min(i + BATCH_SIZE * 4, n_val)
        logits = model(val_t[0][i:end], val_t[1][i:end], val_t[2][i:end],
                       val_t[3][i:end], val_t[4][i:end])
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
    return evaluate(val_t[5].cpu().numpy(), np.concatenate(all_scores))


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()

training_start = time.time()
peak_memory_mb = 0.0
epoch = 0
best_auc = 0.0
best_state = None
evals_without_improvement = 0
total_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
eval_interval = max(1, total_batches // EVALS_PER_EPOCH)

while True:
    if time.time() - training_start >= TIME_BUDGET:
        break

    perm = torch.randperm(n_train, device=DEVICE)
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, n_train, BATCH_SIZE):
        if time.time() - training_start >= TIME_BUDGET:
            break

        end = min(start + BATCH_SIZE, n_train)
        idx = perm[start:end]

        logits = model(train_t[0][idx], train_t[1][idx], train_t[2][idx],
                       train_t[3][idx], train_t[4][idx])
        loss = criterion(logits, train_t[5][idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

        if n_batches % eval_interval == 0:
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / (1024**2))
            avg_loss = epoch_loss / n_batches
            val_metrics = run_eval()
            val_auc = val_metrics["auc"]
            elapsed = time.time() - training_start
            improved = "***" if val_auc > best_auc else ""
            log.info(f"Epoch {epoch+1} batch {n_batches}/{total_batches} | "
                     f"Loss {avg_loss:.4f} | Val AUC {val_auc:.4f} {improved} | "
                     f"{elapsed:.0f}s / {TIME_BUDGET}s")
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            if evals_without_improvement >= PATIENCE:
                log.info(f"Early stopping: {PATIENCE} evals without improvement (best: {best_auc:.4f})")
                break
            model.train()

    if evals_without_improvement >= PATIENCE:
        break

    epoch += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    elapsed = time.time() - training_start
    log.info(f"Epoch {epoch} done | Loss {avg_loss:.4f} | {elapsed:.0f}s / {TIME_BUDGET}s")

training_seconds = time.time() - training_start


# ═══════════════════════════════════════════════════════════════════
# EVALUATION (do not modify — uses prepare.evaluate)
# ═══════════════════════════════════════════════════════════════════

if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model (AUC: {best_auc:.4f})")

metrics = run_eval()
total_seconds = time.time() - total_start

print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
