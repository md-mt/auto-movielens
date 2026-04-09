#!/usr/bin/env python3
"""
DLRM for hybrid engagement prediction on MovieLens.

Label scheme:
  - 1: user rated >= 4 (watched and liked)
  - 0: user rated < 4 (hard negative) OR random unrated movie (easy negative)

Loss: BCE (calibrated probabilities for threshold-based serving).
"""

import copy
import logging
import os
import sys
from pathlib import Path
import time

import numpy as np

# Fix all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prepare import load_data_hybrid, evaluate, print_summary

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
DATASET = os.environ.get("DATASET", "ml-25m")
BATCH_SIZE = 16384
LR = 1e-4
WEIGHT_DECAY = 5e-5
EMBED_DIM = 28
HISTORY_LEN = 100
NUM_DENSE = 17  # 1 timestamp + 5 user hist bins + 1 user count + 5 item hist bins + 1 item count + 1 ug_dot + 1 year + 1 genre_count + 1 movie_age
NEG_RATIO = 1  # random unrated negatives per positive in training data
EVAL_EVERY = 1
PATIENCE = 3

# ─── Device ─────────────────────────────────────────────────────────
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')  # enable TF32 tensor cores
log.info(f"Device: {DEVICE}")

# ─── Load Data ──────────────────────────────────────────────────────
total_start = time.time()
data = load_data_hybrid(DATASET, neg_ratio=NEG_RATIO)
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
user_all_items = data["user_all_items"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | "
         f"Pos rate: {stats['pos_rate']:.2%}")


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (with disk caching)
# ═══════════════════════════════════════════════════════════════════

import hashlib, json
_cache_key = hashlib.md5(json.dumps({
    "dataset": DATASET, "neg_ratio": NEG_RATIO,
    "history_len": HISTORY_LEN, "num_users": num_users, "num_items": num_items,
}).encode()).hexdigest()[:12]
_cache_path = Path(__file__).parent / "data" / f"features_{_cache_key}.npz"

if _cache_path.exists():
    log.info(f"Loading cached features from {_cache_path.name}...")
    _cache = np.load(_cache_path, allow_pickle=True)
    movie_genres = _cache["movie_genres"]
    num_genres = movie_genres.shape[1]
    user_hist_bins = _cache["user_hist_bins"]
    item_hist_bins = _cache["item_hist_bins"]
    user_count_norm = _cache["user_count_norm"]
    item_count_norm = _cache["item_count_norm"]
    user_histories = _cache["user_histories"]
    user_hist_ratings = _cache["user_hist_ratings"]
    item_histories = _cache["item_histories"]
    item_hist_ratings = _cache["item_hist_ratings"]
    user_genre_affinity = _cache["user_genre_affinity"]
    ts_min = float(_cache["ts_min"])
    ts_range = float(_cache["ts_range"])
    movie_year = _cache["movie_year"]
    movie_genre_count = _cache["movie_genre_count"]
    genome_matrix = _cache["genome_matrix"]
    has_genome = _cache["has_genome"]
    GENOME_DIM = genome_matrix.shape[1]
    PAD_IDX = num_items
    USER_PAD_IDX = num_users
    ITEM_HIST_LEN = item_histories.shape[1]
    del _cache
    log.info("Cached features loaded.")
else:
    log.info("Computing features (will cache for next run)...")
    real_train = train_df[train_df["rating"] > 0]

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

    # 2. Rating histograms (vectorized)
    _rt_uids = real_train["userId"].values
    _rt_mids = real_train["movieId"].values
    _rt_ratings = real_train["rating"].values
    _rt_bins = np.clip((_rt_ratings - 0.01) // 1, 0, 4).astype(np.intp)
    user_hist_bins = np.zeros((num_users, 5), dtype=np.float32)
    item_hist_bins = np.zeros((num_items, 5), dtype=np.float32)
    np.add.at(user_hist_bins, (_rt_uids, _rt_bins), 1)
    np.add.at(item_hist_bins, (_rt_mids, _rt_bins), 1)
    user_count = user_hist_bins.sum(axis=1)
    item_count = item_hist_bins.sum(axis=1)
    user_hist_bins = user_hist_bins / (user_count[:, None] + 1e-8)
    item_hist_bins = item_hist_bins / (item_count[:, None] + 1e-8)
    user_count_norm = (user_count - user_count.mean()) / (user_count.std() + 1e-8)
    item_count_norm = (item_count - item_count.mean()) / (item_count.std() + 1e-8)
    del _rt_bins

    # 3. User/item history sequences (vectorized)
    PAD_IDX = num_items
    USER_PAD_IDX = num_users
    ITEM_HIST_LEN = 30

    def _build_history_arrays(ids, targets, ratings, timestamps, num_entities, pad_idx, max_len):
        sort_idx = np.lexsort((timestamps, ids))
        s_ids, s_targets = ids[sort_idx], targets[sort_idx]
        s_ratings = ratings[sort_idx].astype(np.float32) / 5.0
        hist = np.full((num_entities, max_len), pad_idx, dtype=np.int64)
        hist_rat = np.zeros((num_entities, max_len), dtype=np.float32)
        boundaries = np.where(np.diff(s_ids) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(s_ids)]])
        entity_ids = s_ids[starts]
        for i in range(len(entity_ids)):
            eid, s, e = entity_ids[i], starts[i], ends[i]
            length = min(e - s, max_len)
            hist[eid, -length:] = s_targets[e - length:e]
            hist_rat[eid, -length:] = s_ratings[e - length:e]
        return hist, hist_rat

    _rt_ts = real_train["timestamp"].values
    user_histories, user_hist_ratings = _build_history_arrays(
        _rt_uids, _rt_mids, _rt_ratings, _rt_ts, num_users, PAD_IDX, HISTORY_LEN)
    item_histories, item_hist_ratings = _build_history_arrays(
        _rt_mids, _rt_uids, _rt_ratings, _rt_ts, num_items, USER_PAD_IDX, ITEM_HIST_LEN)

    # 4. User-genre affinity
    user_genre_affinity = np.zeros((num_users, num_genres), dtype=np.float32)
    user_genre_count = np.zeros((num_users, num_genres), dtype=np.float32)
    _genres_of_items = movie_genres[_rt_mids]
    np.add.at(user_genre_affinity, _rt_uids, _rt_ratings[:, None].astype(np.float32) * _genres_of_items)
    np.add.at(user_genre_count, _rt_uids, _genres_of_items)
    del _rt_uids, _rt_mids, _rt_ratings, _rt_ts, _genres_of_items
    mask = user_genre_count > 0
    user_genre_affinity[mask] /= user_genre_count[mask]
    mu, sigma = user_genre_affinity[mask].mean(), user_genre_affinity[mask].std() + 1e-8
    user_genre_affinity[mask] = (user_genre_affinity[mask] - mu) / sigma

    # 5. Timestamp normalization
    ts_min = float(real_train["timestamp"].min())
    ts_range = float(real_train["timestamp"].max() - ts_min) + 1.0

    # 6. Movie release year
    import re
    movie_year = np.zeros(num_items, dtype=np.float32)
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        if mid < num_items:
            m = re.search(r'\((\d{4})\)', str(row.get("title", "")))
            if m:
                movie_year[mid] = float(m.group(1))
    valid_years = movie_year[movie_year > 0]
    if len(valid_years) > 0:
        median_year = np.median(valid_years)
        movie_year[movie_year == 0] = median_year
        movie_year = (movie_year - movie_year.mean()) / (movie_year.std() + 1e-8)

    # 7. Genre count per movie
    movie_genre_count = movie_genres.sum(axis=1).astype(np.float32)
    movie_genre_count = (movie_genre_count - movie_genre_count.mean()) / (movie_genre_count.std() + 1e-8)

    # 8. Tag genome features (1128 relevance scores per movie)
    _genome_path = Path(__file__).parent / "data" / DATASET / "genome-scores.csv"
    if _genome_path.exists():
        log.info("Loading tag genome scores...")
        # Replicate prepare.py's ID remapping: sort ratings by timestamp, unique movieIds in order
        _raw_ratings_path = Path(__file__).parent / "data" / DATASET
        if DATASET == "ml-25m":
            _raw_ratings = pd.read_csv(_raw_ratings_path / "ratings.csv")
        elif DATASET == "ml-10m":
            _raw_ratings = pd.read_csv(
                Path(__file__).parent / "data" / "ml-10M100K" / "ratings.dat",
                sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"])
        elif DATASET == "ml-1m":
            _raw_ratings = pd.read_csv(
                _raw_ratings_path / "ratings.dat", sep="::", engine="python",
                names=["userId", "movieId", "rating", "timestamp"])
        else:
            _raw_ratings = None
        if _raw_ratings is not None:
            _raw_ratings = _raw_ratings.sort_values("timestamp").reset_index(drop=True)
            _all_movies_ordered = _raw_ratings["movieId"].unique()
            _movie_map = {mid: i for i, mid in enumerate(_all_movies_ordered)}
            _genome_df = pd.read_csv(_genome_path)
            _num_tags = int(_genome_df["tagId"].max())
            genome_matrix = np.zeros((num_items, _num_tags), dtype=np.float32)
            has_genome = np.zeros(num_items, dtype=np.float32)
            # Map genome movieIds to remapped IDs
            _genome_df["mapped_mid"] = _genome_df["movieId"].map(_movie_map)
            _genome_df = _genome_df.dropna(subset=["mapped_mid"])
            _genome_df["mapped_mid"] = _genome_df["mapped_mid"].astype(int)
            # Pivot: fill genome_matrix
            for mid, group in _genome_df.groupby("mapped_mid"):
                if mid < num_items:
                    tag_ids = group["tagId"].values.astype(int) - 1  # 1-indexed to 0-indexed
                    genome_matrix[mid, tag_ids] = group["relevance"].values.astype(np.float32)
                    has_genome[mid] = 1.0
            log.info(f"Tag genome: {int(has_genome.sum())} movies with data out of {num_items} ({has_genome.mean()*100:.1f}%)")
            del _genome_df, _raw_ratings, _all_movies_ordered, _movie_map
        else:
            genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
            has_genome = np.zeros(num_items, dtype=np.float32)
    else:
        log.info("No genome-scores.csv found, using zeros.")
        genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
        has_genome = np.zeros(num_items, dtype=np.float32)
    GENOME_DIM = genome_matrix.shape[1]

    # Save cache
    np.savez_compressed(_cache_path,
        movie_genres=movie_genres, user_hist_bins=user_hist_bins,
        item_hist_bins=item_hist_bins, user_count_norm=user_count_norm,
        item_count_norm=item_count_norm, user_histories=user_histories,
        user_hist_ratings=user_hist_ratings, item_histories=item_histories,
        item_hist_ratings=item_hist_ratings, user_genre_affinity=user_genre_affinity,
        ts_min=np.array(ts_min), ts_range=np.array(ts_range),
        movie_year=movie_year, movie_genre_count=movie_genre_count,
        genome_matrix=genome_matrix, has_genome=has_genome,
    )
    log.info(f"Features cached to {_cache_path.name}")


# ═══════════════════════════════════════════════════════════════════
# RECENCY FILTER — drop oldest 20% of real ratings (features already computed from full data)
# ═══════════════════════════════════════════════════════════════════

RECENCY_FRAC = 0.8
real_mask = train_df["rating"] > 0
real_ratings = train_df[real_mask].sort_values("timestamp")
cutoff = int(len(real_ratings) * (1 - RECENCY_FRAC))
keep_real = real_ratings.iloc[cutoff:].index
keep_neg = train_df[~real_mask].index
train_df = train_df.loc[keep_real.union(keep_neg)].reset_index(drop=True)
n_dropped = int(real_mask.sum()) - len(keep_real)
log.info(f"Recency filter: kept {RECENCY_FRAC:.0%} of real ratings, dropped {n_dropped} oldest")


# ═══════════════════════════════════════════════════════════════════
# DATASET — precompute features on GPU (lookup tables stay compact)
# ═══════════════════════════════════════════════════════════════════

# Compact lookup tables on GPU (indexed by user/item ID, not per-sample)
_user_histories_gpu = torch.from_numpy(user_histories).to(DEVICE)       # (num_users, L)
_user_hist_ratings_gpu = torch.from_numpy(user_hist_ratings).to(DEVICE) # (num_users, L)
_item_histories_gpu = torch.from_numpy(item_histories).to(DEVICE)       # (num_items, IL)
_item_hist_ratings_gpu = torch.from_numpy(item_hist_ratings).to(DEVICE) # (num_items, IL)
_movie_genres_gpu = torch.from_numpy(movie_genres).to(DEVICE)           # (num_items, G)
_genome_gpu = torch.from_numpy(genome_matrix).to(DEVICE)               # (num_items, GENOME_DIM)
_has_genome_gpu = torch.from_numpy(has_genome).to(DEVICE)              # (num_items,)

def _build_gpu_tensors(df):
    """Precompute per-sample features on GPU. Histories/genres looked up at training time."""
    uids = df["userId"].values.astype(np.int64)
    mids = df["movieId"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)
    ts_norm = ((df["timestamp"].values - ts_min) / ts_range).astype(np.float32)
    ug_dot = np.sum(user_genre_affinity[uids] * movie_genres[mids], axis=1).astype(np.float32)
    movie_age = ts_norm - movie_year[mids]
    dense = np.column_stack([
        ts_norm,
        user_hist_bins[uids], user_count_norm[uids],
        item_hist_bins[mids], item_count_norm[mids],
        ug_dot,
        movie_year[mids], movie_genre_count[mids], movie_age,
    ]).astype(np.float32)
    return (
        torch.from_numpy(uids).to(DEVICE),
        torch.from_numpy(mids).to(DEVICE),
        torch.from_numpy(dense).to(DEVICE),
        torch.from_numpy(labels).to(DEVICE),
    )

# Store full train_df for recency-diverse training
_full_train_df = train_df.copy()

log.info("Precomputing training tensors on GPU...")
train_uids, train_mids, train_dense, train_labels = _build_gpu_tensors(train_df)
n_train = len(train_labels)
n_batches_per_epoch = n_train // BATCH_SIZE

# Pre-compute recency-filtered training data variants
_recency_variants = {}
def get_recency_data(frac):
    if frac not in _recency_variants:
        if frac >= 0.99:
            _recency_variants[frac] = (train_uids, train_mids, train_dense, train_labels)
        else:
            real_mask = _full_train_df["rating"] > 0
            real_ratings = _full_train_df[real_mask].sort_values("timestamp")
            cutoff = int(len(real_ratings) * (1 - frac))
            keep_real = real_ratings.iloc[cutoff:].index
            keep_neg = _full_train_df[~real_mask].index
            subset = _full_train_df.loc[keep_real.union(keep_neg)].reset_index(drop=True)
            _recency_variants[frac] = _build_gpu_tensors(subset)
            log.info(f"  Built recency={frac:.0%} data: {len(subset)} samples")
    return _recency_variants[frac]


# ═══════════════════════════════════════════════════════════════════
# EVAL SET — val rated items + sampled unrated negatives
# ═══════════════════════════════════════════════════════════════════

# Val positives: rated >= 4 in val set (label=1)
# Val hard negatives: rated < 4 in val set (label=0)
# Val easy negatives: random unrated movies (label=0), 1 per val positive
_val_pos_mask = val_df["label"] == 1
_val_pos = val_df[_val_pos_mask]
_val_hard_neg = val_df[~_val_pos_mask]
_n_val_pos = len(_val_pos)

# Build user -> all items (train + val) for eval neg sampling exclusion
_val_user_all = {}
for uid, items in user_all_items.items():
    _val_user_all[uid] = set(items)
for uid, group in val_df.groupby("userId"):
    if uid not in _val_user_all:
        _val_user_all[uid] = set()
    _val_user_all[uid].update(group["movieId"].values)

# Sample fixed easy negatives for val positives
_eval_rng = np.random.RandomState(42)
_easy_neg_users = _val_pos["userId"].values.astype(np.int64)
_easy_neg_items = np.empty(_n_val_pos, dtype=np.int64)
for i in range(_n_val_pos):
    uid = _easy_neg_users[i]
    rated = _val_user_all.get(uid, set())
    mid = _eval_rng.randint(0, num_items)
    while mid in rated:
        mid = _eval_rng.randint(0, num_items)
    _easy_neg_items[i] = mid

# Combine: val positives + val hard negatives + easy negatives
_eval_users = np.concatenate([
    _val_pos["userId"].values, _val_hard_neg["userId"].values, _easy_neg_users
]).astype(np.int64)
_eval_items = np.concatenate([
    _val_pos["movieId"].values, _val_hard_neg["movieId"].values, _easy_neg_items
]).astype(np.int64)
_eval_ts = np.concatenate([
    _val_pos["timestamp"].values, _val_hard_neg["timestamp"].values, _val_pos["timestamp"].values
])
_eval_ts_norm = ((_eval_ts - ts_min) / ts_range).astype(np.float32)
_eval_labels = np.concatenate([
    np.ones(_n_val_pos), np.zeros(len(_val_hard_neg)), np.zeros(_n_val_pos)
])

log.info("Precomputing eval tensors on GPU...")
_eval_uids_t, _eval_mids_t, _eval_dense_t, _ = _build_gpu_tensors(
    pd.DataFrame({
        "userId": _eval_users, "movieId": _eval_items,
        "timestamp": _eval_ts, "label": _eval_labels,
    })
)
n_eval = len(_eval_users)
log.info(f"Eval set: {_n_val_pos} pos + {len(_val_hard_neg)} hard neg + {_n_val_pos} easy neg = {n_eval} total")


def run_eval():
    """Evaluate AUC on validation set."""
    model.eval()
    eval_batch = BATCH_SIZE * 2
    all_scores = []
    with torch.no_grad():
        for start in range(0, n_eval, eval_batch):
            end = min(start + eval_batch, n_eval)
            u = _eval_uids_t[start:end]
            m = _eval_mids_t[start:end]
            logits = model(
                u, m, _eval_dense_t[start:end],
                _user_histories_gpu[u], _user_hist_ratings_gpu[u],
                _movie_genres_gpu[m],
                _item_histories_gpu[m], _item_hist_ratings_gpu[m],
                _genome_gpu[m], _has_genome_gpu[m],
            )
            all_scores.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(all_scores)
    return evaluate(_eval_labels, scores)


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class DLRM(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, use_din=True, use_item_din=True,
                 use_causal_sa=True, use_genome=True, use_field_attn=True,
                 use_streams=True):
        super().__init__()
        D = embed_dim
        self.D = D
        self.use_din = use_din
        self.use_item_din = use_item_din
        self.use_causal_sa = use_causal_sa
        self.use_genome = use_genome
        self.use_field_attn = use_field_attn
        self.use_streams = use_streams

        self.user_embed = nn.Embedding(num_users, D)
        self.item_embed = nn.Embedding(num_items, D)
        self.hist_embed = nn.Embedding(num_items + 1, D, padding_idx=PAD_IDX)
        self.rating_proj = nn.Linear(1, D)

        if use_causal_sa:
            self.hist_q = nn.Linear(D, D, bias=False)
            self.hist_k = nn.Linear(D, D, bias=False)
            self.hist_v = nn.Linear(D, D, bias=False)

        if use_din:
            self.din_attn = nn.Sequential(nn.Linear(3*2*D, 64), nn.ReLU(), nn.Linear(64, 1))

        if use_item_din:
            self.rater_embed = nn.Embedding(num_users + 1, D, padding_idx=USER_PAD_IDX)
            self.item_din_attn = nn.Sequential(nn.Linear(3*2*D, 64), nn.ReLU(), nn.Linear(64, 1))

        self.bottom_mlp = nn.Sequential(nn.Linear(NUM_DENSE, 128), nn.ReLU(), nn.Linear(128, D), nn.ReLU())
        self.genre_proj = nn.Sequential(nn.Linear(num_genres, D), nn.ReLU())

        if use_genome:
            self.genome_proj = nn.Sequential(
                nn.Linear(GENOME_DIM, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, D),
            )
            self.genome_gate = nn.Linear(D, D)

        # Count fields: user, item, user_hist, dense, genre + optional(item_hist, genome)
        n_fields = 5 + (1 if use_item_din else 0) + (1 if use_genome else 0)
        cross_dim = n_fields * D

        if use_field_attn:
            self.field_attn = nn.MultiheadAttention(D, num_heads=1, batch_first=True, dropout=0.1)

        if use_streams:
            self.user_stream = nn.Sequential(nn.Linear(3*D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.ReLU())
            item_stream_in = (3 + (1 if use_genome else 0)) * D
            self.item_stream = nn.Sequential(nn.Linear(item_stream_in, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.ReLU())
            top_in = cross_dim + 64 + 64 + 64
        else:
            top_in = cross_dim

        self.top_mlp = nn.Sequential(
            nn.Linear(top_in, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.padding_idx is not None: nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, user_id, movie_id, dense, history, hist_ratings,
                genres, item_hist, item_hist_ratings, genome_raw, has_genome_mask):
        D = self.D
        user_e = nn.functional.dropout(self.user_embed(user_id), 0.1, self.training)
        item_e = nn.functional.dropout(self.item_embed(movie_id), 0.1, self.training)

        # User history
        raw_hist = self.hist_embed(history)
        hist_rat_e = self.rating_proj(hist_ratings.unsqueeze(-1))
        hist_item_e = raw_hist

        if self.use_causal_sa:
            Q, K, V = self.hist_q(hist_item_e), self.hist_k(hist_item_e), self.hist_v(hist_item_e)
            scores = torch.bmm(Q, K.transpose(1,2)) / (D**0.5)
            L = history.size(1)
            cmask = torch.triu(torch.full((L,L), -1e4, device=history.device), diagonal=1)
            pmask = (history == PAD_IDX).unsqueeze(1).expand(-1, L, -1)
            scores = (scores + cmask.unsqueeze(0)).masked_fill(pmask, -1e4)
            hist_item_e = torch.bmm(torch.softmax(scores, dim=-1), V) + raw_hist

        if self.use_din:
            h = torch.cat([hist_item_e, hist_rat_e], dim=-1)
            t = torch.cat([item_e, item_e], dim=-1).unsqueeze(1).expand_as(h)
            w = self.din_attn(torch.cat([h, t, h*t], dim=-1)).squeeze(-1)
            w = w.masked_fill(history == PAD_IDX, -1e4)
            user_hist_e = (hist_item_e * torch.softmax(w, dim=-1).unsqueeze(-1)).sum(dim=1)
        else:
            mask = (history != PAD_IDX).unsqueeze(-1).float()
            user_hist_e = (hist_item_e * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Item history (raters)
        if self.use_item_din:
            re = nn.functional.dropout(self.rater_embed(item_hist), 0.1, self.training)
            rr = self.rating_proj(item_hist_ratings.unsqueeze(-1))
            h = torch.cat([re, rr], dim=-1)
            q = torch.cat([user_e, user_e], dim=-1).unsqueeze(1).expand_as(h)
            w = self.item_din_attn(torch.cat([h, q, h*q], dim=-1)).squeeze(-1)
            w = w.masked_fill(item_hist == USER_PAD_IDX, -1e4)
            item_hist_e = (re * torch.softmax(w, dim=-1).unsqueeze(-1)).sum(dim=1)

        dense_e = self.bottom_mlp(dense)
        genre_e = self.genre_proj(genres)

        # Genome
        if self.use_genome:
            ge = self.genome_proj(genome_raw)
            gate = torch.sigmoid(self.genome_gate(ge))
            genome_field = gate * ge + (1 - gate) * item_e.detach()

        # Assemble fields
        field_list = [user_e, item_e, user_hist_e]
        if self.use_item_din:
            field_list.append(item_hist_e)
        field_list.extend([dense_e, genre_e])
        if self.use_genome:
            field_list.append(genome_field)

        fields = torch.stack(field_list, dim=1)

        if self.use_field_attn:
            attn_out, _ = self.field_attn(fields, fields, fields)
            x = (fields + attn_out).reshape(fields.size(0), -1)
        else:
            x = fields.reshape(fields.size(0), -1)

        if self.use_streams:
            us_in = torch.cat([user_e, user_hist_e, dense_e], dim=-1)
            is_parts = [item_e]
            if self.use_item_din: is_parts.append(item_hist_e)
            else: is_parts.append(torch.zeros_like(item_e))
            is_parts.append(genre_e)
            if self.use_genome: is_parts.append(genome_field)
            us = self.user_stream(us_in)
            its = self.item_stream(torch.cat(is_parts, dim=-1))
            x = torch.cat([x, us, its, us * its], dim=-1)

        return self.top_mlp(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# TRAIN ONE MODEL VARIANT
# ═══════════════════════════════════════════════════════════════════

use_amp = DEVICE.type == "cuda"

def get_preds(mdl):
    mdl.eval()
    scores = []
    with torch.no_grad():
        for s in range(0, n_eval, BATCH_SIZE*2):
            e = min(s + BATCH_SIZE*2, n_eval)
            u, m = _eval_uids_t[s:e], _eval_mids_t[s:e]
            logits = mdl(u, m, _eval_dense_t[s:e],
                         _user_histories_gpu[u], _user_hist_ratings_gpu[u],
                         _movie_genres_gpu[m], _item_histories_gpu[m],
                         _item_hist_ratings_gpu[m], _genome_gpu[m], _has_genome_gpu[m])
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)

def train_one(model_kwargs, seed=42, lr=LR, wd=WEIGHT_DECAY, accum=4,
              patience=3, label_smooth=0.05, recency_frac=0.8):
    torch.manual_seed(seed)
    model = DLRM(**model_kwargs).to(DEVICE)
    if DEVICE.type == "cuda":
        model = torch.compile(model)
    # Get training data for this recency fraction
    t_uids, t_mids, t_dense, t_labels = get_recency_data(recency_frac)
    nt = len(t_labels)
    nbpe = nt // BATCH_SIZE
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_auc, best_state, no_imp, gstep = 0.0, None, 0, 0
    eval_every = max(nbpe // 3, 1)
    t0 = time.time()
    for _ in range(20):
        model.train()
        perm = torch.randperm(nt, device=DEVICE)
        for i in range(nbpe):
            idx = perm[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            u, m = t_uids[idx], t_mids[idx]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(u, m, train_dense[idx],
                               _user_histories_gpu[u], _user_hist_ratings_gpu[u],
                               _movie_genres_gpu[m], _item_histories_gpu[m],
                               _item_hist_ratings_gpu[m], _genome_gpu[m], _has_genome_gpu[m])
                loss = crit(logits, t_labels[idx] * (1-2*label_smooth) + label_smooth)
            scaler.scale(loss / accum).backward()
            if (i+1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            gstep += 1
            if gstep % eval_every == 0:
                preds = get_preds(model)
                auc = evaluate(_eval_labels, preds)["auc"]
                if auc > best_auc:
                    best_auc = auc; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= patience: break
                model.train()
        if no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    preds = get_preds(model)
    elapsed = time.time() - t0
    del model, opt, scaler; torch.cuda.empty_cache()
    return preds, best_auc, elapsed


# ═══════════════════════════════════════════════════════════════════
# DIVERSE ENSEMBLE VARIANTS (architecture × HP × seed)
# ═══════════════════════════════════════════════════════════════════

VARIANTS = [
    # --- Full model, seed diversity ---
    {"name": "full_s42",      "model_kwargs": {}, "seed": 42},
    {"name": "full_s43",      "model_kwargs": {}, "seed": 43},
    {"name": "full_s44",      "model_kwargs": {}, "seed": 44},
    # --- Architecture ablations ---
    {"name": "meanpool",      "model_kwargs": {"use_din": False, "use_causal_sa": False}},
    {"name": "no_itemdin",    "model_kwargs": {"use_item_din": False}},
    {"name": "no_genome",     "model_kwargs": {"use_genome": False}},
    {"name": "no_fieldattn",  "model_kwargs": {"use_field_attn": False}},
    {"name": "no_streams",    "model_kwargs": {"use_streams": False}},
    {"name": "minimal",       "model_kwargs": {"use_din": False, "use_item_din": False, "use_causal_sa": False, "use_genome": False, "use_field_attn": False, "use_streams": False}},
    {"name": "din_only",      "model_kwargs": {"use_item_din": False, "use_genome": False, "use_field_attn": False, "use_streams": False}},
    # --- Embed dim ---
    {"name": "dim16",         "model_kwargs": {"embed_dim": 16}},
    {"name": "dim56",         "model_kwargs": {"embed_dim": 56}},
    # --- RECENCY DIVERSITY (key for ensemble per Ning) ---
    {"name": "recent50",      "model_kwargs": {}, "recency_frac": 0.5},
    {"name": "recent60",      "model_kwargs": {}, "recency_frac": 0.6},
    {"name": "recent70",      "model_kwargs": {}, "recency_frac": 0.7},
    {"name": "recent90",      "model_kwargs": {}, "recency_frac": 0.9},
    {"name": "recent100",     "model_kwargs": {}, "recency_frac": 1.0},
    # --- Recency × architecture combos ---
    {"name": "r50_meanpool",  "model_kwargs": {"use_din": False, "use_causal_sa": False}, "recency_frac": 0.5},
    {"name": "r50_noitemdin", "model_kwargs": {"use_item_din": False}, "recency_frac": 0.5},
    {"name": "r60_nogenome",  "model_kwargs": {"use_genome": False}, "recency_frac": 0.6},
    {"name": "r70_dim16",     "model_kwargs": {"embed_dim": 16}, "recency_frac": 0.7},
    {"name": "r90_s43",       "model_kwargs": {}, "seed": 43, "recency_frac": 0.9},
    # --- HP diversity ---
    {"name": "lr_high",       "model_kwargs": {}, "lr": 2e-4},
    {"name": "lr_low",        "model_kwargs": {}, "lr": 5e-5},
    {"name": "no_smooth",     "model_kwargs": {}, "label_smooth": 0.0},
    # --- More arch+seed combos ---
    {"name": "meanpool_s43",  "model_kwargs": {"use_din": False, "use_causal_sa": False}, "seed": 43},
    {"name": "nogenome_s43",  "model_kwargs": {"use_genome": False}, "seed": 43},
]

training_start = time.time()
peak_memory_mb = 0.0
all_preds, all_aucs = {}, {}

log.info(f"Training {len(VARIANTS)} diverse ensemble variants...")
for i, v in enumerate(VARIANTS):
    name = v.pop("name")
    log.info(f"[{i+1}/{len(VARIANTS)}] {name}...")
    preds, auc, elapsed = train_one(**v)
    all_preds[name] = preds
    all_aucs[name] = auc
    log.info(f"  {name:20s} | AUC {auc:.4f} | {elapsed:.0f}s")
    if DEVICE.type == "cuda":
        peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / (1024**2))

training_seconds = time.time() - training_start

# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE STACKING
# ═══════════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

names = sorted(all_preds.keys())
X = np.column_stack([all_preds[n] for n in names])
y = _eval_labels.astype(int)

log.info(f"\n{'='*60}")
log.info(f"Trained {len(names)} variants in {training_seconds:.0f}s")
log.info(f"Individual AUCs: min={min(all_aucs.values()):.4f} max={max(all_aucs.values()):.4f}")

# Simple average
avg = evaluate(_eval_labels, X.mean(axis=1))["auc"]
log.info(f"Simple average AUC: {avg:.4f}")

# LogReg stacking (5-fold CV, heavy regularization per Ning: C=0.0005)
lr_scores = cross_val_predict(
    LogisticRegression(C=0.0005, max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
lr_auc = evaluate(_eval_labels, lr_scores)["auc"]
log.info(f"LogReg stacked AUC: {lr_auc:.4f}")

# HistGBM stacking (5-fold CV, conservative settings)
hgb_scores = cross_val_predict(
    HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                   learning_rate=0.05, min_samples_leaf=500,
                                   max_depth=3, l2_regularization=1.0,
                                   random_state=42),
    X, y, cv=5, method="predict_proba")[:, 1]
hgb_auc = evaluate(_eval_labels, hgb_scores)["auc"]
log.info(f"HistGBM stacked AUC: {hgb_auc:.4f}")

best_method = max([("avg", avg), ("logreg", lr_auc), ("histgbm", hgb_auc)], key=lambda x: x[1])
log.info(f"Best: {best_method[0]} ({best_method[1]:.4f})")

if best_method[0] == "histgbm":
    metrics = evaluate(_eval_labels, hgb_scores)
elif best_method[0] == "logreg":
    metrics = evaluate(_eval_labels, lr_scores)
else:
    metrics = evaluate(_eval_labels, X.mean(axis=1))

num_params = len(names)
total_seconds = time.time() - total_start
print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
