# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fresh-start autoresearch for MovieLens movie recommendation. The goal is to maximize val_auc on ml-25m, starting from a vanilla DLRM baseline (~0.770 AUC) and iteratively improving through autonomous experimentation.

Prior work by Xu Ning reached 0.854 AUC (ensemble) / 0.821 (single model) over ~500 experiments. This project replicates that journey from scratch on a single A100 80GB GPU.

## Commands

```bash
# Quick smoke test (ml-100k, ~seconds) — only for crash detection, NOT for AUC comparison
DATASET=ml-100k python3 train.py

# Standard experiment (ml-25m on A100, ~3-8 minutes)
DATASET=ml-25m python3 train.py

# Full experiment run (redirected, for autoresearch loop)
DATASET=ml-25m python3 train.py > run.log 2>&1

# Check results
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Architecture (baseline)

- **`prepare.py`** — Data download/loading (all MovieLens sizes), train/val/test splits, AUC evaluation, `print_summary()`. Do NOT modify the evaluation harness.
- **`train.py`** — The experimentation file. Vanilla DLRM: user/item embeddings, genre multi-hot, mean-pooled history, dense stats, pairwise dots, top MLP. **This is the file you modify.**
- **`program.md`** — The autoresearch protocol: setup, experiment loop, logging format, idea backlog, and experiment history.
- **`results.tsv`** — Experiment log (untracked). Tab-separated: commit, val_auc, memory_mb, status, description.

## Current model (vanilla DLRM baseline)

```
Features:
  - Sparse: userId, movieId (embeddings, dim=16)
  - Genre: multi-hot → linear projection → dim=16
  - User history: last 50 items → mean pooling → dim=16
  - Dense: timestamp, user stats (count/mean/std), item stats (count/mean/std) → bottom MLP → dim=16

Interaction: pairwise dot products of all 5 embedding vectors (C(5,2)=10 dots)

Top MLP: (10 + 5*16=90) → 256 → 128 → 1 (with dropout 0.2)

Loss: BCEWithLogitsLoss
Optimizer: Adam, LR=1e-3, weight_decay=1e-5
Training: batch=2048, eval every 5 epochs, patience=20
```

## Key Details

- **Metric**: val_auc (higher is better).
- **Hardware**: 1x NVIDIA A100 80GB GPU.
- **Datasets**: `ml-100k` (smoke test), `ml-1m` (fast), `ml-10m` (medium), `ml-25m` (default). Via `DATASET` env var.
- **Goal**: Maximize val_auc on ml-25m. Everything in train.py is fair game.
- **Rules**: Do not modify `evaluate()` or `print_summary()` in prepare.py. Do not add dependencies.
