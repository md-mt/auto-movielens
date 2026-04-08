# Plan: Set Up auto-movielens Fresh Start

## Context

Replicate Xu Ning's MovieLens autoresearch experiment from scratch in `~/auto-movielens/`.
The current `~/movielens/` contains Ning's finished results (0.854 AUC).
This fresh project will be pushed to a GPU dev server with **1x A100 80GB** for autonomous experimentation.

## Files to Create

All files go in `~/auto-movielens/`.

### 1. `prepare.py` — Data & Eval Harness (read-only)
- Copy verbatim from `~/movielens/prepare.py` (419 lines)
- Contains: dataset download, `load_data()`, `load_data_hybrid()`, `evaluate()`, `print_summary()`
- This file should NOT be modified during experimentation

### 2. `train.py` — Vanilla DLRM Baseline
- Copy the original baseline from Ning's first commit (`git -C ~/movielens show c65f884:train.py`)
- Vanilla DLRM: user/item embeddings, genre multi-hot, mean-pooled history, dense stats, pairwise dots, top MLP
- Expected baseline AUC: ~0.770 on ml-25m

### 3. `pyproject.toml` — Dependencies
```toml
[project]
name = "auto-movielens"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "lightgbm",
]
```

### 4. `program.md` — Autoresearch Protocol
Adapted from Ning's `~/movielens/program.md`:
- **GPU**: 1x A100 80GB (larger VRAM budget than Ning's 2x L4 24GB)
- **Process rule**: 10 HP variations per idea before discarding
- **Dataset default**: ml-25m
- **Early stopping**: patience=3, sub-epoch eval 3x/epoch
- Empty experiment history (fresh start)
- Idea backlog seeded from Ning's key wins (item-side DIN, tag genome, rating histograms, NEG_RATIO tuning, ensemble stacking)

### 5. `CLAUDE.md` — Claude Code Instructions
- Project overview: fresh MovieLens autoresearch, starting from vanilla DLRM
- Commands: smoke test (ml-100k), standard run (ml-25m)
- Architecture: describe the baseline DLRM
- Rules: modify train.py freely, don't modify prepare.py eval harness
- Goal: maximize val_auc on ml-25m
- Hardware: 1x A100 80GB GPU

### 6. `.gitignore`
```
data/
run.log
*.pyc
__pycache__/
results.tsv
```

## Setup Commands

```bash
# 1. Fix permissions and populate the repo
cd ~/auto-movielens

# Copy files (run these manually or via script)
cp ~/movielens/prepare.py .
git -C ~/movielens show c65f884:train.py > train.py
# Create pyproject.toml, program.md, CLAUDE.md, .gitignore (from this plan)

# 2. Initial commit
git add prepare.py train.py pyproject.toml program.md CLAUDE.md .gitignore
git commit -m "Initial vanilla DLRM baseline for MovieLens autoresearch"

# 3. Create GitHub repo and push
gh repo create auto-movielens --private --source=. --push
```

## On the GPU Dev Server

```bash
# Clone
git clone <repo-url> ~/auto-movielens
cd ~/auto-movielens

# Install deps
pip install torch pandas numpy scikit-learn scipy lightgbm

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Establish baseline
DATASET=ml-25m python3 train.py > run.log 2>&1
grep "^val_auc:\|^peak_memory_mb:" run.log
# Expected: AUC ~0.770, <10 min on A100
```

## Verification

1. **Local smoke test**: `DATASET=ml-100k python3 train.py` — no crashes
2. **Baseline on GPU**: `DATASET=ml-25m python3 train.py` — AUC ~0.770
3. **Git status**: clean working tree after initial commit
