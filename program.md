# autoresearch — MovieLens Recommendation

Autonomous experimentation loop for improving pointwise recommendation (AUC) on MovieLens.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr07`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed: data download/loading, train/val/test splits, AUC evaluation, constants. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model architecture, optimizer, training loop.
4. **Verify dependencies**: Run `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` to confirm CUDA is available.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on NVIDIA A100 80GB GPU (CUDA). Training terminates via early stopping (patience=3 evals, sub-epoch eval 3x/epoch), not a fixed time budget. Launch it as:

```bash
DATASET=ml-25m python3 train.py > run.log 2>&1
```

**Dataset selection** via the `DATASET` env var:
- `ml-100k` — 100K ratings, for quick smoke testing of code changes (~seconds)
- `ml-1m` — 1M ratings, fast iteration (~minutes)
- `ml-10m` — 10M ratings, medium scale (~5-15 minutes on A100)
- `ml-25m` — 25M ratings, **default for experimentation** (~3-8 minutes on A100)

Use `ml-100k` to quickly validate that code changes don't crash, then `ml-25m` for real metric comparison.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Everything is fair game: feature engineering, feature transformations, model architecture, optimizer, hyperparameters, training loop, batch size, model size, sequence modeling, negative sampling, etc.
- Modify `prepare.py` when the model demands a different training data setup (e.g. implicit feedback, negative sampling, different label definitions, new data splits). The evaluation function and summary printer should remain stable.

**What you CANNOT do:**
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_auc.** Training terminates via early stopping, so you don't need to worry about time budgets. Everything is fair game: change the feature engineering, the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing.

**Memory** is a soft constraint. The NVIDIA A100 has 80 GB VRAM. Some increase is acceptable for meaningful AUC gains, but it should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_auc:          0.823456
val_logloss:      0.543210
training_seconds: 600.1
total_seconds:    615.3
peak_memory_mb:   2048.0
dataset:          ml-1m
num_users:        6040
num_items:        3706
num_train:        800168
num_params_M:     1.2
```

Extract the key metrics:

```bash
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_auc	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.823456) — use 0.000000 for crashes
3. peak memory in MB, round to .0f (e.g. 2048) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_auc	memory_mb	status	description
a1b2c3d	0.823456	2048	keep	baseline DLRM
b2c3d4e	0.831200	2100	keep	increase embed_dim to 32
c3d4e5f	0.820100	2048	discard	remove history features
d4e5f6g	0.000000	0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr07`).

LOOP FOREVER:

1. Pick an experiment idea from the backlog (or invent a new one).
2. Modify `train.py` with the experimental idea.
3. git commit.
4. Smoke test: `DATASET=ml-100k python3 train.py` — check it doesn't crash.
5. Real run: `DATASET=ml-25m python3 train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
6. Read results: `grep "^val_auc:\|^peak_memory_mb:" run.log`
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on this idea.
8. Record results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git).
9. If val_auc improved: keep the commit, `git push` to upstream.
10. If val_auc is equal or worse: `git reset --hard HEAD~1` to discard.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Crashes**: If a run crashes (OOM, CUDA error, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, try different feature engineering. The loop runs until the human interrupts you, period.

## Research backlog

Prioritized by expected impact and implementation difficulty.

### Idea backlog (seeded from prior research)

> These ideas were validated in prior experiments by Xu Ning (0.770 → 0.854 AUC over ~500 experiments).
> They are listed here as a roadmap — each must be re-implemented and validated from scratch.

**Tier 1 — High impact, proven wins:**
1. **Item-side DIN attention** — Attention over recent raters of the target item. Was the single biggest architectural win (+0.029 AUC on ml-10m). Captures collaborative signal.
2. **Tag genome features** — 1128-dim relevance vectors with learned 3-layer MLP bottleneck compression (1128→256→64→D) + sigmoid gate for missing data (78% missing). PCA failed; learned compression works.
3. **Rating histogram features** — Replace mean+std with 5-bin rating distributions for both users and items. Full distribution shape > summary statistics (+0.003).
4. **NEG_RATIO tuning** — Reducing from 4→1 gave +0.005 AUC. Fewer random negatives = cleaner signal.
5. **GDCN gated cross layers** — Gated cross-network for feature interactions. 3-4 layers on ml-25m.
6. **Field attention** — 1-head MHA across feature fields with additive residual. Slightly better than GDCN.

**Tier 2 — Medium impact, incremental:**
7. **DIN attention for user history** — Replace mean pooling with target-aware attention (+0.011 on ml-1m).
8. **Causal self-attention** — Lightweight self-attention before DIN for history sequence modeling.
9. **FinalMLP two-stream** — User stream + item stream with bilinear interaction.
10. **embed_dim tuning** — 24-28 is the sweet spot on ml-25m (16 is too small, 32 overfits).
11. **Batch size + gradient accumulation** — batch_size=16384 + ACCUM_STEPS=4-8 helps on ml-25m.

**Tier 3 — Ensemble (after single model is saturated):**
12. **Diverse model ensemble** — Train 20-60 architecturally diverse variants, stack with HistGBM.
13. **Recency-diverse models** — Train on different time slices for temporal diversity.
14. **Non-linear stacking** — HistGBM >> LogReg >> simple averaging for stacking.

### Key learnings from prior research

1. **New information > more capacity.** Features with genuinely new signal help. Bigger MLPs/more heads don't.
2. **Richer features unlock more capacity.** 4 GDCN layers only work because histograms provide richer input.
3. **Training procedure changes rarely work.** LR schedules, warmup, multi-task, contrastive, BPR, focal all failed.
4. **10 HP trials per idea.** Never test an architecture idea once and discard.
5. **fp16 > bf16 for attention-heavy models.**
6. **Fixed random seeds are essential.** Use SEED=42 for determinism.
7. **Architecture changes work, training procedure changes don't.**

### Experiment history

(empty — fresh start)
