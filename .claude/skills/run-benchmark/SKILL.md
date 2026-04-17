---
name: run-benchmark
description: How to invoke the Swiss River Network benchmark scripts (data prep, Ray Tune hyperparameter search, evaluation, and per-model training). Use when the user wants to launch or explain a benchmark run, or asks which flags a script takes.
---

# Running the Swiss River Network benchmark

All scripts are Python modules under `swissrivernetwork.benchmark`. Always launch via `uv run python -m <module> ...` from the repo root. GPU is required; `RAY_CHDIR_TO_TRIAL_DIR=0` must stay set (it's set inside `ray_tune.py`).

## Models and graphs

- **Methods**: `lstm`, `graphlet`, `lstm_embedding`, `stgnn`, `transformer`, `transformer_graphlet`, `transformer_embedding`, `transformer_stgnn`
- **Graphs**: `swiss-1990`, `swiss-2010`, `zurich`

## 1. Prepare the dataset

```bash
uv run python -m swissrivernetwork.benchmark.data_preparation
```

Uses in-file config (no CLI args). Edit the `__main__` block of `data_preparation.py` to change inputs or outputs. Run once before training or evaluation.

## 2. Hyperparameter tuning with Ray Tune

```bash
uv run python -m swissrivernetwork.benchmark.ray_tune \
  -m transformer_embedding \
  -g swiss-2010 \
  -n 200 \
  -wl 90
```

Key flags (see `ray_tune.py` `__main__` for the full list):

- `-c, --config` YAML config file (fills in unset flags)
- `-m, --method` one of the methods above
- `-g, --graph` one of the graphs above
- `-wl, --window_len` history window in days (default 90)
- `-n, --num_samples` random-search samples (default 200)
- `-s, --storage_path` where to write trial results (default: `swissrivernetwork/benchmark/outputs/ray_results/`)
- `-r, --resume` resume from previous run; pair with `-rts, --resume_timestamp YYYY-MM-DD_HH-MM-SS`
- Missing-value handling: `-mvm {mask_embedding,interpolation,zero,none}`, `-mmc`, `-mmr`
- Subsequence handling: `-ssm {pad,drop}`
- Forecasting: `-ucx, --use_current_x` (bool), `-fs, --future_steps`, `-em, --extrapo_mode {none,limo,future_embedding,recursive}`
- Transformer-specific: `-pe, --positional_encoding {none,sinusoidal,rope,learnable}`, `-ml, --max_len`
- Embeddings: `-use, --use_station_embedding` (bool)
- General: `-v, --verbose`, `-d, --dev_run` (tiny subsets for pipeline smoke test)

For Weights & Biases logging set `WANDB_API_KEY`; otherwise prefer `WANDB_MODE=disabled`.

## 3. Evaluate trained models

```bash
uv run python -m swissrivernetwork.benchmark.ray_evaluation
```

**No CLI flags** — edit the `settings` dict and `SINGLE_RUN` / `GRAPH_NAMES[...]` / `METHODS[...]` selections in the `__main__` block of `ray_evaluation.py` before running. Reads trials from `swissrivernetwork/benchmark/outputs/ray_results/`.

## 4. Train a single model (without Ray Tune)

```bash
uv run python -m swissrivernetwork.benchmark.train_single_model         # unified model
uv run python -m swissrivernetwork.benchmark.train_isolated_station     # per-station models
```

Configuration is in the module `__main__` blocks — edit the hyperparameter block rather than passing flags.

## Output hygiene

`swissrivernetwork/benchmark/outputs/ray_results/` can grow to hundreds of trial directories. Use `swissrivernetwork/benchmark/outputs/trim_checkpoints.py` to prune. Never list that directory unfiltered.
