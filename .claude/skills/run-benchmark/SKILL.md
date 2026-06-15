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

## 5. Window-length sweep (paper Fig. 4 / HLE metric)

```bash
uv run python -m swissrivernetwork.benchmark.run_win_len_sweep
```

Evaluates each `trained_wl = 90` checkpoint at many eval window lengths (default `[1, 3, 5, 7, 15, 30, 60, 90, 120, 150]` capped per dataset). Runs in two strict phases:

1. **ISOLATED** — `lstm`, `transformer` (× PE in `{learnable, sinusoidal, rope}`) run test-time inference and dump `wt_hat` to `dump/predictions/<path_extra_keys>-evalwl{W}/`.
2. **GRAPHLET** — `graphlet`, `transformer_graphlet` read those dumps as neighbor features.

The `-evalwl{W}` suffix is appended by `util.get_evaluation_path_keys` whenever the eval W differs from the trained W. This prevents the earlier bug where every eval W shared one directory and graphlet silently read neighbor predictions written by the last W in the sweep. Graphlet sweep numbers produced **before** this fix are unreliable for W ≠ 90 — regenerate them.

Outputs append to `visualize_results/outputs/win_lens/{graph}_{method}_win_lens_resu.csv`. If a row for the same `(wl)` or `(wl, pe)` already exists, the driver raises `FileExistsError` and prints a backup command; that is deliberate — back up the pre-fix CSVs before re-running instead of mixing rows.

Set `DEBUG_SINGLE = True` in the `__main__` block to restrict to one `(graph, method, wl)` tuple for PyCharm breakpoint work. See the module docstring for which downstream notebooks consume the CSVs (`window_lens_resu.ipynb`, `visual_win_lens.ipynb`, `results_in_polar.ipynb`).

## Output hygiene

`swissrivernetwork/benchmark/outputs/ray_results/` can grow to hundreds of trial directories. Use `swissrivernetwork/benchmark/outputs/trim_checkpoints.py` to prune. Never list that directory unfiltered.
