# Swiss River Network Benchmarks

Code and data for the ICPR 2026 submission *"Benchmarking Transformers on Spatio-Temporal River Water Temperature Modeling"*.

## How to install

- Install by uv:
```bash
uv sync --no-cache
```

- Or install by python install:
```bash
pip install -e .
```

GPU with CUDA is required for training and evaluation; there is no CPU fallback path.

## How to run

All experiment drivers live in `swissrivernetwork/benchmark/` and are invoked via `uv run python -m …`:

| Module | Purpose |
| --- | --- |
| `data_preparation` | Build the three dataset splits (`swiss-1990`, `swiss-2010`, `zurich`). |
| `ray_tune` | Hyperparameter search with Ray Tune. |
| `ray_evaluation` | Evaluate Ray-tuned checkpoints; produces the wl=90 tables. |
| `train_single_model` / `train_isolated_station` | Train individual models without Ray Tune. |
| `run_win_len_sweep` | Sweep the trained-at-wl=90 checkpoints across many eval window lengths; produces the CSVs behind paper Fig. 4 and the HLE dimension of Fig. 2. |

Flags and full invocation examples are in [`.claude/skills/run-benchmark/SKILL.md`](.claude/skills/run-benchmark/SKILL.md).

### Window-length sweep (paper Fig. 4 / HLE)

```bash
uv run python -m swissrivernetwork.benchmark.run_win_len_sweep
```

Each model is trained once at `window_len = 90` and then evaluated at many window lengths `W ∈ {1, 3, 5, 7, 15, 30, 60, 90, 120, 150}` (capped per dataset). The driver runs in two strict phases:

1. **ISOLATED** — `lstm`, `transformer` (× PE in `{learnable, sinusoidal, rope}`). Writes `wt_hat` predictions under `dump/predictions/<path_extra_keys>-evalwl{W}/`.
2. **GRAPHLET** — `graphlet`, `transformer_graphlet`. Reads the Phase-1 `wt_hat` dumps as neighbor features.

Results append to `swissrivernetwork/benchmark/visualize_results/outputs/win_lens/{graph}_{method}_win_lens_resu.csv`. The driver refuses (raises `FileExistsError`) if a row for the same `(wl)` or `(wl, pe)` is already in the CSV — back up the old file first.

Set `DEBUG_SINGLE = True` in `run_win_len_sweep.py` `__main__` to restrict the sweep to a single `(graph, method, wl)` tuple for PyCharm breakpoint work.

#### Dump-path fix (eval window-length leakage)

Before this fix the isolated-model `wt_hat` dump path was keyed only by the *trained* config, so every eval `W` wrote to the same directory and later `W` values overwrote earlier ones. Graphlet models — which consume those dumps as neighbor features — ended up reading whichever `W` had written last, typically the longest one in the sweep. This silently gave Graphlet a long-history advantage at short eval windows.

The fix is in `util.get_evaluation_path_keys`: when `eval_wl ≠ trained_wl`, the path now carries an `-evalwl{W}` suffix, isolating each `(trained_config, eval_W)` into its own dump dir. `ray_evaluation.process_method` threads the required `trained_window_len` into the suffix builder.

**What needs re-running:** only Graphlet and Transformer-Graphlet sweep rows for `W ≠ 90`. Isolated / Embedded / ST-GNN sweep numbers, and anything evaluated only at `W = 90` (e.g. Table 3, Fig. 3, noise experiments), are unaffected.

### Downstream visualization

CSVs produced by `run_win_len_sweep` are consumed by three notebooks in `swissrivernetwork/benchmark/visualize_results/`:

- `window_lens_resu.ipynb` — paper Fig. 4 grid plot (PDF).
- `visual_win_lens.ipynb` — interactive Plotly views of the same data.
- `results_in_polar.ipynb` — HLE dimension of the Fig. 2 radar plots (uses `variable_list = [1, 3, 5, 7, 15, 30, 60, 90]` with exponential weight `w(l) = 2^{-l/45}`).

Each notebook includes a prerequisite note at the top pointing back to `run_win_len_sweep.py`.

## Project layout

- `swissrivernetwork/benchmark/` — experiment drivers, model training / evaluation code.
- `swissrivernetwork/benchmark/visualize_results/` — notebooks that build the paper figures.
- `swissrivernetwork/benchmark/outputs/ray_results/` — Ray Tune trial directories (do not list unfiltered; use `outputs/trim_checkpoints.py` to prune).
- `CLAUDE.md` — conventions for assistive tooling; mirrors the above for coding agents.
