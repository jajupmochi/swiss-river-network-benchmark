# Working style

The conventions below are duplicated in
[`CLAUDE.md`](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/CLAUDE.md),
which is the source of truth for agent-driven contributions.

## Environment

- Python 3.12, managed with `uv`. Install with `uv sync --no-cache`.
- Run every script via `uv run python -m <module>`. Never `pip install`.
- PyTorch + PyTorch-Geometric assume **CUDA is available**; there is no
  CPU fallback path.

## Entry points

Executable modules live in `swissrivernetwork/benchmark/`:

- `data_preparation.py` — dataset build.
- `ray_tune.py` — hyperparameter search.
- `ray_evaluation.py` — test-time evaluation.
- `train_single_model.py` — single unified model.
- `train_isolated_station.py` — per-station models.
- `run_win_len_sweep.py` — window-length sweep.

## Gotchas

- `RAY_CHDIR_TO_TRIAL_DIR=0` is set inside `ray_tune.py` on purpose —
  Ray otherwise changes the working directory per trial and breaks
  relative paths. **Do not remove.**
- Weights & Biases needs `WANDB_API_KEY`. If you don't have one, set
  `WANDB_MODE=disabled` rather than stripping the code.
- `swissrivernetwork/benchmark/outputs/ray_results/` can hold hundreds
  of subdirectories — never list it unfiltered. Use
  `outputs/trim_checkpoints.py` to prune.

## Commit style

Conventional Commits. Good: `feat(app): add Streamlit compare tab`.
Bad: `update`, `fix stuff`, `wip`.

## No test suite

Verify changes by running the relevant entry point, not by inventing
tests. CI will grow a real test suite over time; today it runs ruff
and a handful of notebook executions.
