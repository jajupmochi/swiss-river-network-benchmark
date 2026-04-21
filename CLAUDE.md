# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.12+, managed with `uv` (see `uv.lock`). Install: `uv sync --no-cache`.
- Run scripts with `uv run python -m <module>` — never bare `python` or `pip install`.
- PyTorch + PyTorch Geometric assume **CUDA is available**; there is no CPU fallback path.

## Entry points

Executable modules live in `swissrivernetwork/benchmark/`:

- `uv run python -m swissrivernetwork.benchmark.data_preparation` — prepare the dataset.
- `uv run python -m swissrivernetwork.benchmark.ray_tune` — hyperparameter tuning with Ray Tune.
- `uv run python -m swissrivernetwork.benchmark.ray_evaluation` — evaluate trained models.
- `uv run python -m swissrivernetwork.benchmark.train_single_model` — train one unified model.
- `uv run python -m swissrivernetwork.benchmark.train_isolated_station` — train per-station models.
- `uv run python -m swissrivernetwork.benchmark.run_win_len_sweep` — sweep trained-at-wl=90 checkpoints across many eval window lengths; produces the CSVs behind paper Fig. 4 and the HLE dimension of Fig. 2. Runs ISOLATED then GRAPHLET in strict order; see the module docstring.

For invocation details and flags, see `.claude/skills/run-benchmark/SKILL.md` or run `/run-benchmark`.

## Gotchas

- **`RAY_CHDIR_TO_TRIAL_DIR=0`** is set in `ray_tune.py` on purpose — Ray otherwise changes the working directory per trial and breaks relative paths. Do not remove.
- **Weights & Biases** logging needs `WANDB_API_KEY`. If it's not set, prefer disabling wandb (`WANDB_MODE=disabled`) over stripping the code.
- **`swissrivernetwork/benchmark/outputs/ray_results/`** can hold hundreds of trial subdirectories — never glob or list it unfiltered. Use `swissrivernetwork/benchmark/outputs/trim_checkpoints.py` to prune.

## Working style

- For non-trivial changes (new features, cross-module refactors, algorithm choices), propose a plan before editing.
- Commits follow **Conventional Commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`. Match the existing `git log` style.
- This repo has no configured test suite or CI — verify changes by running the relevant entry point, not by inventing tests.
