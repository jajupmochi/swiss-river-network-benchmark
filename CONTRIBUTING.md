# Contributing to Swiss River Network Benchmark

Thanks for considering a contribution. This project is both a research artefact (ICPR 2026
submission) and a reusable benchmark for spatio-temporal river-temperature modeling.

## TL;DR

- Open a GitHub issue before large changes.
- Fork, branch from `main`, keep PRs focused.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
- `uv sync` to install, `uvx ruff format .` before committing, `uv run pytest` to check.

## 1. Development setup

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
uv sync --no-cache          # create .venv/, install all dependencies + dev extras
uv run pytest -q            # smoke tests
```

GPU with CUDA is required to run training/evaluation end-to-end; there is no CPU fallback.
For documentation / lint / packaging work a CPU-only machine is sufficient.

## 2. Reporting bugs

Open an [Issue](https://github.com/jajupmochi/swiss-river-network-benchmark/issues/new/choose)
using the **Bug report** template. Include:

- A minimal reproduction: exact command, dataset, method, window length.
- Full traceback (not a screenshot).
- `python -V`, `uv --version`, GPU model, CUDA version.
- Whether the issue reproduces with the default Ray Tune sweep or only a specific driver.

## 3. Requesting features

Use the **Feature request** issue template and describe:

- The scientific / engineering motivation.
- Whether the change affects the paper numbers (if yes, highlight it prominently).
- Backward-compatibility implications (the benchmark must stay reproducible).

## 4. Reproducing the paper

If your report is "I cannot reproduce the numbers in Table 3 / Fig. 4 / …", please use the
dedicated **Paper reproduction** issue template. Tell us:

- Which figure / table / row.
- Which checkpoint directory (Ray Tune trial hash) you evaluated.
- Your `ray_evaluation.py` `settings` block.
- The CSV file you compared against.

## 5. Pull requests

1. Create a topic branch off `main`: `git checkout -b feat/<short-slug>`.
2. Keep the diff focused. Large refactors belong in their own PR.
3. Add or update docstrings for every public function you touch. Do not add narrative
   comments inside functions unless the *why* is non-obvious.
4. Run `uvx ruff format .` and `uvx ruff check .` before pushing.
5. If you touched training / evaluation logic, run the relevant entry point on one dataset
   and attach the command + output summary in the PR description.
6. Update `CHANGELOG.md` under `[Unreleased]` with a one-line bullet.
7. Open the PR against `main`; fill in the PR template.

### Commit message style

```
<type>(<optional scope>): <subject>
```

Types we use: `feat`, `fix`, `refactor`, `docs`, `chore`, `style`, `test`, `perf`, `build`.

Example: `fix(util): use inner join in merge_graphlet_dfs to avoid NaN at W > trained_wl`

## 6. What NOT to commit

- Ray Tune trial directories (`swissrivernetwork/benchmark/outputs/ray_results/`). Use
  `outputs/trim_checkpoints.py` to prune.
- Weights & Biases runs (`wandb/`).
- `dump/` prediction CSVs.
- `.venv/`, `__pycache__/`, `.idea/`, `.ruff_cache/`, `.pytest_cache/`.
- `.env` files or API keys (see `SECURITY.md`).

## 7. Working with assistive coding agents

This repo ships with a Claude Code `CLAUDE.md` and skill files under `.claude/skills/`.
If you use Claude Code, Codex, Gemini CLI, or Copilot CLI, those already encode the
preferred workflow. See `README.md` → *Install path D (LLM-agent)* for the paste-and-run
onboarding prompt.

## 8. Code of Conduct

Participation in this project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 9. Licensing

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers the project.
