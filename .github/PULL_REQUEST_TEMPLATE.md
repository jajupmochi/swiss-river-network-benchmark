<!-- Thanks for contributing! Please fill in the sections that apply. -->

## Summary

<!-- One or two sentences: what does this PR change and why? -->

## Type of change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `refactor` — code restructuring without behaviour change
- [ ] `docs` — documentation only
- [ ] `chore` — build, CI, tooling, or dependencies
- [ ] `perf` — performance improvement
- [ ] `test` — add or update tests

## Impact on paper numbers

- [ ] **No** — this PR cannot change any reported metric.
- [ ] **Yes** — this PR affects a metric; I have listed the affected rows/figures below and
      attached before/after numbers.

<!-- If Yes: list which CSVs / figures / tables change. -->

## How I tested

<!--
Describe what you ran locally. Examples:
- `uvx ruff format --check .` and `uvx ruff check .`
- `uv run pytest -q`
- `uv run python -m swissrivernetwork.benchmark.run_win_len_sweep` (DEBUG_SINGLE on zurich/graphlet/wl=30)
- Built the docs: `uv run mkdocs build`
-->

## Checklist

- [ ] I opened / referenced an issue for non-trivial changes.
- [ ] Commits follow Conventional Commits style.
- [ ] Ruff format + check pass locally.
- [ ] Docstrings / type hints updated for touched public APIs.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.
- [ ] No secrets / dataset credentials in the diff.
- [ ] The Ray Tune `outputs/ray_results/` directory, `dump/`, and `wandb/` are untouched.

## Screenshots / logs (optional)

<!-- Paste relevant log excerpts or screenshots of figures. -->
