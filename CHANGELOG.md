# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community / OSS metadata: `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue and pull-request templates, Dependabot
  config.
- `swissrivernetwork/cli.py` typer-based command-line entry (`srn …`) that forwards to the
  existing `swissrivernetwork.benchmark.*` drivers without modifying their implementations.
- Gradio Space app (`swissrivernetwork/app/gradio_app.py`) with live interactive visualization.
- Streamlit local UI (`swissrivernetwork/app/streamlit_app.py`) with Explore / Predict /
  Compare tabs that reuse the existing visualization code.
- Desktop installers (Windows `.exe`, macOS `.dmg`, Linux `.AppImage`) built via PyInstaller
  in a cross-platform GitHub Actions matrix.
- LLM-agent install path: Claude Code skill at `.claude/skills/install/SKILL.md` plus a
  paste-once prompt for Claude Code / Codex / Gemini / Copilot CLI.
- Docker image with CUDA base + docker-compose stack for training, Jupyter, and the UI.
- MkDocs Material documentation site with English / 中文 / Deutsch / Français via
  `mkdocs-static-i18n` and versioning via `mike`.
- Multi-language README: `README.md` (English, default), `README.zh.md`, `README.de.md`,
  `README.fr.md`.
- Regression test pinning the `merge_graphlet_dfs` inner-join fix.

### Changed
- Rewrote `README.md` with badges, TL;DR, hero banner, feature list, multi-path install
  (uv / pip / docker / desktop / LLM-agent), reproducing-the-paper section, results gallery,
  citation, and language switcher.
- `pyproject.toml` gained authors, URLs, classifiers, keywords, optional extras
  (`viz`, `app`, `dev`, `docs`, `all`), and the `srn` console script.

### Fixed
- *(from prior release cycle, included for completeness)* Eval-window-length leakage: the
  isolated `wt_hat` dump path now carries an `-evalwl{W}` suffix when `eval_wl ≠ trained_wl`,
  preventing later sweep steps from overwriting earlier ones and silently giving Graphlet a
  long-history advantage at short eval windows. See `util.get_evaluation_path_keys`.
- *(from prior release cycle)* `merge_graphlet_dfs` now uses an inner join so the target
  time axis intersects the neighbour dumps. Previously an outer join kept target-only days
  whose neighbour features were NaN, raising `AssertionError: NaN in neigh …_wt_hat` at
  `W > trained_wl`.
- Fresh-kernel vs. repeat-run rendering inconsistency in `results_in_polar.ipynb` (radar
  plot spines were black on the first run and light grey on later runs): `sns.set_theme(...)`
  now runs before `plt.subplots(...)` so rcParams are stable.

## [0.1.0] — 2026-04-21

Initial public release accompanying the ICPR 2026 submission
*"Benchmarking Transformers on Spatio-Temporal River Water Temperature Modeling"*.

- Three graph datasets: `swiss-1990`, `swiss-2010`, `zurich`.
- Eight methods: `lstm`, `graphlet`, `lstm_embedding`, `stgnn`, `transformer`,
  `transformer_graphlet`, `transformer_embedding`, `transformer_stgnn`.
- Ray Tune hyperparameter search (`ray_tune.py`) and evaluation (`ray_evaluation.py`).
- Window-length sweep driver (`run_win_len_sweep.py`) producing paper Fig. 4 and the HLE
  dimension of Fig. 2.
- Gaussian / impulse noise robustness sweep.
- Notebooks under `swissrivernetwork/benchmark/visualize_results/` that produce every paper
  figure.

[Unreleased]: https://github.com/jajupmochi/swiss-river-network-benchmark/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jajupmochi/swiss-river-network-benchmark/releases/tag/v0.1.0
