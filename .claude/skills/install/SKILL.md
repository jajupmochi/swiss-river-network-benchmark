---
name: install
description: Paste-once install + smoke-test playbook for the Swiss River Network Benchmark. Use when a user (or agent) asks to install the project from scratch, when bootstrapping a fresh machine/container, or when the README "LLM-agent install" path is requested. Covers prerequisites, `uv sync` install, CLI smoke test, and launching the Streamlit UI.
---

# Installing the Swiss River Network Benchmark

This skill is a **paste-once, agent-friendly playbook** for getting the benchmark from a clean checkout to a running UI in ~5 minutes. A coding agent (Claude Code / Codex / Gemini CLI / Copilot CLI) can execute every step below verbatim.

---

## 0. What you're installing

- `swissrivernetwork/` — Python package: datasets, models, training loops, CLI (`srn`).
- 3 benchmark graphs (`swiss-1990`, `swiss-2010`, `zurich`) and 8 methods.
- Optional extras: `app` (Gradio + Streamlit), `docs` (MkDocs), `dev` (ruff + pytest + notebook tooling).

## 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Linux / macOS / Windows (WSL recommended on Windows) | PyTorch + Ray | `uname -a` |
| Python 3.12+ | PEP 695 generics used internally | `python3 --version` |
| `uv` ≥ 0.5 | The only supported installer | `uv --version` |
| NVIDIA GPU + CUDA 12.x drivers | **Required** — no CPU fallback | `nvidia-smi` |
| Git | to clone | `git --version` |
| 5 GB free disk | wheels + datasets | `df -h .` |

### Install `uv` if missing

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # Linux / macOS
# or:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

### Verify GPU visibility

```bash
nvidia-smi | head -n 5
```

If `nvidia-smi` is missing or prints no device, **stop here** — the benchmark will fail at the first `.cuda()` call. Install matching NVIDIA drivers first.

## 2. Clone

```bash
git clone https://github.com/jajupmochi/swiss-river-network-benchmark.git
cd swiss-river-network-benchmark
```

## 3. Install (the only supported path)

```bash
uv sync --no-cache --all-extras
```

What this does:

- Creates `.venv/` in the repo.
- Installs PyTorch + PyTorch Geometric (CUDA-linked), Ray Tune, benedict, typer, Gradio, Streamlit, MkDocs Material, and all dev tooling.
- Builds the `swissrivernetwork` package in editable mode.
- Registers the `srn` console script on the venv's PATH.

**Never use `pip install` by itself.** `pyproject.toml` is the source of truth and `uv` locks dependencies via `uv.lock`.

### Minimal install (no UI, no docs)

If you only need the core training/eval scripts:

```bash
uv sync --no-cache
```

### App-only install (for running the demos without dev tooling)

```bash
uv sync --no-cache --extra app
```

## 4. Smoke test

Run these in order. Stop at the first failure — each step depends on the previous one.

### 4a. CLI help works (no GPU touched)

```bash
uv run srn --help
```

Expected: typer prints a table with `prepare-data`, `tune`, `evaluate`, `sweep`, `train-single`, `train-isolated`, `app`, and `version`.

### 4b. Version stamp

```bash
uv run srn version
```

Expected: one line with the installed package version (from `importlib.metadata`).

### 4c. GPU-backed import

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
```

Expected: `cuda: True | devices: >=1`. If `False`, the wheel wasn't linked against CUDA — rerun `uv sync --no-cache --all-extras --reinstall`.

### 4d. Package import

```bash
uv run python -c "import swissrivernetwork, swissrivernetwork.benchmark.util as u; print(swissrivernetwork.__version__); print(u.is_transformer_model('transformer_graphlet'))"
```

Expected: a version string followed by `True`.

### 4e. (Optional) pytest smoke

```bash
uv run pytest -q --maxfail=1 --disable-warnings 2>&1 | tail -20
```

Expected: all tests pass, or an informative skip if no GPU-visible tests are configured. If this fails because no test files exist yet, that's fine — the repo's CI style relies on entry points, not unit tests.

## 5. Launch the UI (optional, but a good end-to-end check)

### 5a. Local Streamlit (recommended for first run)

```bash
uv run srn app streamlit
```

Opens `http://localhost:8501`. You should see three tabs: **Explore**, **Predict**, **Compare**.

### 5b. Local Gradio (same models, simpler UI)

```bash
uv run srn app gradio
```

Opens `http://localhost:7860`.

### 5c. Stop the UI

`Ctrl+C` in the terminal. No cleanup needed.

## 6. (Optional) Prepare the dataset and run a tiny training

Only do this if the user explicitly asked to train something — it downloads raw timeseries and takes several minutes.

```bash
uv run srn prepare-data            # fetches + splits; one-time
uv run srn train-isolated          # single-station dev run; edit __main__ config first
```

The `train-isolated` driver's `__main__` block has `settings["dev_run"] = True` by default — it will train on a tiny subset so you can verify the pipeline without burning GPU hours.

## 7. Recovering from common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv: command not found` | uv not on PATH | `source ~/.cargo/env` or restart shell after install |
| `torch.cuda.is_available() == False` | CPU-only wheel got installed | `uv sync --no-cache --all-extras --reinstall` |
| `ModuleNotFoundError: torch_geometric` | partial sync | rerun `uv sync --no-cache --all-extras` |
| `srn: command not found` | venv not activated | prefix with `uv run` (preferred) or `source .venv/bin/activate` |
| `RAY_CHDIR_TO_TRIAL_DIR` warnings | Ray 2.x trial cwd behaviour | **already handled** in `ray_tune.py` — don't "fix" it |
| Streamlit port in use | `8501` busy | `uv run srn app streamlit -- --server.port 8502` |

## 8. What NOT to do

- ❌ Don't call `pip install -e .` — bypasses the lockfile.
- ❌ Don't `conda create` a new env — the project is `uv`-first and `uv.lock` is checked in.
- ❌ Don't edit `swissrivernetwork/benchmark/*.py` to make tests pass — the drivers are the source of truth. Add new code in new files.
- ❌ Don't commit `uv.lock` resolution changes unless the user asked you to bump a dep.
- ❌ Don't delete `RAY_CHDIR_TO_TRIAL_DIR=0` from `ray_tune.py`. It's load-bearing.
- ❌ Don't upload raw data or trained checkpoints anywhere — the repo's policy is "install interface only; user brings their own artifacts".

## 9. After install

- Read `README.md` section **"Reproducing the paper"** for the ICPR 2026 recipe.
- Read `CLAUDE.md` for working-style conventions (Conventional Commits, no test suite, verify via entry points).
- Read `.claude/skills/run-benchmark/SKILL.md` for per-command invocation details.

---

**Success criterion for this skill:** after running sections 1–4, `uv run srn --help` prints the CLI table and `uv run python -c "import torch; assert torch.cuda.is_available()"` returns quietly. Everything else is optional.
