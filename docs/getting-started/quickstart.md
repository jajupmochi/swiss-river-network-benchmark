# Quickstart

Run each step from the repo root after `uv sync --no-cache`.

## 1. Prepare the three datasets

```bash
uv run srn prepare-data
```

Produces `swissrivernetwork/benchmark/dump/<graph>_{train,test}.csv`.

## 2. Tune one method on one graph

```bash
uv run srn tune -m lstm -g swiss-2010 -n 50 -wl 90
```

- `-m` — method name (see [methods](../user-guide/methods.md)).
- `-g` — graph name (one of `swiss-1990`, `swiss-2010`, `zurich`).
- `-n` — Ray Tune trial count.
- `-wl` — training window length.

Checkpoints land under `swissrivernetwork/benchmark/outputs/ray_results/`.

## 3. Evaluate

```bash
uv run srn evaluate
```

Writes per-station and per-method tables under
`swissrivernetwork/benchmark/visualize_results/outputs/stations/`.

## 4. Sweep window lengths

```bash
uv run srn sweep
```

Two phases (ISOLATED → GRAPHLET) write
`dump/predictions/<path_extra_keys>-evalwl{W}/` files and the CSVs under
`visualize_results/outputs/win_lens/`.

!!! warning "Eval-window leakage"
    Earlier sweep runs suffered from two bugs that are fixed on `main`
    since commit `4daeff3`. See
    [Explainers](../explainers/eval-window-leakage.md) before trusting
    numbers from pre-fix runs.

## 5. Visualise

```bash
uv run jupyter lab swissrivernetwork/benchmark/visualize_results/
```

Every notebook under that directory reproduces one paper figure from the
CSVs you just wrote.
