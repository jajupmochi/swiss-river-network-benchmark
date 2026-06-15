# FAQ

## Why do I need a GPU?

The benchmark's PyTorch + PyTorch-Geometric stack is built around CUDA
tensors. There is no CPU fallback path — both the Transformer and the
ST-GNN implementations call `.cuda()` directly. If you only want to
explore the pre-computed figures, use the desktop installer or the
Hugging Face Space; neither requires a GPU.

## Why `uv` and not `pip` / `conda`?

The lockfile (`uv.lock`) is the source of truth and guarantees
reproducible installs across contributors and CI. `uv` is the only
tool that reads and writes that lockfile without drift. We still
support `pip` as an install path (see README section B) but it is not
used for CI or for reproducing paper numbers.

## My sweep rows at `W > 90` look wrong — why?

You are probably on a pre-`4daeff3` binary. Upgrade to `main` and re-run
the graphlet methods at `W ≠ 90`. Details:
[eval-window leakage](explainers/eval-window-leakage.md) and
[graphlet NaN fix](explainers/graphlet-nan-fix.md).

## Can I redistribute the raw station data?

No. Raw station measurements belong to FOEN and AWEL. This repository
ships only the code, the derived aggregated CSVs used by the paper
figures, and placeholder visual assets. The desktop installer likewise
ships no raw station data. Obtain the measurements directly from the
sources.

## How do I add a new method?

1. Create a new file under `swissrivernetwork/benchmark/` (don't edit
   the existing drivers in place).
2. Add a factory / training function modelled on `train_transformer` in
   `train_isolated_station.py`.
3. Register it with the `method` dispatch in the CLI / drivers.
4. Add its hyperparameter search space to `ray_tune.py`.
5. PR — see [CONTRIBUTING.md](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/CONTRIBUTING.md).
