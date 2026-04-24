# Train an LSTM on one station

This tutorial trains the `lstm` baseline on one Swiss station for a few
minutes. Use it as a smoke test after a fresh install.

## Prerequisites

- Finished [Quickstart](../getting-started/quickstart.md) step 1
  (`srn prepare-data` has produced the CSV splits).
- A CUDA GPU.

## Run

```bash
uv run srn train-isolated
```

The `__main__` block of
`swissrivernetwork/benchmark/train_isolated_station.py` has
`settings["dev_run"] = True` by default, which:

- caps training data to 4 windows,
- caps validation to 4 windows,
- disables Weights & Biases.

The run should finish in under a minute.

## Switch to a real training run

Open the file and set:

```python
settings = {
    "dev_run": False,
    "enable_wandb": True,  # or False if you don't have an API key
}

config["station"]  = "2091"     # pick any station from read_stations(graph_name)
config["graph_name"] = "swiss-1990"
config["epochs"]   = 30
```

Then rerun `uv run srn train-isolated`. Checkpoints land in
`outputs/ray_results/<run_id>/`.

## Inspect the result

Open the `visualize_results/visualize_results.ipynb` notebook and set
`graph_name = "swiss-1990"`, `method = "lstm"`, `station = "2091"`.
