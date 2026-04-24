# API reference

The API reference is generated from in-source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/).

- [`swissrivernetwork.benchmark.util`](benchmark-util.md) — helper
  functions that glue the pipeline together, including the load-bearing
  `merge_graphlet_dfs`.
- [`swissrivernetwork.benchmark.dataset`](benchmark-dataset.md) —
  readers for raw / prediction CSVs and the windowed datasets used by
  the training loops.
- [`swissrivernetwork.benchmark.model`](benchmark-model.md) — the
  LSTM / Transformer / ST-GNN model classes.

!!! note "Partial coverage"
    Only the modules linked above have formal docstrings today. The
    rest of the codebase is documented through module-level comments
    and the [developer guide](../developer-guide/index.md). New
    docstrings are welcome — see [CONTRIBUTING.md](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/CONTRIBUTING.md).
