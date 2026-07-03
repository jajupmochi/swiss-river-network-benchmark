# API reference

The API reference is generated from in-source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/) — Google-style docstrings, resolved by
static analysis (griffe), so the pages stay in lock-step with the code on every build.

- **[CLI](cli.md)** — the `srn` console entry point.
- **[Benchmark pipeline](benchmark-pipeline.md)** — the data-preparation → tune → evaluate
  → window-length-sweep drivers.
- **[Data & datasets](benchmark-dataset.md)** — readers for raw / prediction CSVs and the
  windowed datasets used by the training loops.
- **[Models](benchmark-model.md)** — the LSTM / Transformer / ST-GNN model classes.
- **[NN layers & training](benchmark-nn.md)** — Transformer / positional-encoding blocks,
  the PyTorch Geometric temporal-convolution layers, and the training loop.
- **[Helpers](benchmark-util.md)** — pipeline glue, including the load-bearing
  `merge_graphlet_dfs`.
- **[Results & tables](results.md)** — per-statistic LaTeX / JSON / PDF export from the
  committed result CSVs.
- **[Interactive workbench](app.md)** — the importable, unit-tested functions behind the
  local / Hugging Face app.
- **[Utilities](utilities.md)** — feature scaling, date conversions, filesystem and plotting
  helpers.
