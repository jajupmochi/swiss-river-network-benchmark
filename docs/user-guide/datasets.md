# Datasets

The benchmark ships three graph datasets.

| Graph | Stations | Time span | Source |
| --- | --- | --- | --- |
| `swiss-1990` | ~40 | 1990-01-01 → 2022-12-31 | FOEN hydrometric stations |
| `swiss-2010` | ~65 | 2010-01-01 → 2022-12-31 | FOEN hydrometric stations |
| `zurich` | ~10 | 1981-01-01 → 2022-12-31 | Zürich AWEL |

## Generating the splits

```bash
uv run srn prepare-data
```

Writes per-graph CSVs under `swissrivernetwork/benchmark/dump/`:

- `<graph>_train.csv`, `<graph>_test.csv` — wide-format tables where
  columns are station IDs and rows are daily timestamps.
- `graph_<graph>.pth` — the PyTorch-Geometric graph (edges + node
  features).

## Features per timestep

Each station column is a timeseries of daily mean values. Edge features
encode upstream / downstream distance on the river network.

## Provenance

Raw measurements come from the Swiss Federal Office for the Environment
(FOEN) and the Zürich Amt für Abfall, Wasser, Energie und Luft (AWEL).
See the [Acknowledgments](../index.md) for the full attribution.

!!! warning "Data policy"
    Raw station files are *not* redistributed via this repository or
    any release artefact. `srn prepare-data` assumes you have obtained
    a copy directly from the source. Contact the authors if you need
    help locating a public mirror.
