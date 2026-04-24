# For hydrologists

This page walks you through the benchmark without using a terminal.

## 1. Install the desktop app

Grab the release artefact from the
[Releases page](https://github.com/jajupmochi/swiss-river-network-benchmark/releases)
and double-click.

| Platform | File |
| --- | --- |
| Windows 10/11 | `.zip` → unzip → double-click `SwissRiverNetworkBenchmark.exe` |
| macOS (Apple Silicon) | `.tar.gz` → unpack → open the `.app` |
| Linux | `.tar.gz` → unpack → run `./SwissRiverNetworkBenchmark` |

Your browser opens at `http://localhost:8501`.

## 2. Explore a river

In the **Explore** tab, pick a graph (`swiss-1990`, `swiss-2010`, or
`zurich`) and one or more stations. The chart overlays the raw
water-temperature measurements for the training split.

## 3. Compare methods

The **Compare** tab plots RMSE / MAE / NSE against window length for
your chosen methods. Use this to reproduce the behaviour behind paper
Figure 4.

## 4. Inspect individual stations

In **Predict** you can point at a cached prediction file and overlay it
against the ground-truth series. Useful when you want to see where a
given method fails on a specific river.

## When to switch to the code path

- You want to *train* a model on new data → follow
  [Quickstart](../../getting-started/quickstart.md).
- You want to hack on a new method → read the
  [developer guide](../../developer-guide/index.md).
