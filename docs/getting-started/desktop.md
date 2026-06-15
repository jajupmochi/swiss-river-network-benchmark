# Desktop app

The desktop installer is a UI-only bundle of the Streamlit app. It runs
on a laptop without a GPU and points at the paper CSVs bundled with the
release.

## Download

Grab the latest artefact from the
[Releases page](https://github.com/jajupmochi/swiss-river-network-benchmark/releases):

| Platform | Artefact |
| --- | --- |
| Windows 10/11 x64 | `SwissRiverNetworkBenchmark-<ver>-win-x86_64.zip` |
| macOS (Apple Silicon) | `SwissRiverNetworkBenchmark-<ver>-macos-arm64.tar.gz` |
| Linux x64 | `SwissRiverNetworkBenchmark-<ver>-linux-x86_64.tar.gz` |

Unzip, open the `SwissRiverNetworkBenchmark` folder, and double-click the
platform binary. The bundled browser tab opens at
`http://localhost:8501`.

## Build locally

```bash
uv sync --all-extras
uv run pyinstaller packaging/swissrivernetwork.spec --clean --noconfirm
```

See [`packaging/README.md`](https://github.com/jajupmochi/swiss-river-network-benchmark/blob/main/packaging/README.md)
for the details.

!!! note "Training still needs a GPU"
    The installer is deliberately CPU-only. Use installation path A / B
    / C if you need to train your own models.
