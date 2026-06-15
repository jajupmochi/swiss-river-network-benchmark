# `packaging/` — desktop installer assets

This folder ships everything needed to build a UI-only desktop bundle of the
Swiss River Network Benchmark for Windows / macOS / Linux. The bundle runs the
Streamlit UI against the checked-in paper CSVs; it does **not** ship PyTorch,
Ray, or any training-grade dependency.

| File | Purpose |
| --- | --- |
| `launch_desktop.py` | Entry script run by the frozen binary — picks a free port, opens the browser, then hands off to Streamlit. |
| `swissrivernetwork.spec` | PyInstaller spec for all three platforms. |
| `entitlements.plist` | macOS entitlements for code signing (unused until a release cert is configured). |

## Building locally

```bash
uv sync --all-extras
uv run pyinstaller packaging/swissrivernetwork.spec --clean --noconfirm
```

Artefacts land in `dist/SwissRiverNetworkBenchmark/`. Zip the folder before
uploading it to a release.

## CI

`.github/workflows/release.yml` runs the same command on `ubuntu-latest`,
`macos-latest` (Apple Silicon), and `windows-latest` whenever a tag of the
form `v*.*.*` is pushed. The three artefacts are attached to the GitHub
release automatically. On Linux we additionally package the `dist/` tree into
an AppImage with `appimagetool` if available.

## Why UI-only?

Training requires CUDA wheels (`torch`, `torch_geometric`, `ray`) that sum to
several GB and only make sense on a GPU-equipped machine. Shipping them in a
double-click installer would be wasteful and brittle. Users who want to *run*
the benchmark should follow install paths A / B / C in the top-level README.
