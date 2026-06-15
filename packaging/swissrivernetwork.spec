# PyInstaller spec for the Swiss River Network Benchmark desktop bundle.
#
# Build it from the repo root:
#
#     uv sync --all-extras
#     uv run pyinstaller packaging/swissrivernetwork.spec --clean --noconfirm
#
# GitHub Actions picks this up for Linux / macOS / Windows runners (see
# `.github/workflows/release.yml`).
#
# The bundle is intentionally UI-only: it starts the Streamlit app with the
# in-repo paper CSVs, and does *not* carry CUDA wheels. Training workflows
# use the `uv`/`pip`/Docker install paths instead.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

REPO_ROOT = Path(os.getcwd())
APP_DIR = REPO_ROOT / "swissrivernetwork" / "app"
BENCHMARK_VIZ = REPO_ROOT / "swissrivernetwork" / "benchmark" / "visualize_results" / "outputs"
ASSETS = REPO_ROOT / "assets"

# Streamlit + Plotly ship runtime assets that PyInstaller needs collected.
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
plotly_datas = collect_data_files("plotly")
altair_datas = collect_data_files("altair")

datas = [
    # The Streamlit app lives next to the launcher inside the bundle.
    (str(APP_DIR / "streamlit_app.py"), "."),
    # The read-only CSVs behind the paper figures.
    (str(BENCHMARK_VIZ), "swissrivernetwork/benchmark/visualize_results/outputs"),
    # Logo + banner for the window icon + "About" page.
    (str(ASSETS / "logo"), "assets/logo"),
    (str(ASSETS / "social"), "assets/social"),
]
datas += streamlit_datas
datas += plotly_datas
datas += altair_datas

hiddenimports = list(streamlit_hiddenimports) + [
    "swissrivernetwork",
    "swissrivernetwork.app",
    "swissrivernetwork.app.streamlit_app",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "pandas",
    "numpy",
]

a = Analysis(
    [str(REPO_ROOT / "packaging" / "launch_desktop.py")],
    pathex=[str(REPO_ROOT)],
    binaries=list(streamlit_binaries),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy wheels not needed for the UI; they would double the bundle size.
        "torch",
        "torchvision",
        "torchaudio",
        "torch_geometric",
        "torch_sparse",
        "torch_scatter",
        "ray",
        "ray.tune",
        "tensorflow",
        "sklearn",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SwissRiverNetworkBenchmark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI; Streamlit opens the browser itself.
    icon=str(ASSETS / "logo" / "favicon.svg") if (ASSETS / "logo" / "favicon.svg").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SwissRiverNetworkBenchmark",
)
