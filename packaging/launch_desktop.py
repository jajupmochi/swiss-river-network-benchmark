"""Entry point that the PyInstaller-packaged desktop app runs.

The bundle shipped on the Releases page is a thin launcher: it starts
the Streamlit UI on a free local port and opens the default browser.
There's no training-grade dependency wall — only what's needed to drive
the UI against bundled, precomputed CSVs.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _pick_port(default: int = 8501) -> int:
    """Return ``default`` if free, otherwise the first free port above it."""
    port = default
    while port < default + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return default


def _open_browser(url: str, delay: float = 1.5) -> None:
    def _go() -> None:
        time.sleep(delay)
        webbrowser.open_new(url)

    threading.Thread(target=_go, daemon=True).start()


def main() -> None:
    here = Path(__file__).resolve().parent
    # When frozen by PyInstaller, the module sits in the bundle root next to
    # the streamlit script. When run from source, it sits in `packaging/`.
    app_path_candidates = [
        here / "streamlit_app.py",
        here.parent / "swissrivernetwork" / "app" / "streamlit_app.py",
    ]
    app_path = next((p for p in app_path_candidates if p.exists()), None)
    if app_path is None:
        sys.stderr.write("Bundled streamlit_app.py not found; aborting.\n")
        raise SystemExit(2)

    port = _pick_port(int(os.environ.get("SRN_DESKTOP_PORT", "8501")))
    url = f"http://localhost:{port}"
    _open_browser(url)

    # Streamlit's CLI parser reads argv, so swap it and hand off.
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    from streamlit.web import cli as stcli

    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
