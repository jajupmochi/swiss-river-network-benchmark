#!/usr/bin/env bash
# One-click launcher for the Swiss River Network Benchmark workbench (macOS).
#
# Double-click this file in Finder. macOS opens it in Terminal, installs everything
# on first run, and opens the app in your web browser. No coding required.
# (If macOS blocks it: right-click → Open the first time, or run
#  `xattr -d com.apple.quarantine launch-workbench.command`.)
set -e
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "First run: installing the uv package manager (one-time)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "Starting the workbench — a browser tab will open at http://127.0.0.1:7860"
echo "(the first run downloads dependencies and can take a few minutes). Press Ctrl+C to stop."
export SRN_INBROWSER=1
export SRN_HOST=127.0.0.1
exec uv run --extra app python -m swissrivernetwork.app.workbench
