"""Gradio entry point — launches the in-package interactive workbench.

Thin shim so that both ``python -m swissrivernetwork.app.gradio_app`` and the
``srn app gradio`` CLI command launch the full local workbench defined in
:mod:`swissrivernetwork.app.workbench` (map, seasonality, 3D network, forecast
outlook, model comparison, a CPU train/eval sandbox, bring-your-own-model
inference with live streaming, residual / seasonal / threshold analysis, and
multi-format upload).

The workbench module is the single source of truth shared byte-for-byte with the
Hugging Face Space (``tmp/hf-space/app.py``), so local and hosted stay in
lock-step. PyTorch is imported lazily inside the workbench: when a CUDA GPU is
present locally, uploaded TorchScript models run on it; otherwise everything
runs on CPU.
"""

from __future__ import annotations

from swissrivernetwork.app.workbench import build_demo, demo, main

__all__ = ["build_demo", "demo", "main"]

if __name__ == "__main__":
    main()
