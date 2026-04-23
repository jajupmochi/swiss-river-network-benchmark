"""Interactive demo apps for the Swiss River Network Benchmark.

Two entry points live here:

* :mod:`swissrivernetwork.app.gradio_app` — the Gradio demo used for the
  Hugging Face Space and for a one-command local demo.
* :mod:`swissrivernetwork.app.streamlit_app` — a richer local UI with
  Explore / Predict / Compare tabs that reuse the existing visualization
  code in :mod:`swissrivernetwork.benchmark.visualize_results`.

Both apps are intentionally read-only against existing artefacts: they
never trigger training and never modify files under
``swissrivernetwork/benchmark/outputs``. Prediction calls reuse the
checkpoints produced by :mod:`swissrivernetwork.benchmark.ray_tune`.
"""
