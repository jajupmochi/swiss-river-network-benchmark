"""Gradio demo for the Swiss River Network Benchmark.

Serves a small, self-contained UI that works both:

* on the Hugging Face Space (no GPU, no training) — binds to
  ``0.0.0.0:7860`` by default and exposes three tabs powered by the
  real CSVs produced by the paper's window-length sweep; and
* locally via ``uv run srn app gradio`` for the same workflow on a
  developer machine.

**Data policy.** The app is strictly read-only: it reads the CSVs under
``swissrivernetwork/benchmark/visualize_results/outputs/`` (checked in by
the paper authors) and the raw train/test splits under
``swissrivernetwork/benchmark/dump/`` (not checked in). If the raw
splits are missing, the *Explore* tab shows a friendly placeholder and
the rest of the app still works from the CSV summaries.

The visualization is done with Plotly so the same figures work in the
HF Space (no matplotlib backend) and locally.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
OUTPUTS_DIR = BENCHMARK_DIR / "visualize_results" / "outputs"
WIN_LENS_DIR = OUTPUTS_DIR / "win_lens"
STATIONS_DIR = OUTPUTS_DIR / "stations"
DUMP_DIR = BENCHMARK_DIR / "dump"

GRAPHS = ["swiss-1990", "swiss-2010", "zurich"]
METHODS = [
    "lstm",
    "lstm_embedding",
    "graphlet",
    "stgnn",
    "transformer",
    "transformer_embedding",
    "transformer_graphlet",
    "transformer_stgnn",
]
DEFAULT_METRIC = "RMSE_Mean"
METRIC_CHOICES = [
    "RMSE_Mean",
    "RMSE_Median",
    "MAE_Mean",
    "MAE_Median",
    "NSE_Mean",
    "NSE_Median",
]


def _win_lens_csv(graph: str, method: str) -> Path:
    return WIN_LENS_DIR / f"{graph}_{method}_win_lens_resu.csv"


def _load_win_lens(graph: str, method: str) -> pd.DataFrame | None:
    path = _win_lens_csv(graph, method)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["method"] = method
    df["graph"] = graph
    return df


def plot_win_lens_sweep(graph: str, methods: list[str], metric: str) -> go.Figure:
    """Plot the window-length sweep for the selected methods on one graph."""
    frames: list[pd.DataFrame] = []
    for m in methods:
        df = _load_win_lens(graph, m)
        if df is not None and metric in df.columns:
            frames.append(df[["window_len", metric, "method"]])
    if not frames:
        return _empty_figure(f"No CSV found under {WIN_LENS_DIR} for the selected methods.")
    df_all = pd.concat(frames, ignore_index=True)
    fig = px.line(
        df_all,
        x="window_len",
        y=metric,
        color="method",
        markers=True,
        title=f"{graph} — {metric} vs. window length",
        labels={"window_len": "window length (days)", metric: metric},
    )
    fig.update_layout(legend_title_text="method", height=500, template="plotly_white")
    return fig


def _station_csv(graph: str, method: str) -> Path:
    path = STATIONS_DIR / f"{graph}_{method}_station_resu.csv"
    if path.exists():
        return path
    alt = STATIONS_DIR / f"{graph}_{method}_station_resu_f.csv"
    return alt if alt.exists() else path


def plot_per_station(graph: str, method: str, metric: str) -> go.Figure:
    path = _station_csv(graph, method)
    if not path.exists():
        return _empty_figure(f"No per-station CSV for {graph}/{method}. Run the sweep and re-render figures.")
    df = pd.read_csv(path)
    # Heuristic: find a station/column and a metric column.
    station_col = next((c for c in df.columns if c.lower() in {"station", "stations"}), df.columns[0])
    # Pick closest metric, otherwise first numeric column.
    metric_col = (
        metric
        if metric in df.columns
        else next(
            (c for c in df.columns if c.lower().startswith(metric.split("_")[0].lower())),
            None,
        )
    )
    if metric_col is None:
        return _empty_figure(f"Metric {metric!r} not found in {path.name}.")
    df_plot = df[[station_col, metric_col]].dropna().sort_values(metric_col)
    fig = px.bar(
        df_plot,
        x=station_col,
        y=metric_col,
        title=f"{graph} / {method} — {metric_col} per station",
        labels={station_col: "station", metric_col: metric_col},
    )
    fig.update_layout(height=500, template="plotly_white", xaxis_tickangle=-45)
    return fig


def plot_raw_series(graph: str, n_stations: int) -> go.Figure:
    path = DUMP_DIR / f"{graph}_train.csv"
    if not path.exists():
        return _empty_figure(f"Raw series for {graph} not found at {path}. Run `uv run srn prepare-data` first.")
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in {"time", "date", "datetime"}), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    wt_cols = [c for c in df.columns if c.lower().startswith("wt_") or "_wt" in c.lower()]
    if not wt_cols:
        wt_cols = [c for c in df.columns if c != time_col][:n_stations]
    wt_cols = wt_cols[:n_stations]
    if not wt_cols:
        return _empty_figure("Could not auto-detect water temperature columns.")
    plot_df = df[[time_col] + wt_cols].melt(id_vars=time_col, var_name="station", value_name="wt")
    fig = px.line(
        plot_df,
        x=time_col,
        y="wt",
        color="station",
        title=f"{graph} — water temperature (train split, first {len(wt_cols)} stations)",
        labels={time_col: "time", "wt": "water temperature (°C)"},
    )
    fig.update_layout(height=500, template="plotly_white")
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(template="plotly_white", height=300)
    return fig


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Swiss River Network Benchmark") as demo:
        gr.Markdown(
            """
            # 🌊 Swiss River Network Benchmark — live demo
            Explore the results of the ICPR 2026 benchmark: window-length sweep,
            per-station error distribution, and (if raw splits are available) raw water
            temperature time series.
            """
        )

        with gr.Tab("Compare methods"):
            with gr.Row():
                graph_dd = gr.Dropdown(GRAPHS, value="swiss-1990", label="Graph")
                metric_dd = gr.Dropdown(METRIC_CHOICES, value=DEFAULT_METRIC, label="Metric")
            methods_cb = gr.CheckboxGroup(METHODS, value=METHODS[:4], label="Methods")
            plot = gr.Plot(label="Window-length sweep")
            refresh = gr.Button("Plot", variant="primary")
            refresh.click(plot_win_lens_sweep, inputs=[graph_dd, methods_cb, metric_dd], outputs=plot)
            demo.load(plot_win_lens_sweep, inputs=[graph_dd, methods_cb, metric_dd], outputs=plot)

        with gr.Tab("Per-station error"):
            with gr.Row():
                graph_s = gr.Dropdown(GRAPHS, value="swiss-1990", label="Graph")
                method_s = gr.Dropdown(METHODS, value="transformer_embedding", label="Method")
                metric_s = gr.Dropdown(METRIC_CHOICES, value=DEFAULT_METRIC, label="Metric")
            plot_s = gr.Plot(label="Per-station")
            btn_s = gr.Button("Plot", variant="primary")
            btn_s.click(plot_per_station, inputs=[graph_s, method_s, metric_s], outputs=plot_s)
            demo.load(plot_per_station, inputs=[graph_s, method_s, metric_s], outputs=plot_s)

        with gr.Tab("Raw time series"):
            with gr.Row():
                graph_r = gr.Dropdown(GRAPHS, value="swiss-1990", label="Graph")
                n_stations = gr.Slider(1, 12, value=3, step=1, label="Stations to overlay")
            plot_r = gr.Plot(label="Water temperature")
            btn_r = gr.Button("Plot", variant="primary")
            btn_r.click(plot_raw_series, inputs=[graph_r, n_stations], outputs=plot_r)

        gr.Markdown(
            """
            ---
            Built from the ICPR 2026 submission. CSVs live under
            `swissrivernetwork/benchmark/visualize_results/outputs/`. Source:
            [github.com/jajupmochi/swiss-river-network-benchmark](https://github.com/jajupmochi/swiss-river-network-benchmark).
            """
        )
    return demo


def main() -> None:
    """Entry point used by the ``srn app gradio`` CLI."""
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
