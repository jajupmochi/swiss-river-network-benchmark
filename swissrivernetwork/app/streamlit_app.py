"""Streamlit UI for the Swiss River Network Benchmark.

Launched with ``uv run srn app streamlit``. Three tabs:

* **Explore** — inspect the raw water temperature splits per graph.
* **Predict** — preview the cached predictions dumped by the window-length
  sweep (read-only; no training is triggered).
* **Compare** — overlay the window-length sweep across methods and
  graphs, directly from the CSVs used in the paper.

Everything is read-only against the on-disk artefacts under
``swissrivernetwork/benchmark/``: this is a *visualisation* surface, not a
training UI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
OUTPUTS_DIR = BENCHMARK_DIR / "visualize_results" / "outputs"
WIN_LENS_DIR = OUTPUTS_DIR / "win_lens"
STATIONS_DIR = OUTPUTS_DIR / "stations"
DUMP_DIR = BENCHMARK_DIR / "dump"
PREDICTIONS_DIR = DUMP_DIR / "predictions"

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
METRICS = [
    "RMSE_Mean",
    "RMSE_Median",
    "MAE_Mean",
    "MAE_Median",
    "NSE_Mean",
    "NSE_Median",
]


@st.cache_data(show_spinner=False)
def load_raw_split(graph: str, split: str = "train") -> pd.DataFrame | None:
    path = DUMP_DIR / f"{graph}_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in {"time", "date", "datetime"}), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.rename(columns={time_col: "time"})
    return df


@st.cache_data(show_spinner=False)
def load_win_lens(graph: str, method: str) -> pd.DataFrame | None:
    path = WIN_LENS_DIR / f"{graph}_{method}_win_lens_resu.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["method"] = method
    df["graph"] = graph
    return df


@st.cache_data(show_spinner=False)
def load_stations_csv(graph: str, method: str) -> pd.DataFrame | None:
    for suffix in ("_station_resu.csv", "_station_resu_f.csv"):
        path = STATIONS_DIR / f"{graph}_{method}{suffix}"
        if path.exists():
            return pd.read_csv(path)
    return None


def _station_cols(df: pd.DataFrame) -> list[str]:
    ignore = {"time", "date", "datetime"}
    return [c for c in df.columns if c.lower() not in ignore]


def tab_explore() -> None:
    st.subheader("Raw river temperature — train / test splits")
    col_a, col_b = st.columns(2)
    graph = col_a.selectbox("Graph", GRAPHS, index=0)
    split = col_b.radio("Split", ["train", "test"], horizontal=True)

    df = load_raw_split(graph, split)
    if df is None:
        st.warning(
            f"{graph}_{split}.csv not found under `swissrivernetwork/benchmark/dump/`. "
            "Run `uv run srn prepare-data` to generate the splits."
        )
        return

    cols = _station_cols(df)
    chosen = st.multiselect(
        "Stations to overlay",
        cols,
        default=cols[:3],
        help="Column names in the raw CSV; mapping to station IDs is dataset-specific.",
    )
    if not chosen:
        st.info("Pick at least one station.")
        return

    plot_df = df[["time", *chosen]].melt(id_vars="time", var_name="station", value_name="value")
    fig = px.line(plot_df, x="time", y="value", color="station", title=f"{graph} — {split} split")
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview the first rows of the raw CSV"):
        st.dataframe(df.head(50), use_container_width=True)


def tab_predict() -> None:
    st.subheader("Cached predictions (from the sweep)")
    st.caption(
        "Reads `dump/predictions/<path_extra_keys>-evalwl{W}/` files written by "
        "`srn sweep`. No training is triggered in this tab."
    )

    col_a, col_b, col_c = st.columns(3)
    graph = col_a.selectbox("Graph", GRAPHS, index=0, key="pred_graph")
    method = col_b.selectbox("Method", METHODS, index=4, key="pred_method")
    eval_wl = col_c.slider("Eval window length", 1, 450, 90, step=15)

    # Try to find any matching prediction CSV under the evalwl-tagged directories.
    candidates = list(PREDICTIONS_DIR.glob(f"*evalwl{eval_wl}/{graph}_{method}_*.csv"))
    if not candidates:
        st.info(
            f"No prediction CSV found for `{graph}`/`{method}` at W={eval_wl}. "
            "Run `uv run srn sweep` to populate `dump/predictions/`."
        )
        return

    path = st.selectbox("Prediction file", candidates, format_func=lambda p: p.name)
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in {"time", "date", "datetime"}), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    y_cols = [c for c in df.columns if c != time_col]
    chosen = st.multiselect("Series", y_cols, default=y_cols[:2])
    if chosen:
        plot_df = df[[time_col, *chosen]].melt(id_vars=time_col, var_name="series", value_name="wt")
        fig = px.line(plot_df, x=time_col, y="wt", color="series", title=path.name)
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Preview rows"):
        st.dataframe(df.head(100), use_container_width=True)


def tab_compare() -> None:
    st.subheader("Compare methods — window-length sweep")
    col_a, col_b = st.columns(2)
    graph = col_a.selectbox("Graph", GRAPHS, index=0, key="cmp_graph")
    metric = col_b.selectbox("Metric", METRICS, index=0, key="cmp_metric")
    methods = st.multiselect("Methods", METHODS, default=METHODS[:4], key="cmp_methods")

    frames = [load_win_lens(graph, m) for m in methods]
    frames = [f for f in frames if f is not None and metric in f.columns]
    if not frames:
        st.warning(f"No sweep CSV found under `{WIN_LENS_DIR}` for the selected methods.")
        return

    df = pd.concat([f[["window_len", metric, "method"]] for f in frames], ignore_index=True)
    fig = px.line(
        df,
        x="window_len",
        y=metric,
        color="method",
        markers=True,
        title=f"{graph} — {metric} vs. window length",
        labels={"window_len": "window length (days)"},
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-station bar plot"):
        method = st.selectbox("Method for per-station view", methods, index=0, key="cmp_perstation")
        df_st = load_stations_csv(graph, method)
        if df_st is None:
            st.info(f"No per-station CSV found for `{graph}`/`{method}`.")
        else:
            station_col = next(
                (c for c in df_st.columns if c.lower() in {"station", "stations"}),
                df_st.columns[0],
            )
            metric_col = (
                metric
                if metric in df_st.columns
                else next(
                    (c for c in df_st.columns if c.lower().startswith(metric.split("_")[0].lower())),
                    None,
                )
            )
            if metric_col is None:
                st.info(f"Metric {metric!r} not in the per-station CSV.")
            else:
                plot = df_st[[station_col, metric_col]].dropna().sort_values(metric_col)
                fig2 = px.bar(plot, x=station_col, y=metric_col)
                fig2.update_layout(template="plotly_white", height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Swiss River Network Benchmark",
        page_icon="🌊",
        layout="wide",
    )
    st.title("🌊 Swiss River Network Benchmark")
    st.caption("ICPR 2026 · reproducible benchmark for spatio-temporal river water-temperature modelling")

    explore, predict, compare = st.tabs(["Explore", "Predict", "Compare"])
    with explore:
        tab_explore()
    with predict:
        tab_predict()
    with compare:
        tab_compare()

    with st.sidebar:
        st.markdown(
            """
            ### Links

            - [📦 GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark)
            - [🤗 HF Space](https://huggingface.co/spaces/jajupmochi/swiss-river-network-benchmark)
            - [📚 Docs](https://jajupmochi.github.io/swiss-river-network-benchmark/)
            - [📄 Paper (ICPR 2026)](#)

            ### Tips

            - `Explore` needs raw splits under `benchmark/dump/`.
            - `Predict` needs dumps from `srn sweep`.
            - `Compare` works out of the box (CSVs are in the repo).
            """
        )


if __name__ == "__main__":
    main()
