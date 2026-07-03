"""Swiss River Network Benchmark — interactive workbench (Gradio).

A dual-audience workbench for the ICPR 2026 paper *Benchmarking Transformers on
Spatio-Temporal River Water Temperature Modeling*. It powers BOTH the Hugging
Face Space and the one-command **local** app (``srn app gradio`` /
``python -m swissrivernetwork.app.workbench``) from a single source file, so the
two stay in lock-step. Capabilities: explore the released CSVs (map, seasonality,
3D network, drought.ch-style outlook), compare models (radar, window-length,
horizon, noise, per-station, ranking), a CPU train/eval sandbox,
**bring-your-own-model inference with live streaming**, residual / seasonal /
threshold **analysis**, and multi-format upload. Read-only over artefacts.

Compute: scikit-learn models run on CPU; TorchScript models run on a CUDA GPU
when one is detected (local only), else CPU. PyTorch is optional and imported
lazily so the Space stays light. Data resolution order: ``$SRN_WORKBENCH_DATA``
→ co-located ``data`` → repo ``tmp/hf-space/data``.
Source: https://github.com/jajupmochi/swiss-river-network-benchmark
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import folium
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from branca.element import Figure
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:  # PyTorch is optional (absent on the light HF Space); enables GPU inference.
    import torch

    TORCH_OK = True
except Exception:  # pragma: no cover - torch simply unavailable
    torch = None
    TORCH_OK = False


def _resolve_data() -> Path:
    """Locate a workbench data bundle (see module docstring for the order)."""
    here = Path(__file__).resolve().parent
    cands = []
    env = os.environ.get("SRN_WORKBENCH_DATA")
    if env:
        cands.append(Path(env))
    cands.append(here / "data")
    cands.append(here.parents[1] / "tmp" / "hf-space" / "data")
    for c in cands:
        if c and (c / "stations_zurich.csv").exists():
            return c
    return here / "data"


APP = Path(__file__).resolve().parent
DATA = _resolve_data()
FIG = APP / "figures"

# Most-recently trained sandbox model, reused by the Inference / Analysis tabs.
_STATE: dict = {"model": None, "kind": "sklearn", "label": None, "window": 14}
INFER_SOURCES = ["Last trained sandbox model", "Uploaded model file"]
MODEL_EXT = [".joblib", ".pkl", ".pickle", ".pt", ".ts"]
STREAM_STEPS = 24
# On a hosted multi-tenant Space (e.g. Hugging Face) loading a user model would execute code
# on a shared server (pickle / TorchScript), so uploads are disabled there; local is safe.
_HOSTED = bool(os.environ.get("SPACE_ID") or os.environ.get("SYSTEM") == "spaces")

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
PES = ("rope", "sinusoidal", "learnable")
METRICS = ["RMSE", "MAE", "NSE"]
STATS = ["Mean", "Median", "Std", "Min", "Max"]
WL_METRICS = [f"{m}_{s}" for m in METRICS for s in ("Mean", "Median")]
NOISE_TYPES = ["gaussian_a", "impulse_a"]
MAX_UPLOAD_MB = 50


# ------------------------------------------------------------------ loaders / metrics
def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def stations_df(graph):
    return _csv(DATA / f"stations_{graph}.csv")


def edges_df(graph):
    return _csv(DATA / f"edges_{graph}.csv")


def win_lens(graph, method):
    return _csv(DATA / "win_lens" / f"{graph}_{method}_win_lens_resu.csv")


def future_steps(graph, method):
    return _csv(DATA / "future_steps" / f"{graph}_{method}_fs_resu.csv")


def noise_df(graph, method, nt):
    return _csv(DATA / "noises" / f"{graph}_{method}_noises_resu_{nt}.csv")


def series_df(graph, split):
    return _csv(DATA / "series" / f"{graph}_{split}.csv")


def station_resu(graph, method):
    cands = [
        DATA / "stations" / f"{graph}_{method}_station_resu.csv",
        DATA / "stations" / f"{graph}_{method}_station_resu_f.csv",
    ]
    for pe in PES:
        cands += [
            DATA / "stations" / f"{graph}_{method}_{pe}_station_resu.csv",
            DATA / "stations" / f"{graph}_{method}_{pe}_station_resu_f.csv",
        ]
    for c in cands:
        if c.exists():
            return pd.read_csv(c)
    return pd.DataFrame()


def best_pe(graph, method, loader, col):
    df = loader(graph, method)
    if df.empty or "positional_encoding" not in df.columns:
        return df
    larger = col.startswith("NSE")
    best, score = None, None
    for pe, sub in df.groupby("positional_encoding"):
        s = sub[col].mean()
        if score is None or (s > score if larger else s < score):
            score, best = s, pe
    return df[df["positional_encoding"] == best]


def stations_of(graph):
    s = stations_df(graph)
    return [str(int(x)) for x in s.station] if not s.empty else []


def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def nse(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2))


def _empty(msg):
    f = go.Figure()
    f.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font={"size": 14})
    f.update_layout(template="plotly_white", height=360)
    return f


# ------------------------------------------------------------------ map
def build_map(graph, color_method, metric):
    s = stations_df(graph)
    if s.empty:
        return "<p>No station coordinates available.</p>"
    e = edges_df(graph)
    fig = Figure(width="100%", height=560)
    m = folium.Map(
        location=[float(s.lat.mean()), float(s.lon.mean())], zoom_start=8, tiles="CartoDB positron", control_scale=True
    )
    fig.add_child(m)
    for _, r in e.iterrows():
        if pd.notna(r.get("src_lat")) and pd.notna(r.get("dst_lat")):
            folium.PolyLine(
                [[r.src_lat, r.src_lon], [r.dst_lat, r.dst_lon]], color="#3186cc", weight=2, opacity=0.7
            ).add_to(m)
    vals, popup_lab = None, metric
    if color_method in ("water-temp (max)", "water-temp (mean)"):
        vals = station_temps(graph, "max" if "max" in color_method else "mean")
        popup_lab = "max °C" if "max" in color_method else "mean °C"
    elif color_method != "(none)":
        rs = station_resu(graph, color_method)
        if not rs.empty and metric in rs.columns:
            rs = rs[~rs["Station"].astype(str).isin(STATS + ["CI95"])]
            vals = {str(k): float(v) for k, v in zip(rs["Station"], rs[metric])}
    vmin, vmax = (min(vals.values()), max(vals.values())) if vals else (0, 1)
    for _, r in s.iterrows():
        sid = str(int(r.station))
        if vals and sid in vals:
            v = vals[sid]
            t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
            color = f"#{int(255 * t):02x}{int(90 + 80 * (1 - t)):02x}{int(255 * (1 - t)):02x}"
            popup = f"Station {sid}<br>{popup_lab}={v:.3f}"
        else:
            color, popup = "#2c7fb8", f"Station {sid}"
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=6,
            color="#333",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=200),
        ).add_to(m)
    return fig._repr_html_()


def overview_md(graph):
    s, e = stations_df(graph), edges_df(graph)
    meta = {
        "swiss-1990": "1990-2012 train / 2013-2020 test, no missing ground truth",
        "swiss-2010": "2010-2017 / 2018-2020, with missing ground truth",
        "zurich": "2009-2019 / 2020-2022, with missing ground truth",
    }
    return (
        f"### {graph}\n**{len(s)} monitoring stations · {len(e)} river-graph edges** — {meta.get(graph, '')}.\n\n"
        "Coordinates converted from CH1903/LV03 to WGS84. Marker colour = selected per-station error "
        "metric (blue = low error, red = high). Lines = directed river-network edges."
    )


# ------------------------------------------------------------------ raw series
def plot_series(graph, station, split):
    df = series_df(graph, split)
    wt, at = f"{station}_wt", f"{station}_at"
    if df.empty or wt not in df.columns:
        return _empty(f"No raw series for {graph} / station {station}.")
    d = df[["epoch_day", wt, at]].copy()
    d["date"] = pd.to_datetime(d["epoch_day"], unit="D")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d[wt], name="water temp", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=d["date"], y=d[at], name="air temp", line=dict(color="#ff7f0e", width=1), opacity=0.6))
    fig.update_layout(
        title=f"{graph} · station {station} ({split})",
        height=420,
        template="plotly_white",
        xaxis_title="date",
        yaxis_title="temperature (°C)",
    )
    return fig


def plot_seasonal(graph, station, split):
    df = series_df(graph, split)
    wt = f"{station}_wt"
    if df.empty or wt not in df.columns:
        return _empty("No data.")
    d = df[["epoch_day", wt]].dropna().copy()
    d["doy"] = pd.to_datetime(d["epoch_day"], unit="D").dt.dayofyear
    g = d.groupby("doy")[wt].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["doy"], y=g["mean"], name="mean", line=dict(color="#1f77b4")))
    fig.add_trace(
        go.Scatter(
            x=list(g["doy"]) + list(g["doy"][::-1]),
            y=list(g["mean"] + g["std"]) + list((g["mean"] - g["std"])[::-1]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.2)",
            line=dict(width=0),
            name="±std",
        )
    )
    fig.update_layout(
        title=f"{graph} · station {station}: seasonal cycle",
        height=420,
        template="plotly_white",
        xaxis_title="day of year",
        yaxis_title="water temp (°C)",
    )
    return fig


def plot_coverage(graph, split):
    df = series_df(graph, split)
    if df.empty:
        return _empty("No series.")
    wt_cols = [c for c in df.columns if c.endswith("_wt")]
    d = df[["epoch_day"] + wt_cols].copy()
    d["date"] = pd.to_datetime(d["epoch_day"], unit="D")
    mat = (~d[wt_cols].isna()).astype(int).T
    fig = px.imshow(
        mat,
        aspect="auto",
        color_continuous_scale=["#eee", "#1f77b4"],
        labels=dict(x="time index", y="station", color="has value"),
        title=f"{graph} ({split}) — water-temperature data coverage",
    )
    fig.update_yaxes(tickvals=list(range(len(wt_cols))), ticktext=[c[:-3] for c in wt_cols])
    fig.update_layout(height=460, coloraxis_showscale=False)
    return fig


# ------------------------------------------------------------------ comparisons
def plot_win_lens(graph, methods, metric):
    fr = [
        win_lens(graph, m).assign(method=m)[["window_len", metric, "method"]]
        for m in methods
        if not win_lens(graph, m).empty and metric in win_lens(graph, m).columns
    ]
    if not fr:
        return _empty("No window-length CSVs for the selection.")
    fig = px.line(
        pd.concat(fr),
        x="window_len",
        y=metric,
        color="method",
        markers=True,
        title=f"{graph} — {metric} vs. history-window length",
    )
    fig.update_layout(height=460, template="plotly_white", xaxis_title="window length (days)")
    return fig


def plot_per_station(graph, method, metric):
    df = station_resu(graph, method)
    if df.empty:
        return _empty(f"No per-station CSV for {graph}/{method}.")
    df = df[~df["Station"].astype(str).isin(STATS + ["CI95"])]
    base = metric.split("_")[0]
    if base not in df.columns:
        return _empty(f"Metric {base} absent.")
    d = df[["Station", base]].dropna().sort_values(base)
    fig = px.bar(d, x="Station", y=base, title=f"{graph} / {method} — {base} per station")
    fig.update_layout(height=440, template="plotly_white", xaxis_tickangle=-45)
    return fig


def plot_forecast(graph, methods, metric):
    fr = []
    for m in methods:
        df = best_pe(graph, m, future_steps, metric)
        if not df.empty and metric in df.columns and "future_step" in df.columns:
            fr.append(df[["future_step", metric]].assign(method=m).sort_values("future_step"))
    if not fr:
        return _empty("No forecasting CSVs for the selection.")
    fig = px.line(
        pd.concat(fr),
        x="future_step",
        y=metric,
        color="method",
        markers=True,
        title=f"{graph} — {metric} vs. forecasting horizon",
    )
    fig.update_layout(height=460, template="plotly_white", xaxis_title="horizon (days ahead)")
    return fig


def plot_noise(graph, methods, nt, metric):
    fr = [
        noise_df(graph, m, nt).assign(method=m)[["noise_level", metric, "method"]].sort_values("noise_level")
        for m in methods
        if not noise_df(graph, m, nt).empty and metric in noise_df(graph, m, nt).columns
    ]
    if not fr:
        return _empty("No noise CSVs for the selection.")
    lab = "Gaussian σ (fraction of std)" if nt == "gaussian_a" else "impulse probability"
    fig = px.line(
        pd.concat(fr),
        x="noise_level",
        y=metric,
        color="method",
        markers=True,
        title=f"{graph} — {metric} vs. {nt} noise",
    )
    fig.update_layout(height=460, template="plotly_white", xaxis_title=lab)
    return fig


# ------------------------------------------------------------------ radar (code-generated)
RADAR_AXES = [
    "RMSE/now",
    "MAE/now",
    "NSE/now",
    "RMSE/fore",
    "MAE/fore",
    "NSE/fore",
    "Gaussian-noise",
    "Impulse-noise",
    "Window-eff.",
]


def _model_axes(graph, method):
    """Raw (un-normalised) value per axis; lower-is-better for errors, higher for NSE."""
    out = {}
    rs = station_resu(graph, method)
    rs = rs[rs["Station"].astype(str) == "Mean"] if not rs.empty else rs
    for k, col in [("RMSE/now", "RMSE"), ("MAE/now", "MAE"), ("NSE/now", "NSE")]:
        out[k] = float(rs[col].iloc[0]) if (not rs.empty and col in rs.columns) else np.nan
    for k, col in [("RMSE/fore", "RMSE_Mean"), ("MAE/fore", "MAE_Mean"), ("NSE/fore", "NSE_Mean")]:
        df = best_pe(graph, method, future_steps, col)
        out[k] = float(df[col].mean()) if (not df.empty and col in df.columns) else np.nan
    for k, nt in [("Gaussian-noise", "gaussian_a"), ("Impulse-noise", "impulse_a")]:
        df = best_pe(graph, method, lambda g, m: noise_df(g, m, nt), "RMSE_Mean")
        out[k] = float(df["RMSE_Mean"].mean()) if (not df.empty and "RMSE_Mean" in df.columns) else np.nan
    df = best_pe(graph, method, win_lens, "RMSE_Mean")
    out["Window-eff."] = float(df["RMSE_Mean"].mean()) if (not df.empty and "RMSE_Mean" in df.columns) else np.nan
    return out


def plot_radar(graph, methods, extra=None):
    raw = {m: _model_axes(graph, m) for m in methods}
    if extra:
        raw["uploaded"] = extra
    # normalise each axis to [0,1], larger = better
    norm = {m: {} for m in raw}
    for ax in RADAR_AXES:
        vals = {m: raw[m].get(ax) for m in raw if raw[m].get(ax) is not None and not np.isnan(raw[m].get(ax, np.nan))}
        if not vals:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        higher_better = ax.startswith("NSE")
        for m, v in vals.items():
            t = 0.5 if hi == lo else (v - lo) / (hi - lo)
            norm[m][ax] = t if higher_better else 1 - t
    fig = go.Figure()
    for m, d in norm.items():
        ax = [a for a in RADAR_AXES if a in d]
        if not ax:
            continue
        fig.add_trace(
            go.Scatterpolar(r=[d[a] for a in ax] + [d[ax[0]]], theta=ax + [ax[0]], fill="toself", name=m, opacity=0.55)
        )
    fig.update_layout(
        title=f"{graph} — multi-aspect comparison (outer = better)",
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
        height=560,
        template="plotly_white",
    )
    return fig


# ------------------------------------------------------------------ sinusoidal positional encoding
def radar_upload(graph, methods, file):
    """Overlay a radar trace computed from an uploaded per-station metrics file."""
    if file is None:
        return plot_radar(graph, methods)
    try:
        df = read_any(file.name)
    except Exception:
        return plot_radar(graph, methods)
    cols = {str(c).lower(): c for c in df.columns}
    if "station" not in cols or not any(m.lower() in cols for m in METRICS):
        return plot_radar(graph, methods)
    d = df[~df[cols["station"]].astype(str).isin(STATS + ["CI95"])]
    extra = {}
    for m, ax in [("RMSE", "RMSE/now"), ("MAE", "MAE/now"), ("NSE", "NSE/now")]:
        if m.lower() in cols:
            try:
                extra[ax] = float(pd.to_numeric(d[cols[m.lower()]], errors="coerce").mean())
            except Exception:
                pass
    return plot_radar(graph, methods, extra=extra or None)


def plot_sinusoidal(d_model, max_len):
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    fig = px.imshow(
        pe.T,
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        labels=dict(x="sequence position", y="embedding dimension", color="value"),
        title=f"Sinusoidal positional encoding (d_model={d_model}, len={max_len})",
    )
    fig.update_layout(height=460)
    return fig


# ------------------------------------------------------------------ results table
def station_temps(graph, agg="max"):
    """Per-station water-temperature aggregate (°C) from the bundled series."""
    out = {}
    for split in ("train", "test"):
        df = series_df(graph, split)
        if df.empty:
            continue
        for c in df.columns:
            if c.endswith("_wt"):
                v = df[c].dropna()
                if len(v):
                    cur = float(v.max() if agg == "max" else v.mean())
                    sid = c[:-3]
                    out[sid] = max(out.get(sid, cur), cur) if agg == "max" else (out.get(sid, cur) + cur) / 2
    return out


THRESHOLDS = ["(none)", "18 °C", "21 °C (grayling stress)", "25 °C (fish-stress / regulatory)"]


def forecast_outlook(graph, station, threshold, horizon):
    """drought.ch-style probabilistic outlook: day-of-year climatology band + thresholds."""
    tr, te = series_df(graph, "train"), series_df(graph, "test")
    wt = f"{station}_wt"
    if tr.empty or wt not in tr.columns:
        return _empty(f"No series for {graph} / station {station}.")
    hist = pd.concat([tr[["epoch_day", wt]], te[["epoch_day", wt]]], ignore_index=True).dropna()
    if hist.empty:
        return _empty("No observations for this station.")
    hist["date"] = pd.to_datetime(hist["epoch_day"], unit="D")
    hist["doy"] = hist["date"].dt.dayofyear
    clim = hist.groupby("doy")[wt].agg(p10=lambda s: s.quantile(0.1), p50="median", p90=lambda s: s.quantile(0.9))
    last = hist["date"].max()
    fut = pd.date_range(last + pd.Timedelta(days=1), periods=int(horizon))
    fdoy = fut.dayofyear

    def band(p):
        return [float(clim[p].get(d, clim[p].mean())) for d in fdoy]

    recent = hist[hist["date"] >= last - pd.Timedelta(days=90)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["date"], y=recent[wt], name="observed", line=dict(color="#1f77b4")))
    fig.add_trace(
        go.Scatter(
            x=list(fut) + list(fut[::-1]),
            y=band("p90") + band("p10")[::-1],
            fill="toself",
            fillcolor="rgba(214,39,40,0.15)",
            line=dict(width=0),
            name="p10–p90 outlook",
        )
    )
    fig.add_trace(go.Scatter(x=fut, y=band("p50"), name="median outlook", line=dict(color="#d62728", dash="dash")))
    fig.add_vline(x=last, line=dict(color="#888", dash="dot"))
    if threshold and threshold != "(none)":
        thv = int(str(threshold).split()[0])
        fig.add_hline(
            y=thv, line=dict(color="#e41a1c", width=1.2), annotation_text=f"{thv} °C", annotation_position="top left"
        )
    fig.update_layout(
        title=f"{graph} · station {station} — probabilistic water-temperature outlook "
        f"(next {int(horizon)} days, day-of-year climatology)",
        height=470,
        template="plotly_white",
        xaxis_title="date",
        yaxis_title="water temp (°C)",
    )
    return fig


def threejs_html(graph, color_method, metric):
    """3D river network rendered with three.js (inside an iframe, like the folium map)."""
    s = stations_df(graph)
    if s.empty:
        return "<p>No coordinates available.</p>"
    e = edges_df(graph)
    vals = station_temps(graph, "max")
    if color_method in METHODS:
        rs = station_resu(graph, color_method)
        if not rs.empty and metric in rs.columns:
            rs = rs[~rs["Station"].astype(str).isin(STATS + ["CI95"])]
            vals = {str(k): float(v) for k, v in zip(rs["Station"], rs[metric])}
    lon0, lon1, lat0, lat1 = s.lon.min(), s.lon.max(), s.lat.min(), s.lat.max()

    def nx(v):
        return (v - lon0) / (lon1 - lon0 + 1e-9) * 20 - 10

    def ny(v):
        return (v - lat0) / (lat1 - lat0 + 1e-9) * 16 - 8

    vv = [vals[str(int(r.station))] for _, r in s.iterrows() if str(int(r.station)) in vals]
    vmin, vmax = (min(vv), max(vv)) if vv else (0, 1)
    xy = {}
    nodes = []
    for _, r in s.iterrows():
        sid = str(int(r.station))
        v = vals.get(sid)
        t = 0.5 if (v is None or vmax == vmin) else (v - vmin) / (vmax - vmin)
        xy[int(r.idx)] = (nx(r.lon), ny(r.lat))
        nodes.append({"x": nx(r.lon), "y": ny(r.lat), "z": round(t * 6, 3), "t": round(t, 3), "id": sid})
    elist = [[xy[int(r.src)], xy[int(r.dst)]] for _, r in e.iterrows() if int(r.src) in xy and int(r.dst) in xy]
    data = json.dumps({"nodes": nodes, "edges": elist})
    inner = (
        "<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;height:100%}"
        "#c{width:100vw;height:100vh}</style></head><body><div id='c'></div>"
        "<script type='module'>"
        "import*as THREE from'https://unpkg.com/three@0.160.0/build/three.module.js';"
        "import{OrbitControls}from'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';"
        f"const D={data};const el=document.getElementById('c');const W=el.clientWidth,H=el.clientHeight;"
        "const sc=new THREE.Scene();sc.background=new THREE.Color(0x0b1021);"
        "const cam=new THREE.PerspectiveCamera(55,W/H,0.1,1000);cam.position.set(0,-22,16);"
        "const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(W,H);el.appendChild(rn.domElement);"
        "const ct=new OrbitControls(cam,rn.domElement);ct.enableDamping=true;"
        "sc.add(new THREE.AmbientLight(0xffffff,0.75));"
        "const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(5,-10,20);sc.add(dl);"
        "const g=new THREE.GridHelper(26,13,0x335577,0x223344);g.rotation.x=Math.PI/2;sc.add(g);"
        "const C=t=>new THREE.Color().setHSL((1-t)*0.66,0.85,0.55);"
        "D.edges.forEach(([a,b])=>{const gg=new THREE.BufferGeometry().setFromPoints("
        "[new THREE.Vector3(a[0],a[1],0),new THREE.Vector3(b[0],b[1],0)]);"
        "sc.add(new THREE.Line(gg,new THREE.LineBasicMaterial({color:0x3186cc})));});"
        "D.nodes.forEach(n=>{const m=new THREE.Mesh(new THREE.SphereGeometry(0.4,16,16),"
        "new THREE.MeshStandardMaterial({color:C(n.t)}));m.position.set(n.x,n.y,n.z);sc.add(m);"
        "const lg=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(n.x,n.y,0),"
        "new THREE.Vector3(n.x,n.y,n.z)]);sc.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:0x556688})));});"
        "function an(){requestAnimationFrame(an);ct.update();rn.render(sc,cam);}an();"
        "addEventListener('resize',()=>{const w=el.clientWidth,h=el.clientHeight;rn.setSize(w,h);"
        "cam.aspect=w/h;cam.updateProjectionMatrix();});"
        "</script></body></html>"
    )
    import html as _html

    return (
        f'<iframe title="3D river network" style="width:100%;height:580px;border:0;border-radius:8px" '
        f'srcdoc="{_html.escape(inner)}"></iframe>'
    )


def results_table(graph, scenario, statistic):
    p = DATA / "results" / "all_results.json"
    if not p.exists():
        return pd.DataFrame({"info": ["all_results.json not bundled"]})
    d = json.load(open(p))
    rows = []
    for arch, models in d.get("results", {}).get(graph, {}).items():
        for model, sc in models.items():
            node = sc.get(scenario, {})
            row = {"architecture": arch, "model": model}
            for metric in METRICS:
                v = node.get(metric, {}).get(statistic)
                row[metric] = round(v, 4) if isinstance(v, (int, float)) else None
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ train / eval sandbox (CPU)
def _windowed(at, wt, w):
    X, y = [], []
    for k in range(w, len(at)):
        if np.isnan(wt[k]) or np.isnan(at[k - w : k]).any():
            continue
        X.append(at[k - w : k])
        y.append(wt[k])
    return np.array(X), np.array(y)


def train_eval(graph, station, model_name, window):
    try:
        tr, te = series_df(graph, "train"), series_df(graph, "test")
        wt, at = f"{station}_wt", f"{station}_at"
        if tr.empty or wt not in tr.columns:
            return _empty("Pick a station with data."), "No data."
        Xtr, ytr = _windowed(tr[at].values.astype(float), tr[wt].values.astype(float), window)
        Xte, yte = _windowed(te[at].values.astype(float), te[wt].values.astype(float), window)
        if len(Xtr) < 30 or len(Xte) < 5:
            return _empty("Not enough clean samples for this station."), "Too few samples."
        model = (
            Ridge(alpha=1.0)
            if model_name == "Ridge regression"
            else MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=0)
        )
        model.fit(Xtr, ytr)
        _STATE.update(model=model, kind="sklearn", label=model_name, window=int(window))
        pred = model.predict(Xte)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=yte, name="ground truth", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(y=pred, name="prediction", line=dict(color="#d62728", dash="dot")))
        fig.update_layout(
            title=f"{graph} · station {station} — CPU baseline ({model_name}, window={window})",
            height=420,
            template="plotly_white",
            xaxis_title="test sample",
            yaxis_title="water temp (°C)",
        )
        msg = (
            f"**{model_name}** | window={window} d | train={len(Xtr)}, test={len(Xte)} samples → "
            f"**RMSE={rmse(yte, pred):.3f} · MAE={mae(yte, pred):.3f} · NSE={nse(yte, pred):.3f}**. "
            "_Lightweight CPU baseline for illustration — not the paper's GPU LSTM/Transformer models._"
        )
        return fig, msg
    except Exception as exc:
        return _empty("Training failed."), f"❌ {type(exc).__name__}: {exc}"


def eval_predictions(file, graph, station):
    if file is None:
        return _empty("Upload a predictions CSV."), "Upload a 2-column file: time + prediction."
    try:
        df = read_any(file.name)
    except Exception as exc:
        return _empty("Could not read file."), f"❌ {exc}"
    pred_col = next(
        (c for c in df.columns if "pred" in str(c).lower() or str(c).lower() in ("wt_hat", "yhat", "value")), None
    )
    if pred_col is None:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        pred_col = num[-1] if num else None
    if pred_col is None:
        return _empty("No numeric prediction column found."), "❌ Need a numeric prediction column."
    te = series_df(graph, "test")
    wt = f"{station}_wt"
    if te.empty or wt not in te.columns:
        return _empty("No bundled ground truth."), "❌ Ground truth unavailable."
    n = min(len(df), len(te))
    y = te[wt].values[:n].astype(float)
    yh = df[pred_col].values[:n].astype(float)
    mask = ~np.isnan(y) & ~np.isnan(yh)
    if mask.sum() < 5:
        return _empty("Too few aligned points."), "❌ Fewer than 5 aligned non-NaN points."
    y, yh = y[mask], yh[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, name="ground truth", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(y=yh, name=f"prediction ({pred_col})", line=dict(color="#d62728", dash="dot")))
    fig.update_layout(title=f"Evaluation vs {graph}/{station} ground truth", height=420, template="plotly_white")
    msg = f"✅ Aligned **{mask.sum()}** points → **RMSE={rmse(y, yh):.3f} · MAE={mae(y, yh):.3f} · NSE={nse(y, yh):.3f}**."
    return fig, msg


# ------------------------------------------------------------------ upload (robust + routing)
ACCEPTED = [".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"]


def read_any(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("File not found.")
    if p.stat().st_size > MAX_UPLOAD_MB * 1e6:
        raise ValueError(f"File too large (> {MAX_UPLOAD_MB} MB).")
    if p.stat().st_size == 0:
        raise ValueError("File is empty.")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        return pd.json_normalize(json.load(open(path)))
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported extension '{ext}'. Accepted: {', '.join(ACCEPTED)}.")


def handle_upload(file, graph):
    if file is None:
        return _empty("Upload a file to plot it."), "No file uploaded yet."
    try:
        df = read_any(file.name)
    except Exception as exc:
        return _empty("Could not read the file."), f"❌ **{type(exc).__name__}**: {exc}"
    try:
        if df is None or len(df) == 0:
            return _empty("Empty table."), "❌ The file parsed but contains no rows."
        df = df.dropna(axis=1, how="all")
        if df.shape[1] == 0:
            return _empty("No usable columns."), "❌ All columns are empty."
        cols = {str(c).lower(): c for c in df.columns}
        # route 1: benchmark result curves
        for key, xlab in [
            ("window_len", "window length (days)"),
            ("future_step", "horizon (days)"),
            ("noise_level", "noise level"),
        ]:
            if key in cols:
                xc = cols[key]
                ys = [c for c in df.columns if c != xc and pd.api.types.is_numeric_dtype(df[c])]
                if not ys:
                    break
                fig = go.Figure()
                for c in ys[:8]:
                    fig.add_trace(go.Scatter(x=df[xc], y=df[c], name=str(c), mode="lines+markers"))
                fig.update_layout(
                    title=f"Uploaded benchmark curve ({key})", height=440, template="plotly_white", xaxis_title=xlab
                )
                return (
                    fig,
                    f"✅ Detected a **benchmark-result curve** (`{key}` axis). Plotted {min(len(ys), 8)} series.",
                )
        # route 2: per-station metrics -> bar
        if "station" in cols and any(m.lower() in cols for m in METRICS):
            sc = cols["station"]
            mc = next(cols[m.lower()] for m in METRICS if m.lower() in cols)
            d = df[~df[sc].astype(str).isin(STATS + ["CI95"])][[sc, mc]].dropna().sort_values(mc)
            fig = px.bar(d, x=sc, y=mc, title=f"Uploaded per-station {mc}")
            fig.update_layout(height=440, template="plotly_white", xaxis_tickangle=-45)
            return fig, f"✅ Detected **per-station metrics**. Plotted `{mc}` for {len(d)} stations."
        # route 3: time series
        time_col = next((cols[k] for k in ("date", "datetime", "time", "timestamp") if k in cols), None)
        if time_col is None:
            time_col = cols["epoch_day"] if "epoch_day" in cols else df.columns[0]
        num = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            return _empty("No numeric value column found."), (
                "❌ Need at least one **numeric value column** (e.g. water/air temperature). "
                f"Columns seen: {', '.join(map(str, df.columns[:10]))}."
            )
        x = df[time_col]
        if str(time_col).lower() == "epoch_day":
            x = pd.to_datetime(df[time_col], unit="D", errors="coerce")
        else:
            xc = pd.to_datetime(df[time_col], errors="coerce")
            if xc.notna().mean() > 0.8:
                x = xc
        fig = go.Figure()
        for c in num[:12]:
            fig.add_trace(go.Scatter(x=x, y=df[c], name=str(c), mode="lines"))
        fig.update_layout(
            title=f"Uploaded series: {Path(file.name).name}",
            height=440,
            template="plotly_white",
            xaxis_title=str(time_col),
            yaxis_title="value",
        )
        return fig, (
            f"✅ Read **{len(df)} rows × {df.shape[1]} cols**. Time/index: `{time_col}`. "
            f"Plotted {min(len(num), 12)} numeric column(s): {', '.join(map(str, num[:12]))}."
        )
    except Exception:
        return _empty("Unexpected error while plotting."), f"❌ {traceback.format_exc().splitlines()[-1]}"


UPLOAD_REQ = """
**Accepted formats:** `.csv` · `.tsv` · `.xlsx` / `.xls` · `.json` · `.parquet` (≤ 50 MB, parsed in
memory, never stored).

**The app auto-detects the content and routes it:**
1. **Time series** — needs a time/index column (`date`/`datetime`/`time`, or an integer `epoch_day`
   = days since 1970-01-01; otherwise the first column is used) **and at least one numeric value
   column** (e.g. water/air temperature in °C). Up to 12 numeric columns are plotted.
2. **Benchmark result curve** — if a `window_len`, `future_step`, or `noise_level` column is present,
   it becomes the x-axis automatically.
3. **Per-station metrics** — a `Station` column plus `RMSE`/`MAE`/`NSE` → a per-station bar chart.

Malformed, empty, oversized, or non-numeric files return a clear error message rather than crashing.
"""

RESOURCES_MD = """
## Compute & resources

- The **same app** runs on the **Hugging Face Space** (free CPU) and **locally** via `srn app gradio` /
  `python -m swissrivernetwork.app.workbench` — one code base, identical features.
- **Explore / compare** views are read-only over the released CSVs (no GPU needed).
- **Train / Eval sandbox** fits a lightweight `scikit-learn` baseline (Ridge or a small MLP) on a window
  of past air temperature — CPU, live, illustrative (not the paper's GPU models).
- **Inference** runs your uploaded model: scikit-learn (`.joblib`/`.pkl`) on CPU, or **TorchScript**
  (`.pt`) on a **GPU auto-detected locally** (else CPU). The detected environment is shown at the top of
  the *Inference* tab.
- The **paper's** 13,200 models were trained on an **RTX 4090 (24 GB)** and evaluated on an **RTX 3070
  (8 GB)**. Reproduce with `srn prepare-data → tune → evaluate → sweep` (see the docs).

## Data provenance
Station coordinates and the river-network edge list come from the benchmark's graph files (CH1903/LV03,
converted to WGS84). Per-station raw series and all result CSVs are the released benchmark outputs.

## Links
[GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark) ·
[Documentation](https://jajupmochi.github.io/swiss-river-network-benchmark/) ·
ICPR 2026 paper #396.
"""


# ------------------------------------------------------------------ compute resources / device
def cuda_available() -> bool:
    return bool(TORCH_OK and torch.cuda.is_available())


def device_choices() -> list[str]:
    return ["auto", "cpu"] + (["cuda"] if cuda_available() else [])


def _resolve_device(sel: str) -> str:
    if not TORCH_OK:
        return "cpu"
    if sel == "cuda" or (sel in ("auto", None) and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def detect_resources() -> str:
    import platform

    lines = [f"**Python** {platform.python_version()}  ·  **CPU cores** {os.cpu_count()}"]
    try:
        with open("/proc/meminfo") as fh:
            kb = int(next(ln for ln in fh if ln.startswith("MemTotal")).split()[1])
        lines.append(f"**RAM** {kb / 1e6:.1f} GB")
    except Exception:
        pass
    if TORCH_OK and torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        lines.append(
            f"**GPU** {p.name} ({p.total_memory / 1e9:.0f} GB, CUDA {torch.version.cuda}) — "
            "TorchScript inference will run here"
        )
    elif TORCH_OK:
        lines.append("**GPU** none detected (PyTorch present, CUDA unavailable) — running on CPU")
    else:
        lines.append(
            "**GPU** PyTorch not installed here — scikit-learn models run on CPU. Install "
            "`swissrivernetwork[app]` **and** PyTorch locally to load TorchScript models on a GPU"
        )
    return "### Detected compute resources\n" + "  \n".join(lines)


# ------------------------------------------------------------------ bring-your-own-model inference
def load_user_model(file):
    """Load an uploaded model: scikit-learn (.joblib/.pkl) or TorchScript (.pt/.ts)."""
    if file is None:
        return None, None, "Upload a **scikit-learn** estimator (`.joblib`/`.pkl`) or a **TorchScript** module (`.pt`)."
    if _HOSTED:
        return (
            None,
            None,
            (
                "⚠️ Model upload is disabled on the hosted Space: loading a model executes code, which is unsafe on a "
                "shared server. Run the workbench locally (`uv run srn app gradio`) to use your own model."
            ),
        )
    p = Path(file.name)
    ext = p.suffix.lower()
    if not p.exists() or p.stat().st_size == 0:
        return None, None, "❌ File missing or empty."
    if p.stat().st_size > MAX_UPLOAD_MB * 1e6:
        return None, None, f"❌ Model file too large (> {MAX_UPLOAD_MB} MB)."
    if ext in (".joblib", ".pkl", ".pickle"):
        try:
            model = joblib.load(p)
        except Exception as exc:
            return None, None, f"❌ Could not load model: {type(exc).__name__}: {exc}"
        if not hasattr(model, "predict"):
            return None, None, "❌ Loaded object exposes no `.predict(X)` method."
        return model, "sklearn", f"✅ Loaded scikit-learn **{type(model).__name__}** (CPU)."
    if ext in (".pt", ".ts"):
        if not TORCH_OK:
            return (
                None,
                None,
                "❌ TorchScript needs PyTorch (absent here). Upload a `.joblib` sklearn model, or run locally.",
            )
        try:
            model = torch.jit.load(str(p), map_location="cpu")
        except Exception as exc:
            return None, None, f"❌ Could not load TorchScript: {type(exc).__name__}: {exc}"
        return model, "torch", "✅ Loaded **TorchScript** module — pick a device below (GPU if available)."
    return None, None, f"❌ Unsupported '{ext}'. Use {', '.join(MODEL_EXT)}."


def model_status(file):
    return load_user_model(file)[2]


def _get_model(source, file):
    if source == "Uploaded model file":
        return load_user_model(file)
    m = _STATE.get("model")
    if m is None:
        return None, None, "No sandbox model yet — train one under **Train / Eval → Train a baseline** first."
    return (
        m,
        _STATE.get("kind", "sklearn"),
        f"✅ Using last sandbox model (**{_STATE.get('label')}**, window={_STATE.get('window')}).",
    )


def _predict(model, kind, X, device="auto"):
    if kind == "torch":
        dev = _resolve_device(device)
        if hasattr(model, "to"):
            model.to(dev)
        t = torch.as_tensor(np.asarray(X, dtype="float32"), device=dev)
        with torch.no_grad():
            out = model(t)
        return np.asarray(out.detach().cpu().numpy(), dtype=float).reshape(len(X), -1)[:, 0]
    return np.asarray(model.predict(X), dtype=float)


def _infer_frame(graph, station, split, window):
    df = series_df(graph, split)
    at, wt = f"{station}_at", f"{station}_wt"
    if df.empty or at not in df.columns or wt not in df.columns:
        return None
    w = int(window)
    a = df[at].to_numpy(dtype=float)
    y_all = df[wt].to_numpy(dtype=float)
    X, y, idx = [], [], []
    for k in range(w, len(a)):
        if np.isnan(y_all[k]) or np.isnan(a[k - w : k]).any():
            continue
        X.append(a[k - w : k])
        y.append(y_all[k])
        idx.append(k)
    if not X:
        return None
    dates = pd.to_datetime(df["epoch_day"].to_numpy()[np.array(idx)], unit="D")
    return np.array(X), np.array(y), dates


def _series_pred(graph, station, split, source, file, window, device):
    model, kind, msg = _get_model(source, file)
    if model is None:
        return None, msg
    fr = _infer_frame(graph, station, split, window)
    if fr is None:
        return None, "❌ Not enough clean samples for this station/split/window."
    X, y, dates = fr
    try:
        yh = _predict(model, kind, X, device)
    except Exception as exc:
        return None, (
            f"❌ {type(exc).__name__}: {exc}. For a feature-size error, set **window** to the number of "
            "input features the model expects."
        )
    return (dates, y, yh), msg


def run_inference(graph, station, split, source, file, window, device):
    out, msg = _series_pred(graph, station, split, source, file, window, device)
    if out is None:
        return _empty(msg), msg
    dates, y, yh = out
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y, name="ground truth", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=dates, y=yh, name="prediction", line=dict(color="#d62728", dash="dot")))
    fig.update_layout(
        title=f"{graph} · station {station} — inference ({split})",
        height=430,
        template="plotly_white",
        xaxis_title="date",
        yaxis_title="water temp (°C)",
    )
    return (
        fig,
        f"{msg}  →  **RMSE={rmse(y, yh):.3f} · MAE={mae(y, yh):.3f} · NSE={nse(y, yh):.3f}** over {len(y)} steps.",
    )


def stream_inference(graph, station, split, source, file, window, device, delay=0.05):
    """Generator: yield the prediction progressively for a real-time output feel."""
    out, msg = _series_pred(graph, station, split, source, file, window, device)
    if out is None:
        yield _empty(msg), msg
        return
    dates, y, yh = out
    n = len(y)
    step = max(1, n // STREAM_STEPS)
    fig = _empty("starting…")
    for end in range(step, n + step, step):
        end = min(end, n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates[:end], y=y[:end], name="ground truth", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=dates[:end], y=yh[:end], name="prediction", line=dict(color="#d62728", dash="dot")))
        fig.update_layout(
            title=f"{graph} · station {station} — live inference {end}/{n}",
            height=430,
            template="plotly_white",
            xaxis_title="date",
            yaxis_title="water temp (°C)",
        )
        if delay:
            time.sleep(delay)
        yield fig, f"⏱ streaming {end}/{n} — running RMSE={rmse(y[:end], yh[:end]):.3f}"
        if end >= n:
            break
    yield fig, f"✅ Done ({n} steps) — **RMSE={rmse(y, yh):.3f} · MAE={mae(y, yh):.3f} · NSE={nse(y, yh):.3f}**."


# ------------------------------------------------------------------ analysis
def residual_analysis(graph, station, split, source, file, window, device):
    out, msg = _series_pred(graph, station, split, source, file, window, device)
    if out is None:
        return _empty(msg), msg
    dates, y, yh = out
    res = yh - y
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("residual (pred − obs) over time", "residual distribution"),
    )
    fig.add_trace(go.Scatter(x=dates, y=res, name="residual", line=dict(color="#8856a7")), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#888", dash="dot"), row=1, col=1)
    fig.add_trace(go.Histogram(x=res, nbinsx=40, marker_color="#8856a7"), row=1, col=2)
    fig.update_layout(
        height=430,
        template="plotly_white",
        showlegend=False,
        title=f"{graph} · station {station} — residual analysis",
    )
    return fig, (
        f"{msg}  →  bias(mean residual)=**{np.mean(res):+.3f} °C**, std=**{np.std(res):.3f} °C**, "
        f"|residual|>2 °C on **{100 * np.mean(np.abs(res) > 2):.1f}%** of days."
    )


def seasonal_error(graph, station, split, source, file, window, device):
    out, msg = _series_pred(graph, station, split, source, file, window, device)
    if out is None:
        return _empty(msg), msg
    dates, y, yh = out
    d = pd.DataFrame({"month": pd.DatetimeIndex(dates).month, "ae": np.abs(yh - y)})
    g = d.groupby("month")["ae"].mean().reindex(range(1, 13)).reset_index()
    fig = px.bar(g, x="month", y="ae", title=f"{graph} · station {station} — mean abs error by month")
    fig.update_layout(
        height=420,
        template="plotly_white",
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13))),
        yaxis_title="MAE (°C)",
    )
    wm = int(g.loc[g["ae"].idxmax(), "month"]) if g["ae"].notna().any() else 0
    return fig, f"{msg}  →  largest seasonal error in **month {wm}** (summer months matter most for fish stress)."


def threshold_exceedance(graph, station, threshold):
    tr, te = series_df(graph, "train"), series_df(graph, "test")
    wt = f"{station}_wt"
    if tr.empty or wt not in tr.columns:
        return _empty("No series for this station."), "❌ No data."
    hist = pd.concat([tr[["epoch_day", wt]], te[["epoch_day", wt]]], ignore_index=True).dropna()
    if hist.empty:
        return _empty("No observations."), "❌ No data."
    hist["year"] = pd.to_datetime(hist["epoch_day"], unit="D").dt.year
    thv = int(str(threshold).split()[0]) if threshold and threshold != "(none)" else 25
    g = hist.assign(ex=(hist[wt] > thv).astype(int)).groupby("year")["ex"].sum().reset_index()
    fig = px.bar(g, x="year", y="ex", title=f"{graph} · station {station} — days per year above {thv} °C")
    fig.update_layout(height=420, template="plotly_white", yaxis_title=f"days > {thv} °C")
    total = int(g["ex"].sum())
    worst = g.loc[g["ex"].idxmax()] if len(g) else None
    msg = f"Observed **{total}** days above **{thv} °C** across {len(g)} years"
    if worst is not None:
        msg += f"; worst year **{int(worst['year'])}** with **{int(worst['ex'])}** days."
    return fig, msg


def model_ranking(graph, metric):
    rows = []
    for m in METHODS:
        rs = station_resu(graph, m)
        if rs.empty or metric not in rs.columns:
            continue
        d = rs[~rs["Station"].astype(str).isin(STATS + ["CI95"])]
        vals = pd.to_numeric(d[metric], errors="coerce").dropna()
        if len(vals):
            rows.append({"model": m, metric: float(vals.mean())})
    if not rows:
        return _empty("No per-station result CSVs for this dataset."), "❌ No results."
    d = pd.DataFrame(rows).sort_values(metric, ascending=not metric.startswith("NSE"))
    fig = px.bar(d, x="model", y=metric, title=f"{graph} — architectures ranked by mean {metric}")
    fig.update_layout(height=440, template="plotly_white", xaxis_tickangle=-30)
    return fig, f"Best by mean **{metric}**: **{d.iloc[0]['model']}**."


# ------------------------------------------------------------------ UI (workbench)
def ui():
    with gr.Blocks(
        title="Swiss River Network Benchmark",
        theme=gr.themes.Soft(),
        css=".gradio-container{max-width:none!important;margin:0 auto!important} "
        ".main{width:100%} footer{display:none!important}",
        fill_width=True,
    ) as demo:
        gr.Markdown(
            "# 🌊 Swiss River Network Benchmark — Workbench\n"
            "Interactive companion to the ICPR 2026 paper *Benchmarking Transformers on Spatio-Temporal "
            "River Water Temperature Modeling*. **Explore** the data and results, **train/evaluate** a "
            "lightweight baseline, and **bring your own data**. "
            "[GitHub](https://github.com/jajupmochi/swiss-river-network-benchmark) · "
            "[Docs](https://jajupmochi.github.io/swiss-river-network-benchmark/)"
        )

        with gr.Tabs():
            # ============ DATA ============
            with gr.Tab("Data"):
                with gr.Tab("Map & overview"):
                    with gr.Row():
                        g = gr.Dropdown(GRAPHS, value="swiss-2010", label="Dataset")
                        cm = gr.Dropdown(
                            ["(none)", "water-temp (max)", "water-temp (mean)"] + METHODS,
                            value="water-temp (max)",
                            label="Colour by",
                        )
                        mt = gr.Dropdown(METRICS, value="RMSE", label="Colour by metric")
                    info = gr.Markdown(overview_md("swiss-2010"))
                    mp = gr.HTML(build_map("swiss-2010", "water-temp (max)", "RMSE"))
                    for c in (g, cm, mt):
                        c.change(build_map, [g, cm, mt], mp)
                    g.change(overview_md, g, info)
                with gr.Tab("Raw series & seasonality"):
                    with gr.Row():
                        gs = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                        st = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                        sp = gr.Dropdown(["train", "test"], value="train", label="Split")
                    ts = gr.Plot(plot_series("zurich", stations_of("zurich")[0], "train"))
                    sea = gr.Plot(plot_seasonal("zurich", stations_of("zurich")[0], "train"))
                    gs.change(lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), gs, st)
                    for c in (gs, st, sp):
                        c.change(plot_series, [gs, st, sp], ts)
                        c.change(plot_seasonal, [gs, st, sp], sea)
                with gr.Tab("Data coverage"):
                    with gr.Row():
                        gc = gr.Dropdown(GRAPHS, value="swiss-2010", label="Dataset")
                        spc = gr.Dropdown(["train", "test"], value="train", label="Split")
                    cov = gr.Plot(plot_coverage("swiss-2010", "train"))
                    for c in (gc, spc):
                        c.change(plot_coverage, [gc, spc], cov)
                with gr.Tab("Forecast outlook"):
                    gr.Markdown(
                        "Probabilistic water-temperature **outlook for the coming weeks**, in the style of the Swiss "
                        "drought portal (drought.ch): a day-of-year climatology band (p10–p90) with the median and "
                        "ecological threshold lines. Illustrative (climatology-based), not an operational forecast."
                    )
                    with gr.Row():
                        gf = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                        stf = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                        thr = gr.Dropdown(THRESHOLDS, value="25 °C (fish-stress / regulatory)", label="Threshold")
                        hor = gr.Slider(7, 56, value=28, step=7, label="horizon (days)")
                    fc = gr.Plot(
                        forecast_outlook("zurich", stations_of("zurich")[0], "25 °C (fish-stress / regulatory)", 28)
                    )
                    gf.change(lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), gf, stf)
                    for c in (gf, stf, thr, hor):
                        c.change(forecast_outlook, [gf, stf, thr, hor], fc)
                with gr.Tab("3D river network"):
                    gr.Markdown(
                        "Interactive **three.js** 3D view: stations placed by longitude/latitude, raised and coloured "
                        "by the selected value; river-network edges on the base plane. Drag to rotate, scroll to zoom."
                    )
                    with gr.Row():
                        g3 = gr.Dropdown(GRAPHS, value="swiss-2010", label="Dataset")
                        cm3 = gr.Dropdown(
                            ["water-temp (max)"] + METHODS, value="water-temp (max)", label="Height / colour by"
                        )
                        mt3 = gr.Dropdown(METRICS, value="RMSE", label="Metric (if a model)")
                    tj = gr.HTML(threejs_html("swiss-2010", "water-temp (max)", "RMSE"))
                    for c in (g3, cm3, mt3):
                        c.change(threejs_html, [g3, cm3, mt3], tj)

            # ============ MODELS & RESULTS ============
            with gr.Tab("Models & results"):
                with gr.Tab("Multi-aspect radar"):
                    with gr.Row():
                        gr_ = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        cbr = gr.CheckboxGroup(
                            METHODS,
                            value=["transformer_embedding", "lstm_embedding", "transformer_stgnn", "stgnn"],
                            label="Models",
                        )
                    pr = gr.Plot(
                        plot_radar(
                            "swiss-1990", ["transformer_embedding", "lstm_embedding", "transformer_stgnn", "stgnn"]
                        )
                    )
                    for c in (gr_, cbr):
                        c.change(plot_radar, [gr_, cbr], pr)
                    rfile = gr.File(
                        label="Optional: overlay your own per-station metrics (needs a Station column + RMSE/MAE/NSE)",
                        file_types=ACCEPTED,
                    )
                    rfile.change(radar_upload, [gr_, cbr, rfile], pr)
                with gr.Tab("Window-length sweep"):
                    with gr.Row():
                        g2 = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        m2 = gr.Dropdown(WL_METRICS, value="RMSE_Mean", label="Metric")
                    cb = gr.CheckboxGroup(METHODS, value=METHODS[:4], label="Models")
                    p2 = gr.Plot(plot_win_lens("swiss-1990", METHODS[:4], "RMSE_Mean"))
                    for c in (g2, m2, cb):
                        c.change(plot_win_lens, [g2, cb, m2], p2)
                with gr.Tab("Forecasting horizons"):
                    with gr.Row():
                        g4 = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        m4 = gr.Dropdown(WL_METRICS, value="RMSE_Mean", label="Metric")
                    cb4 = gr.CheckboxGroup(METHODS, value=METHODS[:4], label="Models")
                    p4 = gr.Plot(plot_forecast("swiss-1990", METHODS[:4], "RMSE_Mean"))
                    for c in (g4, m4, cb4):
                        c.change(plot_forecast, [g4, cb4, m4], p4)
                with gr.Tab("Noise robustness"):
                    with gr.Row():
                        g5 = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        nt = gr.Dropdown(NOISE_TYPES, value="gaussian_a", label="Noise type")
                        m5 = gr.Dropdown(WL_METRICS, value="RMSE_Mean", label="Metric")
                    cb5 = gr.CheckboxGroup(METHODS, value=METHODS[:4], label="Models")
                    p5 = gr.Plot(plot_noise("swiss-1990", METHODS[:4], "gaussian_a", "RMSE_Mean"))
                    for c in (g5, nt, m5, cb5):
                        c.change(plot_noise, [g5, cb5, nt, m5], p5)
                with gr.Tab("Per-station error"):
                    with gr.Row():
                        g3 = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        m3 = gr.Dropdown(METHODS, value="transformer_embedding", label="Model")
                        me3 = gr.Dropdown(METRICS, value="RMSE", label="Metric")
                    p3 = gr.Plot(plot_per_station("swiss-1990", "transformer_embedding", "RMSE"))
                    for c in (g3, m3, me3):
                        c.change(plot_per_station, [g3, m3, me3], p3)
                with gr.Tab("Results table"):
                    with gr.Row():
                        g6 = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        sc6 = gr.Dropdown(["nowcasting", "forecasting"], value="nowcasting", label="Scenario")
                        stt6 = gr.Dropdown(STATS, value="Mean", label="Statistic")
                    tb = gr.Dataframe(results_table("swiss-1990", "nowcasting", "Mean"), label="RMSE / MAE / NSE")
                    for c in (g6, sc6, stt6):
                        c.change(results_table, [g6, sc6, stt6], tb)
                with gr.Tab("Positional encoding"):
                    with gr.Row():
                        dm = gr.Slider(8, 128, value=64, step=8, label="d_model")
                        ml = gr.Slider(20, 365, value=120, step=10, label="sequence length")
                    pe = gr.Plot(plot_sinusoidal(64, 120))
                    for c in (dm, ml):
                        c.change(plot_sinusoidal, [dm, ml], pe)

            # ============ TRAIN / EVAL ============
            with gr.Tab("Train / Eval (CPU sandbox)"):
                gr.Markdown(
                    "Train a **lightweight CPU baseline** (not the paper's GPU models) on past air "
                    "temperature, or evaluate your own predictions against bundled ground truth."
                )
                with gr.Tab("Train a baseline"):
                    with gr.Row():
                        gt = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                        stt = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                        mdl = gr.Dropdown(["Ridge regression", "MLP (64)"], value="Ridge regression", label="Model")
                        win = gr.Slider(3, 60, value=14, step=1, label="window (days)")
                    btn = gr.Button("Train & evaluate", variant="primary")
                    tmsg = gr.Markdown()
                    tfig = gr.Plot()
                    gt.change(lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), gt, stt)
                    btn.click(train_eval, [gt, stt, mdl, win], [tfig, tmsg])
                with gr.Tab("Evaluate predictions"):
                    gr.Markdown(
                        "Upload a CSV with a **numeric prediction column** (named like `prediction`/"
                        "`wt_hat`); it is aligned to the chosen station's bundled test ground truth and "
                        "scored with RMSE / MAE / NSE."
                    )
                    with gr.Row():
                        ge = gr.Dropdown(GRAPHS, value="zurich", label="Ground-truth dataset")
                        ste = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                    ef = gr.File(label="Predictions file", file_types=ACCEPTED)
                    emsg = gr.Markdown()
                    efig = gr.Plot()
                    ge.change(lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), ge, ste)
                    ef.change(eval_predictions, [ef, ge, ste], [efig, emsg])

            # ============ UPLOAD ============
            with gr.Tab("Upload your data"):
                gr.Markdown("## Plot your own data\n" + UPLOAD_REQ)
                with gr.Row():
                    gu = gr.Dropdown(GRAPHS, value="swiss-2010", label="Reference dataset (context)")
                up = gr.File(label="Upload (.csv/.tsv/.xlsx/.xls/.json/.parquet)", file_types=ACCEPTED)
                umsg = gr.Markdown()
                uplot = gr.Plot()
                up.change(handle_upload, [up, gu], [uplot, umsg])

            # ============ INFERENCE ============
            with gr.Tab("Inference & prediction"):
                gr.Markdown(
                    "Predict with the model you trained in the sandbox, or **upload your own model** — a "
                    "scikit-learn estimator (`.joblib`/`.pkl`, CPU) or a **TorchScript** module (`.pt`, GPU when "
                    "available). Input is a window of past daily air-temperature values. The **live** button "
                    "streams the prediction as it is produced.\n\n"
                    "⚠️ *Model files execute code when loaded — only upload models you trust. Enabled locally; "
                    "disabled on the hosted Space for security.*"
                )
                gr.Markdown(detect_resources())
                with gr.Row():
                    gi = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                    sti = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                    spi = gr.Dropdown(["test", "train"], value="test", label="Split")
                with gr.Row():
                    srci = gr.Dropdown(INFER_SOURCES, value="Last trained sandbox model", label="Model source")
                    devi = gr.Dropdown(device_choices(), value="auto", label="Device (TorchScript)")
                    wini = gr.Slider(3, 60, value=14, step=1, label="input window (features)")
                mfile = gr.File(label="Upload model (.joblib/.pkl/.pt)", file_types=MODEL_EXT)
                mstat = gr.Markdown()
                mfile.change(model_status, mfile, mstat)
                with gr.Row():
                    runb = gr.Button("Predict", variant="primary")
                    liveb = gr.Button("▶ Predict live (streaming)")
                imsg = gr.Markdown()
                ifig = gr.Plot()
                gi.change(lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), gi, sti)
                runb.click(run_inference, [gi, sti, spi, srci, mfile, wini, devi], [ifig, imsg])
                liveb.click(stream_inference, [gi, sti, spi, srci, mfile, wini, devi], [ifig, imsg])

            # ============ ANALYSIS ============
            with gr.Tab("Analysis"):
                with gr.Tab("Residuals"):
                    gr.Markdown(
                        "Residual (prediction − observation) over time and its distribution — where and how the "
                        "model is biased."
                    )
                    with gr.Row():
                        ga = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                        sta = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                        spa = gr.Dropdown(["test", "train"], value="test", label="Split")
                        srca = gr.Dropdown(INFER_SOURCES, value="Last trained sandbox model", label="Model source")
                        deva = gr.Dropdown(device_choices(), value="auto", label="Device")
                        wina = gr.Slider(3, 60, value=14, step=1, label="window")
                    afile = gr.File(label="Model (if 'Uploaded model file')", file_types=MODEL_EXT)
                    ab = gr.Button("Analyse residuals", variant="primary")
                    amsg = gr.Markdown()
                    afig = gr.Plot()
                    ga.change(lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), ga, sta)
                    ab.click(residual_analysis, [ga, sta, spa, srca, afile, wina, deva], [afig, amsg])
                with gr.Tab("Seasonal error"):
                    gr.Markdown(
                        "Mean absolute error by calendar month — summer (fish-stress season) accuracy matters most "
                        "for ecology."
                    )
                    with gr.Row():
                        gse = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                        stse = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                        spse = gr.Dropdown(["test", "train"], value="test", label="Split")
                        srcse = gr.Dropdown(INFER_SOURCES, value="Last trained sandbox model", label="Model source")
                        devse = gr.Dropdown(device_choices(), value="auto", label="Device")
                        winse = gr.Slider(3, 60, value=14, step=1, label="window")
                    sefile = gr.File(label="Model (if uploaded)", file_types=MODEL_EXT)
                    seb = gr.Button("Seasonal error", variant="primary")
                    semsg = gr.Markdown()
                    sefig = gr.Plot()
                    gse.change(
                        lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), gse, stse
                    )
                    seb.click(seasonal_error, [gse, stse, spse, srcse, sefile, winse, devse], [sefig, semsg])
                with gr.Tab("Threshold exceedance"):
                    gr.Markdown(
                        "Ecological/regulatory **threshold analysis**: observed days per year above a temperature "
                        "limit (e.g. 25 °C fish-stress)."
                    )
                    with gr.Row():
                        gth = gr.Dropdown(GRAPHS, value="zurich", label="Dataset")
                        stth = gr.Dropdown(stations_of("zurich"), value=stations_of("zurich")[0], label="Station")
                        thh = gr.Dropdown(THRESHOLDS[1:], value="25 °C (fish-stress / regulatory)", label="Threshold")
                    thb = gr.Button("Exceedance by year", variant="primary")
                    thmsg = gr.Markdown()
                    thfig = gr.Plot()
                    gth.change(
                        lambda x: gr.update(choices=stations_of(x), value=(stations_of(x) or [None])[0]), gth, stth
                    )
                    thb.click(threshold_exceedance, [gth, stth, thh], [thfig, thmsg])
                with gr.Tab("Model ranking"):
                    gr.Markdown(
                        "Rank all eight benchmarked architectures by their mean per-station error on this dataset."
                    )
                    with gr.Row():
                        gmr = gr.Dropdown(GRAPHS, value="swiss-1990", label="Dataset")
                        mmr = gr.Dropdown(METRICS, value="RMSE", label="Metric")
                    mrb = gr.Button("Rank models", variant="primary")
                    mrmsg = gr.Markdown()
                    mrfig = gr.Plot()
                    mrb.click(model_ranking, [gmr, mmr], [mrfig, mrmsg])

            # ============ ABOUT ============
            with gr.Tab("About & resources"):
                gr.Markdown(detect_resources())
                gr.Markdown(RESOURCES_MD)

    return demo


demo = ui()


def build_demo():
    """Return the workbench Blocks (used by the package + HF entry points)."""
    return demo


def main():
    demo.launch(
        server_name=os.environ.get("SRN_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("SRN_PORT", "7860")),
        inbrowser=bool(os.environ.get("SRN_INBROWSER")),
    )


if __name__ == "__main__":
    main()
