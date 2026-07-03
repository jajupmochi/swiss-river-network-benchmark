"""Automated tests for the Swiss River Network Benchmark local workbench.

Run:  uv run --extra app --with pytest python -m pytest swissrivernetwork/app/test_workbench.py -q

Covers every loader, metric, visualization, the map + three.js 3D view, the radar
(incl. upload overlay), the forecast outlook, the train/eval sandbox, the
bring-your-own-model inference (scikit-learn + TorchScript) with live streaming,
the residual / seasonal / threshold / ranking analysis, resource detection, and
the multi-format upload router (valid + malformed inputs).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from swissrivernetwork.app import workbench as app

G, Z = "swiss-1990", "zurich"

# The whole suite needs a workbench data bundle; skip cleanly if it is absent
# (e.g. a checkout without app/data and without $SRN_WORKBENCH_DATA set).
pytestmark = pytest.mark.skipif(
    not (app.DATA / "stations_zurich.csv").exists(),
    reason="workbench data bundle not present (set SRN_WORKBENCH_DATA or ship app/data)",
)


class _F:
    def __init__(self, name):
        self.name = name


def _st(graph):
    s = app.stations_of(graph)
    assert s, f"no stations for {graph}"
    return s[0]


def _tmp(df, suffix=".csv", truncate=False):
    t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    t.close()
    if truncate:
        open(t.name, "w").close()
    elif suffix == ".csv":
        df.to_csv(t.name, index=False)
    elif suffix == ".tsv":
        df.to_csv(t.name, sep="\t", index=False)
    elif suffix == ".json":
        df.to_json(t.name, orient="records", date_format="iso")
    elif suffix == ".parquet":
        df.to_parquet(t.name)
    return t.name


def _sk_model(window=14):
    from sklearn.linear_model import Ridge

    fr = app._infer_frame(Z, _st(Z), "train", window)
    assert fr is not None
    X, y, _ = fr
    return Ridge().fit(X, y)


def _dump(model, suffix=".joblib"):
    import joblib

    t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    t.close()
    joblib.dump(model, t.name)
    return t.name


# ---------------------------------------------------------------- loaders
def test_stations_edges():
    for g in app.GRAPHS:
        s = app.stations_df(g)
        assert not s.empty and {"station", "lat", "lon"} <= set(s.columns)
        assert 45 < s.lat.mean() < 48 and 5 < s.lon.mean() < 11  # in Switzerland
        assert not app.edges_df(g).empty


def test_series_and_temps():
    df = app.series_df(Z, "train")
    assert not df.empty and "epoch_day" in df.columns
    t = app.station_temps(Z, "max")
    assert t and all(isinstance(v, float) for v in t.values())


def test_result_loaders():
    assert not app.win_lens(G, "lstm").empty
    assert not app.future_steps(G, "lstm").empty
    assert not app.noise_df(G, "lstm_embedding", "gaussian_a").empty
    assert not app.station_resu(G, "transformer_embedding").empty  # exercises PE fallback


def test_metrics():
    y = np.array([1.0, 2, 3, 4])
    assert app.rmse(y, y) == 0 and app.mae(y, y) == 0 and app.nse(y, y) == 1
    assert app.rmse(y, y + 1) == pytest.approx(1.0)


# ---------------------------------------------------------------- map & 3D
def test_build_map_metric_and_temp():
    assert "leaflet" in app.build_map(G, "transformer_embedding", "RMSE").lower()
    assert "leaflet" in app.build_map(G, "water-temp (max)", "RMSE").lower()


def test_threejs():
    h = app.threejs_html(G, "water-temp (max)", "RMSE")
    assert "<iframe" in h and "three" in h.lower() and "srcdoc" in h


# ---------------------------------------------------------------- single plots
@pytest.mark.parametrize(
    "fn,args,minn",
    [
        (lambda g: app.plot_series(g, _st(g), "train"), (Z,), 2),
        (lambda g: app.plot_seasonal(g, _st(g), "train"), (Z,), 2),
        (lambda g: app.plot_coverage(g, "train"), (G,), 1),
        (lambda g: app.plot_per_station(g, "transformer_embedding", "RMSE"), (G,), 1),
    ],
)
def test_single_plots(fn, args, minn):
    f = fn(*args)
    assert isinstance(f, go.Figure) and len(f.data) >= minn


def test_sinusoidal():
    f = app.plot_sinusoidal(64, 120)
    assert isinstance(f, go.Figure) and len(f.data) == 1


# ---------------------------------------------------------------- multi-model plots
def test_multi_model_plots():
    ms = app.METHODS[:4]
    assert len(app.plot_win_lens(G, ms, "RMSE_Mean").data) == 4
    assert len(app.plot_forecast(G, ms, "RMSE_Mean").data) == 4
    assert len(app.plot_noise(G, ms, "gaussian_a", "RMSE_Mean").data) == 4


def test_radar_and_upload_overlay():
    f = app.plot_radar(G, ["lstm_embedding", "transformer_embedding"])
    assert isinstance(f, go.Figure) and len(f.data) >= 2
    p = _tmp(pd.DataFrame({"Station": [1, 2], "RMSE": [0.8, 0.9], "MAE": [0.6, 0.7], "NSE": [0.95, 0.96]}))
    f2 = app.radar_upload(G, ["lstm_embedding"], _F(p))
    assert "uploaded" in [t.name for t in f2.data]
    os.unlink(p)


def test_forecast_outlook():
    f = app.forecast_outlook(Z, _st(Z), "25 °C (fish-stress / regulatory)", 28)
    assert isinstance(f, go.Figure) and len(f.data) >= 3  # observed + band + median


def test_results_table():
    df = app.results_table(G, "nowcasting", "Mean")
    assert len(df) == 8 and "RMSE" in df.columns


# ---------------------------------------------------------------- train / eval
def test_train_eval():
    for model in ("Ridge regression", "MLP (64)"):
        f, m = app.train_eval(Z, _st(Z), model, 14)
        assert isinstance(f, go.Figure) and "RMSE=" in m


def test_eval_predictions():
    te = app.series_df(Z, "test")
    p = _tmp(pd.DataFrame({"prediction": te[f"{_st(Z)}_wt"].values[:120]}))
    f, m = app.eval_predictions(_F(p), Z, _st(Z))
    assert isinstance(f, go.Figure) and "RMSE=" in m
    os.unlink(p)


# ---------------------------------------------------------------- resources / device
def test_resources_and_devices():
    txt = app.detect_resources()
    assert isinstance(txt, str) and "CPU cores" in txt
    ch = app.device_choices()
    assert "auto" in ch and "cpu" in ch


# ---------------------------------------------------------------- bring-your-own-model
def test_load_user_model_ok_and_bad():
    p = _dump(_sk_model())
    m, kind, msg = app.load_user_model(_F(p))
    assert kind == "sklearn" and hasattr(m, "predict") and "✅" in msg
    os.unlink(p)
    bad = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    bad.write(b"not a model")
    bad.close()
    m2, kind2, msg2 = app.load_user_model(_F(bad.name))
    assert m2 is None and "❌" in msg2
    os.unlink(bad.name)


def test_run_and_stream_inference():
    st = _st(Z)
    p = _dump(_sk_model(14))
    fig, msg = app.run_inference(Z, st, "test", "Uploaded model file", _F(p), 14, "auto")
    assert isinstance(fig, go.Figure) and "RMSE=" in msg
    frames = list(app.stream_inference(Z, st, "test", "Uploaded model file", _F(p), 14, "auto", 0))
    assert frames and isinstance(frames[-1][0], go.Figure) and "Done" in frames[-1][1]
    os.unlink(p)


def test_sandbox_then_last_model_inference():
    st = _st(Z)
    app.train_eval(Z, st, "Ridge regression", 14)  # populates _STATE
    fig, msg = app.run_inference(Z, st, "test", "Last trained sandbox model", None, 14, "auto")
    assert isinstance(fig, go.Figure) and "RMSE=" in msg


# ---------------------------------------------------------------- analysis
def test_analysis_modules():
    st = _st(Z)
    p = _dump(_sk_model(14))
    f1, _ = app.residual_analysis(Z, st, "test", "Uploaded model file", _F(p), 14, "auto")
    f2, _ = app.seasonal_error(Z, st, "test", "Uploaded model file", _F(p), 14, "auto")
    assert isinstance(f1, go.Figure) and isinstance(f2, go.Figure)
    os.unlink(p)
    f3, m3 = app.threshold_exceedance(Z, st, "25 °C (fish-stress / regulatory)")
    assert isinstance(f3, go.Figure) and "days above" in m3
    f4, m4 = app.model_ranking(G, "RMSE")
    assert isinstance(f4, go.Figure) and "Best by mean" in m4


@pytest.mark.skipif(not app.TORCH_OK, reason="PyTorch not installed")
def test_torchscript_inference():
    import torch

    class M(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=1)

    ts = torch.jit.script(M())
    p2 = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    p2.close()
    ts.save(p2.name)
    m, kind, msg = app.load_user_model(_F(p2.name))
    assert kind == "torch" and "✅" in msg
    fig, im = app.run_inference(Z, _st(Z), "test", "Uploaded model file", _F(p2.name), 14, "auto")
    assert isinstance(fig, go.Figure) and "RMSE=" in im
    os.unlink(p2.name)


# ---------------------------------------------------------------- upload router
def test_upload_timeseries():
    p = _tmp(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "water_temp": range(10)}))
    f, m = app.handle_upload(_F(p), Z)
    assert "✅" in m and len(f.data) == 1
    os.unlink(p)


def test_upload_result_curve():
    p = _tmp(pd.DataFrame({"window_len": [30, 60, 90], "RMSE_Mean": [1.2, 0.9, 0.8]}))
    _, m = app.handle_upload(_F(p), Z)
    assert "curve" in m
    os.unlink(p)


def test_upload_per_station():
    p = _tmp(pd.DataFrame({"Station": [1, 2, 3], "RMSE": [0.8, 0.9, 0.7]}))
    _, m = app.handle_upload(_F(p), Z)
    assert "per-station" in m
    os.unlink(p)


def test_upload_bad_and_formats():
    p = _tmp(pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}))
    _, m = app.handle_upload(_F(p), Z)
    assert "❌" in m
    os.unlink(p)
    for suf in (".json", ".parquet", ".tsv"):
        q = _tmp(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=8), "v": list(range(8))}), suf)
        f, _ = app.handle_upload(_F(q), Z)
        assert isinstance(f, go.Figure)
        os.unlink(q)


def test_read_any_errors():
    with pytest.raises(Exception):
        app.read_any("/no/such/file.csv")
    empty = _tmp(None, truncate=True)
    with pytest.raises(Exception):
        app.read_any(empty)
    os.unlink(empty)
    weird = _tmp(pd.DataFrame({"a": [1]}), ".xyz")
    with pytest.raises(Exception):
        app.read_any(weird)
    os.unlink(weird)


# ---------------------------------------------------------------- UI
def test_ui_builds():
    assert app.demo is not None
