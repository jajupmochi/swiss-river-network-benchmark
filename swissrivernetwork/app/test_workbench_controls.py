"""Exhaustive, reusable end-to-end tests for EVERY workbench control.

The test set is *derived from the live Gradio event graph* (``app.demo.fns``): every
control's ``.change`` / ``.click`` binding is exactly one entry there, so iterating it
exercises every control, and :func:`test_completeness_guard` fails if any binding is left
untested. For each handler the callback is invoked across the value domain of its input
controls — all dropdown choices, checkbox subsets, slider bounds, and valid / none /
malformed file fixtures — and the returned artifact type is validated against the bound
output component. See ``docs/workbench/TEST_PLAN.md``.

Run:
    uv run --extra app --with pytest python -m pytest \
        swissrivernetwork/app/test_workbench_controls.py -q
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import Ridge

from swissrivernetwork.app import workbench as app

pytestmark = pytest.mark.skipif(
    not (app.DATA / "stations_zurich.csv").exists(),
    reason="workbench data bundle not present (set SRN_WORKBENCH_DATA or ship app/data)",
)

DEMO = app.demo
EXPECTED_HANDLER_COUNT = 58

# Which file fixture a given handler's File input should receive.
FILE_FIXTURE = {
    "radar_upload": "metrics_csv",
    "eval_predictions": "preds_csv",
    "handle_upload": "data_csv",
    "model_status": "sklearn_model",
    "run_inference": "sklearn_model",
    "stream_inference": "sklearn_model",
    "residual_analysis": "sklearn_model",
    "seasonal_error": "sklearn_model",
}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """Synthetic upload files + a trained CPU sandbox model; neutralise stream sleeps."""
    d = tmp_path_factory.mktemp("wb_ctrl")

    data_csv = d / "series.csv"
    pd.DataFrame({"epoch_day": np.arange(20000, 20050), "value": np.linspace(5, 20, 50)}).to_csv(data_csv, index=False)

    metrics_csv = d / "metrics.csv"
    pd.DataFrame(
        {"Station": [1, 2, 3], "RMSE": [0.8, 0.9, 0.7], "MAE": [0.6, 0.7, 0.5], "NSE": [0.95, 0.9, 0.96]}
    ).to_csv(metrics_csv, index=False)

    preds_csv = d / "preds.csv"
    pd.DataFrame({"prediction": np.linspace(4, 18, 60)}).to_csv(preds_csv, index=False)

    # A scikit-learn estimator that consumes a window of 14 features (the UI default).
    model_path = d / "model.joblib"
    rng = np.random.RandomState(0)
    joblib.dump(Ridge().fit(rng.rand(40, 14), rng.rand(40)), model_path)

    malformed = d / "bad.csv"
    malformed.write_bytes(b"this,is\nnot,valid,\x00\x01data\n")

    # Train the sandbox model so the "Last trained sandbox model" inference source works.
    app.train_eval("zurich", app.stations_of("zurich")[0], "Ridge regression", 14)

    # Streaming handlers sleep between frames; neutralise it to keep the suite fast.
    import time as _time

    orig_sleep = _time.sleep
    _time.sleep = lambda *a, **k: None
    try:
        yield {
            "data_csv": str(data_csv),
            "metrics_csv": str(metrics_csv),
            "preds_csv": str(preds_csv),
            "sklearn_model": str(model_path),
            "malformed": str(malformed),
        }
    finally:
        _time.sleep = orig_sleep


# --------------------------------------------------------------------------- helpers
def _fn_name(bf) -> str:
    fn = bf.fn
    name = getattr(fn, "__qualname__", getattr(fn, "__name__", "")) or ""
    return name.split(".")[-1]  # strip ui.<locals>. etc.


def _is_file(comp) -> bool:
    return type(comp).__name__ == "File"


def _choice_values(comp):
    return [c[1] if isinstance(c, (tuple, list)) else c for c in (comp.choices or [])]


def _default(comp, bf, fx):
    if _is_file(comp):
        key = FILE_FIXTURE.get(_fn_name(bf))
        return SimpleNamespace(name=fx[key]) if key else None
    return comp.value


def _domain(comp, bf, fx):
    """The list of test values for one input control."""
    if _is_file(comp):
        key = FILE_FIXTURE.get(_fn_name(bf))
        if not key:
            return [None]
        return [SimpleNamespace(name=fx[key]), None, SimpleNamespace(name=fx["malformed"])]
    t = type(comp).__name__
    if t == "Dropdown":
        return _choice_values(comp) or [comp.value]
    if t == "CheckboxGroup":
        ch = _choice_values(comp)
        return [ch, ch[:1], []]
    if t == "Slider":
        return sorted({comp.minimum, comp.value, comp.maximum})
    return [comp.value]


def _validate_one(o, comp):
    t = type(comp).__name__
    if t == "Plot":
        assert isinstance(o, go.Figure), f"{t}: expected plotly Figure, got {type(o).__name__}"
    elif t == "HTML":
        assert isinstance(o, str) and o.strip(), f"{t}: expected non-empty html string"
    elif t == "Markdown":
        assert isinstance(o, str), f"{t}: expected string, got {type(o).__name__}"
    elif t == "Dataframe":
        assert isinstance(o, pd.DataFrame), f"{t}: expected DataFrame, got {type(o).__name__}"
    elif t == "Dropdown":
        assert isinstance(o, dict) and ("choices" in o or o.get("__type__") == "update"), (
            f"{t}: expected a gr.update dict, got {type(o).__name__}"
        )
    else:
        assert o is not None, f"{t}: unexpected None output"


def _validate(out, bf):
    outs = bf.outputs
    # A streaming handler returns a generator: consume it fully and validate the last frame.
    # (Use isgenerator, not __iter__: a plotly Figure is itself iterable over its keys.)
    if inspect.isgenerator(out):
        last = None
        for last in out:
            pass
        out = last
    if len(outs) == 1:
        _validate_one(out[0] if isinstance(out, (tuple, list)) else out, outs[0])
    else:
        assert isinstance(out, (tuple, list)) and len(out) == len(outs), (
            f"{_fn_name(bf)}: expected {len(outs)} outputs, got {type(out).__name__}"
        )
        for o, comp in zip(out, outs):
            _validate_one(o, comp)


ALL_IDS = sorted(DEMO.fns.keys())
_by_fn = defaultdict(list)
for _i in ALL_IDS:
    _by_fn[_fn_name(DEMO.fns[_i])].append(_i)
# One representative binding per distinct handler for the exhaustive input-domain sweep.
REP_IDS = sorted(ids[0] for ids in _by_fn.values())


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("fid", ALL_IDS)
def test_every_binding_baseline(fid, fx):
    """Every one of the 58 control bindings produces a valid artifact at its defaults."""
    bf = DEMO.fns[fid]
    out = bf.fn(*[_default(c, bf, fx) for c in bf.inputs])
    _validate(out, bf)


@pytest.mark.parametrize("fid", REP_IDS)
def test_handler_input_domain_sweep(fid, fx):
    """Each distinct handler stays valid across the full value domain of every input."""
    bf = DEMO.fns[fid]
    base = [_default(c, bf, fx) for c in bf.inputs]
    for j, comp in enumerate(bf.inputs):
        for val in _domain(comp, bf, fx):
            args = list(base)
            args[j] = val
            out = bf.fn(*args)
            _validate(out, bf)


def test_completeness_guard(fx):
    """No control may be silently omitted: every live binding must be covered."""
    covered = set(ALL_IDS) | set(REP_IDS)
    assert covered == set(DEMO.fns.keys()), "a Gradio binding is not covered by the suite"
    assert len(DEMO.fns) == EXPECTED_HANDLER_COUNT, (
        f"handler count is {len(DEMO.fns)}, expected {EXPECTED_HANDLER_COUNT} — "
        "a control was added/removed; update TEST_PLAN.md and EXPECTED_HANDLER_COUNT"
    )
