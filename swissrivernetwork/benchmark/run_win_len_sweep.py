"""Driver for the window-length sweep evaluation (paper Fig. 4 / HLE metric).

Background: the eval-window leak
--------------------------------
We train each model once at ``window_len = TRAIN_WIN_LEN = 90`` and then
evaluate that same checkpoint at many shorter / longer eval window lengths W.
At test time the *isolated* models (lstm, transformer[xPE]) dump their
``wt_hat`` predictions so the *graphlet* models can later read them back as
neighbor features.

Before the fix, the dump path was keyed only by the *trained* config
(``path_extra_keys``). That meant every eval W wrote to the same directory,
and each later W overwrote the previous one. When graphlet then evaluated at,
say, W=7, it actually read the isolated model's W=150 predictions — giving
graphlet an unfair long-history advantage exactly in the regime the paper
claims it dominates. See ``util.get_evaluation_path_keys``: it now appends
an ``-evalwl{W}`` suffix whenever the eval W differs from the trained W, so
every (trained-config, eval-W) pair gets its own isolated sub-dir.

Which models / metrics are actually affected
--------------------------------------------
- **Graphlet** (lstm + transformer): consumes neighbor ``wt_hat`` -> values
  change for every W ≠ 90.
- **Isolated / Embedded / ST-GNN**: don't read neighbor dumps -> unchanged.
- **HLE** (radar plot, ``results_in_polar.ipynb``): its short-window weights
  ``w(l)=2^(-l/45)`` dominate -> Graphlet panel moves; others unchanged.
- **Fig. 4** (``window_lens_resu.ipynb``): Graphlet panel moves.
- **Table 3 / Fig. 3 / noise / any wl=90 result**: unchanged (eval_wl == trained_wl).

Two-phase execution
-------------------
Phase 1 (ISOLATED)  lstm + transformer[xPE]
    -> writes ``wt_hat`` to
    ``dump/predictions/<path_extra_keys>-evalwl{W}/`` (one dir per W)
Phase 2 (GRAPHLET)  graphlet + transformer_graphlet
    -> reads the Phase-1 dumps back as neighbor features.

Running Phase 2 without Phase 1 for the same W fails with ``FileNotFoundError``.
The driver below runs Phase 1 for every (graph, W) before entering Phase 2,
so the ordering is automatic — don't reshuffle ``main()``.

Prerequisites
-------------
- Tuned + trained checkpoints at ``trained_wl = TRAIN_WIN_LEN`` must already
  exist under ``OUTPUT_DIR`` for every (graph, method). Run ``ray_tune`` first
  if they're missing.
- This sweep uses test-time evaluation only; it does not re-train anything.

Output layout
-------------
Appends one row per ``(method, wl[, pe])`` to:
    ``visualize_results/outputs/win_lens/{graph}_{method}_win_lens_resu.csv``

Strict duplicate-row guard
--------------------------
If the target CSV already has a row with the same W (or same W + PE for
transformers), the driver raises ``FileExistsError`` and prints a backup
command. This is deliberate: appending onto a CSV written under the old
(buggy) path layout would silently interleave stale Graphlet rows with fresh
ones. Back up / remove the old file, then re-run.

Resuming a partially completed sweep
------------------------------------
Set ``RESUME_SKIP_EXISTING = True`` in the ``__main__`` block to switch the
strict guard into *skip-if-present* mode: rows already on disk are left
untouched and the driver moves on to the next ``(method, wl[, pe])`` tuple.
Use this only when you trust the existing rows (i.e. they were written by
the post-fix path layout). Orthogonally, ``SKIP_PHASE_ISOLATED = True`` lets
you skip Phase 1 entirely when the ISOLATED ``wt_hat`` dumps are already on
disk — the usual case when Phase 2 crashed mid-way.

Typical recovery recipe after a Phase-2 crash:
1. Keep the (partial) ``{graph}_{method}_win_lens_resu.csv`` files as-is.
2. Set ``RESUME_SKIP_EXISTING = True`` and ``SKIP_PHASE_ISOLATED = True``.
3. Re-run ``uv run python -m swissrivernetwork.benchmark.run_win_len_sweep``.

Entry points
------------
    uv run python -m swissrivernetwork.benchmark.run_win_len_sweep

Toggle ``DEBUG_SINGLE`` in ``__main__`` to restrict the sweep to one
(graph, method, wl) tuple for PyCharm breakpoint work. Set breakpoints in
``process_method`` (ray_evaluation.py), ``test_lstm`` / ``test_graphlet``
(test_isolated_station.py), or ``get_evaluation_path_keys`` (util.py).

Downstream consumers of the CSVs
--------------------------------
- ``visualize_results/window_lens_resu.ipynb``  -> paper Fig. 4
- ``visualize_results/visual_win_lens.ipynb``   -> interactive Plotly views
- ``visualize_results/results_in_polar.ipynb``  -> HLE dimension of Fig. 2
"""

from pathlib import Path

import numpy as np
import pandas as pd

from swissrivernetwork.benchmark.ray_evaluation import process_method
from swissrivernetwork.benchmark.util import (
    INFO_TAG,
    SUCCESS_TAG,
    get_run_extra_key,
    is_transformer_model,
)

CUR_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CUR_DIR / "outputs" / "ray_results"
DUMP_DIR = CUR_DIR / "visualize_results" / "outputs" / "win_lens"

TRAIN_WIN_LEN = 90
POSITIONAL_ENCODINGS = ["learnable", "sinusoidal", "rope"]
# Subsequence-mode caps as used by visualize_results/window_lens_resu.ipynb.
MAX_DAYS_PER_GRAPH = {"swiss-1990": 853, "swiss-2010": 1096, "zurich": 1035}


def curate_window_lens(raw: np.ndarray, graph_name: str, cap: int = 150) -> list[int]:
    """Clamp raw window lengths to the graph's day cap and always include the limit."""
    limit = min(MAX_DAYS_PER_GRAPH[graph_name], cap)
    arr = np.array(raw)
    arr = arr[arr < limit]
    arr = np.unique(np.concatenate((arr, [limit])))
    return sorted(int(v) for v in arr)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _row_from_df(df_data: pd.DataFrame, extra: dict, wl: int) -> dict:
    row: dict = {"window_len": wl}
    for metric in ["RMSE", "MAE", "NSE"]:
        for stat in ["Mean", "Std", "Median", "Min", "Max"]:
            col = f"{metric}_{stat}"
            if metric in df_data.columns:
                val = df_data[df_data["Station"] == stat][metric]
                row[col] = val.values[0] if not val.empty else np.nan
            else:
                row[col] = np.nan
    for c in [c for c in df_data.columns if c.startswith("extra__")]:
        val = df_data[df_data["Station"] == "Mean"][c]
        row[f"{c}_Mean"] = val.values[0] if not val.empty else np.nan
    for k, v in extra.items():
        if f"extra__{k}" not in row:
            row[f"extra__{k}"] = v
    return row


def _row_present(dump_dir: Path, graph_name: str, method: str, wl: int, pe: str | None = None) -> bool:
    """Return True iff a row for (wl) or (wl, pe) already exists in the summary CSV."""
    path = dump_dir / f"{graph_name}_{method}_win_lens_resu.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if df.empty or "window_len" not in df.columns or wl not in df["window_len"].values:
        return False
    rows_at_wl = df[df["window_len"] == wl]
    if pe is None:
        return True
    return "positional_encoding" in rows_at_wl.columns and pe in rows_at_wl["positional_encoding"].values


def _assert_row_absent(dump_dir: Path, graph_name: str, method: str, wl: int, pe: str | None = None):
    """Raise FileExistsError if a row for (wl) or (wl, pe) already exists in the summary CSV.

    Appending to a CSV that was produced before the evalwl-path fix would interleave
    stale rows with freshly computed ones, making it impossible to tell which is which.
    Force the user to manually back up / delete the file before rerunning.
    """
    if not _row_present(dump_dir, graph_name, method, wl, pe):
        return
    path = dump_dir / f"{graph_name}_{method}_win_lens_resu.csv"
    conflict_key = f"wl={wl}, pe={pe}" if pe is not None else f"wl={wl}"
    backup_cmd = f"    mkdir -p '{path.parent}/_backup_pre_fix'\n    mv '{path}' '{path.parent}/_backup_pre_fix/'"
    raise FileExistsError(
        f"Summary CSV already contains {conflict_key} — refusing to append and mix stale rows.\n"
        f"  file: {path}\n"
        f"Back up and remove it before re-running, e.g.:\n{backup_cmd}\n"
        f"If the existing rows are known-good (post-fix), set RESUME_SKIP_EXISTING = True to skip them instead."
    )


def _save_row(dump_dir: Path, graph_name: str, method: str, row: dict, pe: str | None = None):
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"{graph_name}_{method}_win_lens_resu.csv"
    df = _read_csv(path)
    new_row = dict(row)
    if pe is not None:
        new_row["positional_encoding"] = pe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)
    pe_tag = f"  pe={pe}" if pe else ""
    print(f"{SUCCESS_TAG}saved {path.name}  wl={row['window_len']}{pe_tag}")


def run_single(graph_name: str, method: str, wl: int, settings: dict) -> dict:
    """Evaluate one ``(graph, method)`` at eval window length ``wl`` and return a summary row.

    Pins ``path_extra_keys`` to the trained checkpoint (always ``TRAIN_WIN_LEN``)
    so :func:`process_method` reads the right model while evaluating at ``wl``.
    """
    s = {**settings, "window_len": wl}
    # path_extra_keys pins the TRAINED checkpoint directory (always wl=TRAIN_WIN_LEN).
    # get_evaluation_path_keys additionally appends -evalwl{W} when wl != TRAIN_WIN_LEN.
    s["path_extra_keys"] = get_run_extra_key({**s, "window_len": TRAIN_WIN_LEN})
    df_data, extra_resu = process_method(graph_name, method, output_dir=OUTPUT_DIR, settings=s, return_extra=True)
    return _row_from_df(df_data, extra_resu, wl)


def run_method_at_wl(
    graph_name: str,
    method: str,
    wl: int,
    base_settings: dict,
    dump_dir: Path,
    resume: bool = False,
):
    """Evaluate ``method`` at window length ``wl`` and append the result row(s) to disk.

    Transformer methods loop over all positional encodings; others run once with
    ``positional_encoding="none"``. Honors the ``resume`` skip-if-present guard
    and the strict duplicate-row guard otherwise.
    """
    if is_transformer_model(method):
        for pe in POSITIONAL_ENCODINGS:
            if resume and _row_present(dump_dir, graph_name, method, wl, pe):
                print(f"{INFO_TAG}resume: skipping existing row  {graph_name}/{method}  wl={wl}  pe={pe}")
                continue
            _assert_row_absent(dump_dir, graph_name, method, wl, pe)
            settings = {
                **base_settings,
                "positional_encoding": pe,
                "max_len": max(500, wl),
            }
            row = run_single(graph_name, method, wl, settings)
            _save_row(dump_dir, graph_name, method, row, pe=pe)
    else:
        if resume and _row_present(dump_dir, graph_name, method, wl):
            print(f"{INFO_TAG}resume: skipping existing row  {graph_name}/{method}  wl={wl}")
            return
        _assert_row_absent(dump_dir, graph_name, method, wl)
        settings = {**base_settings, "positional_encoding": "none"}
        row = run_single(graph_name, method, wl, settings)
        _save_row(dump_dir, graph_name, method, row)


def run_phase(
    phase_label: str,
    graph_names: list[str],
    methods: list[str],
    window_lens_raw: np.ndarray,
    base_settings: dict,
    dump_dir: Path,
    resume: bool = False,
):
    """Run one sweep phase: every ``(method, curated window length)`` over all graphs."""
    print(f"\n{INFO_TAG}=========== PHASE: {phase_label} ===========")
    for graph_name in graph_names:
        wls = curate_window_lens(window_lens_raw, graph_name)
        print(f"{INFO_TAG}graph={graph_name}  window_lens={wls}")
        for method in methods:
            for wl in wls:
                print(f"{INFO_TAG}>>> {phase_label} :: {graph_name} / {method} / wl={wl}")
                run_method_at_wl(graph_name, method, wl, base_settings, dump_dir, resume=resume)


def main(resume: bool = False, skip_isolated: bool = False):
    """Run the full two-phase window-length sweep (ISOLATED then GRAPHLET).

    Args:
        resume: Skip rows already present on disk instead of raising.
        skip_isolated: Skip Phase 1, relying on existing ``wt_hat`` dumps.
    """
    GRAPH_NAMES = ["swiss-1990", "swiss-2010", "zurich"]
    # [1, 3, 5, 7, 15, 30, 60, 90, 120, 150] — same sweep as paper Fig. 4.
    WINDOW_LENS = np.concatenate(([1, 3, 5, 7, 15], 30 * np.arange(1, 6)))

    base_settings = {
        "missing_value_method": None,
        "use_current_x": True,  # nowcasting. Set False for forecasting (and add future_steps).
        "future_steps": 7,  # only used when use_current_x=False
        "extrapo_mode": "future_embedding",
        "noise_type": None,
        "noise_kwargs": {"probability": 0.1, "scale_factor": 5.0},
        "verbose": 1,
        "env": "cli",
    }

    # Phase 1 must complete before Phase 2 for each W.
    if skip_isolated:
        print(f"{INFO_TAG}SKIP_PHASE_ISOLATED=True — relying on existing wt_hat dumps on disk.")
    else:
        run_phase("ISOLATED", GRAPH_NAMES, ["lstm", "transformer"], WINDOW_LENS, base_settings, DUMP_DIR, resume=resume)
    run_phase(
        "GRAPHLET",
        GRAPH_NAMES,
        ["graphlet", "transformer_graphlet"],
        WINDOW_LENS,
        base_settings,
        DUMP_DIR,
        resume=resume,
    )


def debug_single():
    """Minimal case for PyCharm breakpoint debugging — one (graph, method, wl) tuple."""
    base_settings = {
        "missing_value_method": None,
        "use_current_x": True,
        "future_steps": 7,
        "extrapo_mode": "future_embedding",
        "noise_type": None,
        "noise_kwargs": {"probability": 0.1, "scale_factor": 5.0},
        "verbose": 2,
        "env": "cli",
    }
    graph, wl = "zurich", 7
    run_method_at_wl(graph, "lstm", wl, base_settings, DUMP_DIR)
    run_method_at_wl(graph, "graphlet", wl, base_settings, DUMP_DIR)


if __name__ == "__main__":
    DEBUG_SINGLE = False
    # Resume knobs (default: strict, full two-phase run).
    # Set both True after a Phase-2 crash: keep the partial CSVs, keep the
    # ISOLATED dumps on disk, skip rows that are already written.
    RESUME_SKIP_EXISTING = True
    SKIP_PHASE_ISOLATED = True
    if DEBUG_SINGLE:
        debug_single()
    else:
        main(resume=RESUME_SKIP_EXISTING, skip_isolated=SKIP_PHASE_ISOLATED)
