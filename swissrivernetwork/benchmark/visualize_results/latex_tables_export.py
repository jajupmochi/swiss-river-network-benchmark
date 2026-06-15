"""Export the nowcasting+forecasting error tables (paper Table 3) for EVERY
station-level statistic (Mean / Std / Median / Min / Max / CI95), not just Mean.

The data are read from the CSVs already produced by ``err_resu_to_latex_table.ipynb``
(+ ``future_steps_resu.ipynb``), so no model re-run is needed:
  - nowcasting : outputs/radar/{graph}_{method}_err_90.csv
  - forecasting: outputs/future_steps/{graph}_{method}_fs_resu.csv   (avg over horizons 1..7)
  - per-station: outputs/stations/{graph}_{method}[_{pe}]_station_resu[_f].csv  (Wilcoxon)

The 'Mean' table reproduces the paper's Table 3 exactly (verified via --verify).
For each statistic the SAME table structure is built: best-PE selection per metric,
bold+underline best model per column, coloured Transformer-vs-LSTM %-change, and
Wilcoxon significance stars.

Public API (also used by the notebook "button"):
  build_all_tables()      -> {stat: latex_table_str}
  collect_all_results()   -> nested dict with every value/best-pe/significance
  save_tables_to_tex(...) -> write one compilable .tex with all tables
  export_results_json(...)-> write the full-info JSON
  compile_tex_to_pdf(...) -> run pdflatex and return (ok, log_tail)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- config
MODULE_DIR = Path(__file__).parent.resolve()
RADAR_DIR = MODULE_DIR / "outputs" / "radar"
FS_DIR = MODULE_DIR / "outputs" / "future_steps"
STATIONS_DIR = MODULE_DIR / "outputs" / "stations"
OUT_DIR = MODULE_DIR / "outputs" / "latex_tables"

GRAPH_NAMES = ["swiss-1990", "swiss-2010", "zurich"]
ARCHITECTURES = ["Isolated", "Graphlet", "Embedded", "ST-GNN"]
ARCH_TO_METHODS = {
    "Isolated": ["lstm", "transformer"],
    "Graphlet": ["graphlet", "transformer_graphlet"],
    "Embedded": ["lstm_embedding", "transformer_embedding"],
    "ST-GNN": ["stgnn", "transformer_stgnn"],
}
METRICS = ["RMSE", "MAE", "NSE"]
STATS = ["Mean", "Std", "Median", "Min", "Max", "CI95"]
POSITIONAL_ENCODINGS = ["learnable", "sinusoidal", "rope"]  # matches the notebook
WINDOW_LEN = 90
MAX_HORIZON = 7


def is_transformer_model(method: str) -> bool:
    return method.startswith("transformer")


def _read_csv(directory: Path, graph: str, method: str, key: str) -> pd.DataFrame:
    fp = directory / f"{graph}_{method}_{key}.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()


# ---------------------------------------------------------------- gather (mirrors notebook)
def _nowcast_value(graph: str, method: str, metric_col: str):
    """Return (value, best_pe). best_pe is None for LSTM-side methods."""
    df = _read_csv(RADAR_DIR, graph, method, "err_90")
    if df.empty:
        return None, None
    df = df[df["window_len"] == WINDOW_LEN]
    if df.empty:
        return None, None
    if not is_transformer_model(method):
        return float(df.iloc[0][metric_col]), None
    if metric_col.startswith(("RMSE", "MAE")):
        best = df.loc[df[metric_col].idxmin()]
    else:  # NSE -> larger better
        best = df.loc[df[metric_col].idxmax()]
    return float(best[metric_col]), best["positional_encoding"]


def _forecast_value(graph: str, method: str, metric_col: str):
    """Average metric over horizons 1..MAX_HORIZON (mirrors aggregate_fs_metric);
    for Transformers pick the best PE. Return (value, best_pe)."""
    df = _read_csv(FS_DIR, graph, method, "fs_resu")
    if df.empty:
        return None, None
    steps = list(range(1, MAX_HORIZON + 1))

    def _avg(sub):
        vals = []
        for s in steps:
            r = sub[sub["future_step"] == s]
            if r.empty:
                return None
            vals.append(float(r.iloc[0][metric_col]))
        return sum(vals) / len(vals)

    if not is_transformer_model(method):
        return _avg(df), None
    results = []
    for pe in POSITIONAL_ENCODINGS:
        v = _avg(df[df["positional_encoding"] == pe])
        results.append(v)
    if any(v is None for v in results):
        return None, None
    idx = int(np.argmin(results) if metric_col.startswith(("RMSE", "MAE")) else np.argmax(results))
    return results[idx], POSITIONAL_ENCODINGS[idx]


def gather(stat: str) -> tuple[dict, dict]:
    """vals[casting][metric][graph][method]; best_pe[casting][metric][graph][method]."""
    vals = {"nowcasting": {}, "forecasting": {}}
    best_pe = {"nowcasting": {}, "forecasting": {}}
    for casting, fn in (("nowcasting", _nowcast_value), ("forecasting", _forecast_value)):
        for metric in METRICS:
            col = f"{metric}_{stat}"
            vals[casting][metric] = {g: {} for g in GRAPH_NAMES}
            best_pe[casting][metric] = {g: {} for g in GRAPH_NAMES}
            for g in GRAPH_NAMES:
                for arch in ARCHITECTURES:
                    for method in ARCH_TO_METHODS[arch]:
                        v, pe = fn(g, method, col)
                        vals[casting][metric][g][method] = v
                        if pe is not None:
                            best_pe[casting][metric][g][method] = pe
    return vals, best_pe


def compute_significance(best_pe: dict) -> dict:
    """Wilcoxon per (casting, graph, arch, metric) between LSTM and best-PE Transformer
    using per-station values. Depends on best_pe (hence on the stat)."""
    sig = {"nowcasting": {}, "forecasting": {}}
    for casting in ("nowcasting", "forecasting"):
        suffix = "station_resu_f" if casting == "forecasting" else "station_resu"
        for g in GRAPH_NAMES:
            sig[casting][g] = {}
            for arch in ARCHITECTURES:
                sig[casting][g][arch] = {}
                lstm_m, tr_m = ARCH_TO_METHODS[arch]
                for metric in METRICS:
                    ref = _read_csv(STATIONS_DIR, g, lstm_m, suffix)
                    pe = best_pe[casting][metric][g].get(tr_m)
                    cur = _read_csv(STATIONS_DIR, g, tr_m, f"{pe}_{suffix}") if pe else pd.DataFrame()
                    if ref.empty or cur.empty or metric not in ref or metric not in cur:
                        sig[casting][g][arch][metric] = None
                        continue
                    merged = pd.merge(
                        ref[["Station", metric]], cur[["Station", metric]], on="Station", suffixes=("_l", "_t")
                    )
                    if len(merged) < 2:
                        sig[casting][g][arch][metric] = False
                        continue
                    _, p = wilcoxon(merged[f"{metric}_l"].tolist(), merged[f"{metric}_t"].tolist())
                    sig[casting][g][arch][metric] = bool(p < 0.05)
    return sig


# ---------------------------------------------------------------- table data structure
def _build_data(vals: dict) -> list:
    """Replicate the notebook 'data' structure: index 0 = column labels; then per
    (graph, arch) a tuple (label, [[lstm 6-vec],[transformer 6-vec]]). Column order:
    RMSE/N, MAE/N, NSE/N, RMSE/F, MAE/F, NSE/F."""
    data = [["RMSE/N", "MAE/N", "NSE/N", "RMSE/F", "MAE/F", "NSE/F"]]
    order = [("nowcasting", m) for m in METRICS] + [("forecasting", m) for m in METRICS]
    for g in GRAPH_NAMES:
        for arch in ARCHITECTURES:
            lstm_m, tr_m = ARCH_TO_METHODS[arch]
            lstm_vec = [vals[c][m][g].get(lstm_m) for c, m in order]
            tr_vec = [vals[c][m][g].get(tr_m) for c, m in order]
            data.append((f"{g}, {arch}", [lstm_vec, tr_vec]))
    return data


def _get_best_performance(data: list) -> list:
    idx_best = []
    for i_graph in range(len(GRAPH_NAMES)):
        rows = []
        for i_model in range(2):
            for i_arch in range(len(ARCHITECTURES)):
                vec = data[1 + 4 * i_graph + i_arch][1][i_model][0:6]
                rows.append(np.array([np.nan if v is None else v for v in vec], dtype=float))
        arr = np.array(rows)
        max_i = np.nanargmax(arr, axis=0)
        min_i = np.nanargmin(arr, axis=0)
        best = [max_i[i] if i in (2, 5) else min_i[i] for i in range(6)]  # NSE columns: larger better
        idx_best.append(best)
    return idx_best


def _is_best(idx_best, i_graph, i_model, i_arch, i_metric) -> bool:
    return idx_best[i_graph][i_metric] == i_model * len(ARCHITECTURES) + i_arch


def _suffix(change_pct: float, larger_is_better: bool, threshold: float = 0.05) -> str:
    if change_pct is None or np.isnan(change_pct):
        return ""
    better = (change_pct >= threshold) if larger_is_better else (change_pct <= -threshold)
    worse = (change_pct <= -threshold) if larger_is_better else (change_pct >= threshold)
    color, arrow = ("Blue", "")
    if better:
        color, arrow = "Green", "$\\uparrow$"
    elif worse:
        color, arrow = "Red", "$\\downarrow$"
    return f"\\textcolor{{{color}}}{{\\scriptsize{{{abs(change_pct):.1f}\\%{arrow}}}}}~"


def _fmt(x: float) -> str:
    return f"{x:.3f}".lstrip("0").replace("-0", "-")


def build_latex_body(data: list, sig: dict) -> str:
    """Faithful re-implementation of the notebook's body-construction loop."""
    idx_best = _get_best_performance(data)
    s = ""
    for i_graph, graph in enumerate(GRAPH_NAMES):
        s += f"\\bf {graph.capitalize()} & "
        for i_model, model in enumerate(["LSTM", "Transformer"]):
            if i_model == 1:
                s += " & "
            s += f"{model} & "
            for i_arch, arch in enumerate(ARCHITECTURES):
                if not (i_arch == 0):
                    s += " & & "
                s += f"{arch} & "
                vec = data[1 + 4 * i_graph + i_arch][1][i_model][0:6]
                changes = [None] * 6
                if i_model == 1:
                    lvec = data[1 + 4 * i_graph + i_arch][1][0][0:6]
                    for k in range(6):
                        if vec[k] is not None and lvec[k] not in (None, 0):
                            changes[k] = (vec[k] - lvec[k]) / lvec[k] * 100.0
                for i_val, val in enumerate(vec):
                    if val is None:
                        s += "-"
                    else:
                        fv = _fmt(val)
                        if _is_best(idx_best, i_graph, i_model, i_arch, i_val):
                            fv = f"\\underline{{\\textbf{{{fv}}}}}"
                        larger = i_val in (2, 5)
                        s += (_suffix(changes[i_val], larger) if i_model == 1 else "") + fv
                    if i_model == 1:
                        casting = "nowcasting" if i_val < 3 else "forecasting"
                        metric = METRICS[i_val % 3]
                        if sig.get(casting, {}).get(graph, {}).get(arch, {}).get(metric):
                            s += "$^{*}$"
                    if i_val < 5:
                        s += " & "
                if i_arch < len(ARCHITECTURES) - 1:
                    s += " \\\\ \n"
            if i_model < 1:
                s += " \\\\ \n \\cmidrule(r){2-9} \n"
        if i_graph < len(GRAPH_NAMES) - 1:
            s += " \\\\ \n\\midrule \n"
    s += " \\\\"
    return s


_CAPTION = {
    "Mean": "Mean of per-station RMSE/MAE/NSE (reproduces paper Table 3).",
    "Std": "Standard deviation across stations of RMSE/MAE/NSE.",
    "Median": "Median across stations of RMSE/MAE/NSE.",
    "Min": "Minimum across stations of RMSE/MAE/NSE.",
    "Max": "Maximum across stations of RMSE/MAE/NSE.",
    "CI95": "Half-width of the 95\\% CI of the mean across stations (RMSE/MAE/NSE).",
}


def wrap_table(body: str, stat: str) -> str:
    header = (
        "\\begin{tabular}{lllcccccc}\n\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Model}} & "
        "\\multirow{2}{*}{\\textbf{Architecture}} & \\multicolumn{3}{c}{\\textbf{Nowcasting}} & "
        "\\multicolumn{3}{c}{\\textbf{Forecasting}} \\\\\n"
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
        " & & & RMSE\\,$\\downarrow$ & MAE\\,$\\downarrow$ & NSE\\,$\\uparrow$ & "
        "RMSE\\,$\\downarrow$ & MAE\\,$\\downarrow$ & NSE\\,$\\uparrow$ \\\\\n\\midrule\n"
    )
    tabular = header + body + "\n\\bottomrule\n\\end{tabular}"
    return (
        f"\\noindent{{\\bfseries Table -- {stat}: }}{_CAPTION.get(stat, '')}\\par\n"
        "{\\footnotesize Forecasting averaged over horizons 1--7; Transformer rows use the best "
        "positional encoding per metric; $^{*}$ = Wilcoxon $p<0.05$ vs.\\ LSTM; coloured values = "
        "Transformer-vs-LSTM \\%-change.}\\par\\medskip\n"
        f"\\begin{{center}}\n\\resizebox{{0.9\\textwidth}}{{!}}{{%\n{tabular}\n}}\n\\end{{center}}"
    )


def build_all_tables(stats: list[str] = STATS) -> dict[str, str]:
    out = {}
    for stat in stats:
        vals, best_pe = gather(stat)
        sig = compute_significance(best_pe)
        out[stat] = wrap_table(build_latex_body(_build_data(vals), sig), stat)
    return out


# ---------------------------------------------------------------- JSON
def collect_all_results(stats: list[str] = STATS) -> dict:
    results: dict = {g: {a: {"LSTM": {}, "Transformer": {}} for a in ARCHITECTURES} for g in GRAPH_NAMES}
    significance: dict = {}
    for stat in stats:
        vals, best_pe = gather(stat)
        sig = compute_significance(best_pe)
        significance[stat] = sig
        for g in GRAPH_NAMES:
            for arch in ARCHITECTURES:
                lstm_m, tr_m = ARCH_TO_METHODS[arch]
                for casting in ("nowcasting", "forecasting"):
                    for model, method in (("LSTM", lstm_m), ("Transformer", tr_m)):
                        node = results[g][arch][model].setdefault(casting, {})
                        for metric in METRICS:
                            mnode = node.setdefault(metric, {})
                            v = vals[casting][metric][g].get(method)
                            mnode[stat] = None if v is None else float(v)
                            if model == "Transformer":
                                mnode.setdefault("best_pe", {})[stat] = best_pe[casting][metric][g].get(method)
    return {
        "description": "Full per-(dataset, architecture, model, scenario, metric) statistics "
        "behind the LaTeX tables. Transformer values use best-PE per metric/stat.",
        "graphs": GRAPH_NAMES,
        "architectures": ARCHITECTURES,
        "metrics": METRICS,
        "stats": stats,
        "window_len": WINDOW_LEN,
        "forecast_horizons": list(range(1, MAX_HORIZON + 1)),
        "results": results,
        "wilcoxon_significant_p<0.05": significance,
    }


def export_results_json(path: Path | None = None, stats: list[str] = STATS) -> Path:
    path = Path(path) if path else OUT_DIR / "all_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(collect_all_results(stats), f, indent=2)
    return path


# ---------------------------------------------------------------- .tex + PDF
LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,landscape,margin=0.9cm]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[dvipsnames]{xcolor}
\usepackage{amsmath}
\usepackage{graphicx}
\pagestyle{empty}
% ============================================================================
% HOW TO COMPILE THIS FILE INTO A PDF CONTAINING ONLY THE TABLES
% ----------------------------------------------------------------------------
%   pdflatex all_stats_tables.tex          # run once (no bibliography/refs)
%   # or, equivalently / more robustly:
%   latexmk -pdf all_stats_tables.tex
%   # output: all_stats_tables.pdf (one statistic per page, tables only)
% Requires a TeX distribution (TeX Live / MiKTeX) with: booktabs, multirow,
% xcolor (dvipsnames), amsmath, graphicx. On Ubuntu:
%   sudo apt-get install texlive-latex-recommended texlive-latex-extra latexmk
% ============================================================================
\begin{document}
"""


def save_tables_to_tex(tables: dict[str, str], path: Path | None = None) -> Path:
    """Save ALL latex tables into a single compilable .tex (one table per page)."""
    path = Path(path) if path else OUT_DIR / "all_stats_tables.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\\clearpage\n".join(tables[s] for s in tables)
    path.write_text(LATEX_PREAMBLE + body + "\n\\end{document}\n")
    return path


def compile_tex_to_pdf(tex_path: Path) -> tuple[bool, str]:
    tex_path = Path(tex_path)
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
    )
    pdf = tex_path.with_suffix(".pdf")
    ok = proc.returncode == 0 and pdf.exists()
    return ok, proc.stdout[-1500:] + proc.stderr[-500:]


# ---------------------------------------------------------------- verify vs paper
_PAPER = {  # Table 3, (RMSE,MAE,NSE) nowcast | forecast
    "swiss-1990": {
        "LSTM": {
            "Isolated": ((0.782, 0.598, 0.954), (1.167, 0.909, 0.893)),
            "Graphlet": ((0.769, 0.591, 0.956), (1.160, 0.905, 0.896)),
            "Embedded": ((0.802, 0.619, 0.952), (1.165, 0.912, 0.893)),
            "ST-GNN": ((0.942, 0.721, 0.937), (1.175, 0.917, 0.891)),
        },
        "Transformer": {
            "Isolated": ((0.753, 0.573, 0.959), (1.165, 0.908, 0.896)),
            "Graphlet": ((0.746, 0.564, 0.960), (1.147, 0.894, 0.899)),
            "Embedded": ((0.776, 0.589, 0.955), (1.154, 0.903, 0.896)),
            "ST-GNN": ((0.941, 0.716, 0.936), (1.185, 0.920, 0.892)),
        },
    },
    "swiss-2010": {
        "LSTM": {
            "Isolated": ((0.789, 0.609, 0.960), (1.225, 0.957, 0.907)),
            "Graphlet": ((0.745, 0.573, 0.965), (1.211, 0.947, 0.910)),
            "Embedded": ((0.789, 0.610, 0.961), (1.248, 0.974, 0.905)),
            "ST-GNN": ((0.899, 0.684, 0.950), (1.212, 0.943, 0.909)),
        },
        "Transformer": {
            "Isolated": ((0.773, 0.598, 0.963), (1.213, 0.946, 0.910)),
            "Graphlet": ((0.725, 0.559, 0.967), (1.181, 0.922, 0.914)),
            "Embedded": ((0.779, 0.604, 0.963), (1.217, 0.952, 0.909)),
            "ST-GNN": ((0.896, 0.684, 0.951), (1.217, 0.953, 0.909)),
        },
    },
    "zurich": {
        "LSTM": {
            "Isolated": ((0.748, 0.586, 0.979), (1.292, 1.011, 0.934)),
            "Graphlet": ((0.711, 0.559, 0.981), (1.292, 1.019, 0.934)),
            "Embedded": ((0.798, 0.603, 0.977), (1.284, 0.992, 0.936)),
            "ST-GNN": ((0.924, 0.684, 0.970), (1.265, 0.984, 0.936)),
        },
        "Transformer": {
            "Isolated": ((0.722, 0.565, 0.980), (1.271, 0.997, 0.934)),
            "Graphlet": ((0.720, 0.561, 0.980), (1.243, 0.975, 0.939)),
            "Embedded": ((0.700, 0.549, 0.981), (1.275, 0.996, 0.936)),
            "ST-GNN": ((0.903, 0.669, 0.971), (1.282, 1.002, 0.935)),
        },
    },
}


def verify_mean_vs_paper() -> int:
    vals, _ = gather("Mean")
    order = [("nowcasting", m) for m in METRICS] + [("forecasting", m) for m in METRICS]
    n_ok = n_bad = n_na = 0
    print(f"{'graph':11s} {'model':12s} {'arch':9s} | nowcast R/M/N + forecast R/M/N  (CSV vs paper, 3dp)")
    for g in GRAPH_NAMES:
        for model in ("LSTM", "Transformer"):
            for arch in ARCHITECTURES:
                method = ARCH_TO_METHODS[arch][0 if model == "LSTM" else 1]
                csv6 = [vals[c][m][g].get(method) for c, m in order]
                paper6 = list(_PAPER[g][model][arch][0]) + list(_PAPER[g][model][arch][1])
                marks = []
                for cv, pv in zip(csv6, paper6):
                    if cv is None:
                        marks.append("NA")
                        n_na += 1
                    elif round(cv, 3) == round(pv, 3):
                        marks.append("=")
                        n_ok += 1
                    else:
                        marks.append(f"X({cv:.3f}!={pv:.3f})")
                        n_bad += 1
                print(f"{g:11s} {model:12s} {arch:9s} | {' '.join(marks)}")
    print(f"\nSUMMARY vs paper Table 3: match={n_ok}  mismatch={n_bad}  NA={n_na}  (total cells={n_ok + n_bad + n_na})")
    return n_bad


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--generate", action="store_true")
    a = ap.parse_args()
    if a.verify or not (a.verify or a.generate):
        bad = verify_mean_vs_paper()
        print("GATE:", "PASS" if bad == 0 else f"FAIL ({bad} mismatched)")
    if a.generate:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tables = build_all_tables()
        tex = save_tables_to_tex(tables)
        js = export_results_json()
        ok, log = compile_tex_to_pdf(tex)
        print(f"tex={tex}\njson={js}\npdf_ok={ok}")
        if not ok:
            print(log)


if __name__ == "__main__":
    main()
