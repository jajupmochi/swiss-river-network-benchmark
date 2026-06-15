"""Diagnostic + plots: train/valid generalization gap in dumped LSTM/Transformer predictions.

Graphlet models consume `{graph}_{method}_{station}_train.csv` as neighbor features.
That CSV covers the **entire** train CSV — both the portion the base model did
gradient updates on (first `train_split` rows) and the held-out valid portion
(remaining rows). If the RMSE on the first portion is much lower than on the second,
the graphlet's training features are systematically easier than its test features.

Usage:

    uv run python swissrivernetwork/benchmark/visualize_results/diagnose_graphlet_leakage.py \\
        --graph-name swiss-1990 --station 2091 --method lstm \\
        --path-extra-keys=-fs7-future_embedding-wl90-none --also-neighbors

(use the `--path-extra-keys=...` form because the value starts with `-`).

Run the corresponding LSTM evaluation first so the `_train.csv` exists under
`swissrivernetwork/benchmark/dump/predictions/<path-extra-keys>/`.

Plots are written to `swissrivernetwork/benchmark/visualize_results/figures/` (gitignored).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from swissrivernetwork.benchmark.dataset import (
    PROJ_DIR,
    read_csv_train,
    select_isolated_station,
)
from swissrivernetwork.benchmark.util import extract_neighbors

FIG_DIR = Path(__file__).parent / "figures"


def _rmse_mae(err: np.ndarray) -> tuple[float, float]:
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def compute_portion_errors(
    graph_name: str,
    station: str,
    method: str,
    predict_dump_dir: Path,
    train_split: float,
) -> dict | None:
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)
    df = df.sort_values("epoch_day").reset_index(drop=True)

    # Match fit_normalizers / normalize_isolated_station: MinMaxScaler on water_temperature.
    normalizer_wt = MinMaxScaler().fit(df["water_temperature"].values.reshape(-1, 1))
    df["wt_norm"] = normalizer_wt.transform(df["water_temperature"].values.reshape(-1, 1)).flatten()

    pred_path = predict_dump_dir / f"{graph_name}_{method}_{station}_train.csv"
    if not pred_path.is_file():
        return {"error": f"missing dump file: {pred_path}"}
    df_pred = pd.read_csv(pred_path)

    pred_col = f"{station}_wt_hat"
    if pred_col not in df_pred.columns:
        return {
            "error": f"column {pred_col!r} missing in {pred_path.name}; "
            f"available={[c for c in df_pred.columns if c != 'epoch_day']}"
        }

    # 80/20 boundary is index-based on the full train CSV for this station, same as
    # dataset.train_valid_split. Convert to an epoch_day cutoff so we can apply it
    # to the prediction file.
    train_size = int(train_split * len(df))
    if train_size <= 0 or train_size >= len(df):
        return {"error": f"train_split={train_split} yields empty portion (len(df)={len(df)})"}
    cutoff_epoch_day = int(df.iloc[train_size - 1]["epoch_day"])

    # Inner merge: only days the model produced a prediction for and we have a
    # ground-truth wt. Drop NaN wt (missing observations).
    merged = df[["epoch_day", "wt_norm"]].merge(df_pred[["epoch_day", pred_col]], on="epoch_day", how="inner")
    merged = merged.dropna(subset=["wt_norm", pred_col])
    if merged.empty:
        return {"error": "no overlapping days with valid wt after merge"}

    train_part = merged[merged["epoch_day"] <= cutoff_epoch_day]
    valid_part = merged[merged["epoch_day"] > cutoff_epoch_day]

    def stats(part: pd.DataFrame) -> dict | None:
        if part.empty:
            return None
        err = part[pred_col].to_numpy() - part["wt_norm"].to_numpy()
        rmse, mae = _rmse_mae(err)
        return {
            "n_days": int(len(part)),
            "epoch_day_min": int(part["epoch_day"].min()),
            "epoch_day_max": int(part["epoch_day"].max()),
            "rmse_norm": rmse,
            "mae_norm": mae,
        }

    return {
        "cutoff_epoch_day": cutoff_epoch_day,
        "train_portion": stats(train_part),
        "valid_portion": stats(valid_part),
        "merged": merged.rename(columns={pred_col: "pred_norm"}),
    }


def _format_portion(label: str, s: dict | None) -> str:
    if s is None:
        return f"  {label:40s}  (empty)"
    return (
        f"  {label:40s}  n={s['n_days']:6d}  "
        f"epoch_day=[{s['epoch_day_min']}, {s['epoch_day_max']}]  "
        f"RMSE(norm)={s['rmse_norm']:.4f}  MAE(norm)={s['mae_norm']:.4f}"
    )


def _verdict(ratio: float) -> str:
    if ratio > 2.0:
        return "STRONG leakage signal"
    if ratio > 1.3:
        return "mild gap"
    return "no obvious gap"


def _plot_ratio_bars(summary: list[dict], out_path: Path) -> None:
    by_graph: dict[str, list[dict]] = {}
    for row in summary:
        by_graph.setdefault(row["graph_name"], []).append(row)

    n = len(by_graph)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 0.25 * max(len(v) for v in by_graph.values())), 3 * n + 1))
    if n == 1:
        axes = [axes]
    for ax, (graph, rows) in zip(axes, by_graph.items()):
        rows = sorted(rows, key=lambda r: r["ratio"] if np.isfinite(r["ratio"]) else -np.inf)
        stations = [r["station"] for r in rows]
        ratios = [r["ratio"] for r in rows]
        colors = ["#d62728" if r > 2.0 else ("#ff7f0e" if r > 1.3 else "#2ca02c") for r in ratios]
        ax.bar(range(len(stations)), ratios, color=colors)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="ratio=1 (no gap)")
        ax.axhline(1.3, color="#ff7f0e", linestyle=":", linewidth=0.8, label="mild (1.3)")
        ax.axhline(2.0, color="#d62728", linestyle=":", linewidth=0.8, label="strong (2.0)")
        ax.set_xticks(range(len(stations)))
        ax.set_xticklabels(stations, rotation=90, fontsize=7)
        ax.set_ylabel("valid RMSE / train RMSE")
        ax.set_title(f"{graph} — per-station generalization gap ({rows[0]['method']})")
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_time_series_grid(
    graph_name: str,
    method: str,
    per_station: dict[str, dict],
    out_path: Path,
    max_stations: int = 12,
) -> None:
    items = list(per_station.items())[:max_stations]
    if not items:
        return
    ncols = 2
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=False)
    axes = np.atleast_2d(axes).ravel()
    for ax, (station, result) in zip(axes, items):
        merged = result["merged"].sort_values("epoch_day")
        cutoff = result["cutoff_epoch_day"]
        abs_err = np.abs(merged["pred_norm"].to_numpy() - merged["wt_norm"].to_numpy())
        ax.plot(merged["epoch_day"], abs_err, linewidth=0.5, color="#1f77b4", alpha=0.7, label="|pred - actual|")
        w = max(1, len(merged) // 60)
        rolling = pd.Series(abs_err).rolling(w, min_periods=1).mean()
        ax.plot(merged["epoch_day"], rolling, color="#d62728", linewidth=1.0, label=f"rolling mean (w={w})")
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=0.8, label=f"80/20 cutoff")
        tp, vp = result["train_portion"], result["valid_portion"]
        ratio = (vp["rmse_norm"] / tp["rmse_norm"]) if (tp and vp and tp["rmse_norm"] > 0) else float("nan")
        ax.set_title(
            f"station {station} — train RMSE={tp['rmse_norm']:.3f}, valid RMSE={vp['rmse_norm']:.3f} (ratio={ratio:.2f})",
            fontsize=9,
        )
        ax.set_ylabel("|err| (norm)")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(f"{graph_name} — abs. prediction error over train CSV ({method})", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_graph(
    graph_name: str,
    stations: list[str],
    method: str,
    predict_dump_dir: Path,
    train_split: float,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, dict]]:
    summary: list[dict] = []
    per_station: dict[str, dict] = {}
    for s in stations:
        result = compute_portion_errors(graph_name, s, method, predict_dump_dir, train_split)
        if "error" in result:
            if verbose:
                print(f"  {graph_name}/{s}: SKIP — {result['error']}")
            continue
        tp, vp = result["train_portion"], result["valid_portion"]
        if not (tp and vp and tp["rmse_norm"] > 0):
            if verbose:
                print(f"  {graph_name}/{s}: SKIP — empty portion or zero RMSE")
            continue
        ratio = vp["rmse_norm"] / tp["rmse_norm"]
        summary.append(
            {
                "graph_name": graph_name,
                "station": s,
                "method": method,
                "train_rmse_norm": tp["rmse_norm"],
                "valid_rmse_norm": vp["rmse_norm"],
                "train_mae_norm": tp["mae_norm"],
                "valid_mae_norm": vp["mae_norm"],
                "ratio": ratio,
                "n_train": tp["n_days"],
                "n_valid": vp["n_days"],
            }
        )
        per_station[s] = result
        if verbose:
            print(
                f"  {graph_name}/{s}: train RMSE={tp['rmse_norm']:.4f} "
                f"valid RMSE={vp['rmse_norm']:.4f} ratio={ratio:.2f} ({_verdict(ratio)})"
            )
    return summary, per_station


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--graph-name",
        action="append",
        help="e.g. swiss-1990; pass multiple times for multiple datasets. "
        "If omitted, defaults to all three: swiss-1990, swiss-2010, zurich.",
    )
    parser.add_argument(
        "--station",
        help="single target station; if omitted, all stations in the graph are scanned",
    )
    parser.add_argument("--method", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument(
        "--path-extra-keys",
        default="",
        help="subdir under swissrivernetwork/benchmark/dump/predictions/, e.g. -fs7-future_embedding-wl90-none",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="same as config['train_split']")
    parser.add_argument(
        "--also-neighbors",
        action="store_true",
        help="with --station, also check 1-hop neighbors (ignored when --station is omitted)",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip writing plots under figures/, print summary only")
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="optional path to write a per-station summary CSV; defaults to figures/<tag>_summary.csv",
    )
    args = parser.parse_args()

    graphs = args.graph_name or ["swiss-1990", "swiss-2010", "zurich"]

    predict_dump_dir = (
        Path(PROJ_DIR) / "swissrivernetwork" / "benchmark" / "dump" / "predictions" / args.path_extra_keys
    )
    print(f"predict_dump_dir: {predict_dump_dir}")
    if not predict_dump_dir.is_dir():
        siblings = sorted(p.name for p in predict_dump_dir.parent.iterdir()) if predict_dump_dir.parent.is_dir() else []
        print(f"  directory does not exist. siblings under {predict_dump_dir.parent}: {siblings}")
        return

    from swissrivernetwork.benchmark.dataset import read_stations

    tag = (args.path_extra_keys or "root").strip("-") or "root"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict] = []
    for graph in graphs:
        print(f"\n========== {graph} ==========")
        try:
            if args.station is not None:
                stations = [args.station] + (extract_neighbors(graph, args.station, 1) if args.also_neighbors else [])
            else:
                stations = read_stations(graph)
        except Exception as exc:
            print(f"  SKIP — cannot list stations: {exc}")
            continue
        print(f"  {len(stations)} stations")

        summary, per_station = analyze_graph(
            graph, stations, args.method, predict_dump_dir, args.train_split, verbose=True
        )
        if not summary:
            print(f"  no usable stations for {graph}")
            continue
        all_summary.extend(summary)

        if not args.no_plots:
            ts_path = FIG_DIR / f"leakage_{tag}_{graph}_{args.method}_timeseries.png"
            _plot_time_series_grid(graph, args.method, per_station, ts_path, max_stations=12)
            print(f"  wrote {ts_path}")

    if not all_summary:
        print("\nnothing to summarize.")
        return

    df_summary = pd.DataFrame(all_summary)
    csv_path = Path(args.summary_csv) if args.summary_csv else FIG_DIR / f"leakage_{tag}_{args.method}_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nwrote summary: {csv_path}")

    print("\n--- aggregate per graph ---")
    for graph, grp in df_summary.groupby("graph_name"):
        print(
            f"  {graph}: n={len(grp)}  median ratio={grp['ratio'].median():.2f}  "
            f"max ratio={grp['ratio'].max():.2f}  "
            f">2x: {(grp['ratio'] > 2.0).sum()}  >1.3x: {(grp['ratio'] > 1.3).sum()}"
        )

    if not args.no_plots:
        bar_path = FIG_DIR / f"leakage_{tag}_{args.method}_ratios.png"
        _plot_ratio_bars(all_summary, bar_path)
        print(f"\nwrote {bar_path}")


if __name__ == "__main__":
    main()
