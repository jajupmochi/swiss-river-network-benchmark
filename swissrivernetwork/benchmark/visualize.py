"""Parallel-coordinate plots of isolated-station hyperparameter tuning runs.

Reads Ray Tune results per station via
:func:`~swissrivernetwork.benchmark.ray_evaluation.experiment_analysis_isolated_station`
and plots normalized hyperparameters colored by validation MSE.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from .dataset import *
from .ray_evaluation import experiment_analysis_isolated_station


def visualize_all_isolated_experiments(graph_name, method):
    """Plot the best-trial hyperparameters across all stations as parallel coordinates.

    For each station, pull its Ray Tune runs, bin trials by validation MSE, keep only the
    single best trial (normalized hyperparameters), then overlay every station's best trial
    on one MSE-colored parallel-coordinates plot.
    """
    # Create dataframes and combine them:
    cols_to_plot = ["config/batch_size", "config/learning_rate", "config/hidden_size", "config/num_layers"]
    target_col = "validation_mse"
    dfs = []
    for station in read_stations(graph_name):
        analysis = experiment_analysis_isolated_station(graph_name, method, station)
        df = analysis.dataframe(metric="validation_mse", mode="min")
        df = df[cols_to_plot + [target_col]].dropna()

        # Bin into top and others based on performance
        top_threshold = df["validation_mse"].quantile(0.05)
        flop_threshold = df["validation_mse"].quantile(0.9)
        df["performance"] = [
            "top" if mse <= top_threshold else "flop" if mse >= flop_threshold else "rest"
            for mse in df["validation_mse"]
        ]

        df.loc[df["validation_mse"] == df["validation_mse"].min(), "performance"] = "best"

        # Normalize hyperparameters to [0, 1] for plotting clarity (optional but helpful)
        normalized_df = df.copy()
        for col in cols_to_plot:
            min_val = df[col].min()
            max_val = df[col].max()
            normalized_df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

        # Select Top performaners:
        normalized_df = normalized_df[normalized_df["performance"] == "best"]
        dfs.append(normalized_df)

    df = pd.concat(dfs, ignore_index=True)

    # validate:
    print("Average MSE", df["validation_mse"].mean())

    norm = Normalize(vmin=0.3, vmax=1.5)
    cmap = cm.viridis_r

    plt.figure(figsize=(14, 6))
    x = list(range(len(cols_to_plot)))
    for _, row in df.iterrows():
        y = [row[col] for col in cols_to_plot]
        color = cmap(norm(row[target_col]))
        plt.plot(x, y, color=color, alpha=0.3)
    plt.xticks(x, cols_to_plot, rotation=45)
    # plt.yticks[0, 0.5, 1.0]
    plt.ylabel("Normalized Hyperparameter")
    plt.title("Validation MSE")
    plt.grid(True, axis="y")
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm)
    # cbar.set_label('Validation MSE')
    plt.tight_layout()
    plt.show()


def visualize_isolated_experiment(graph_name, method, station):
    """Plot the top-decile trials for one station as MSE-colored parallel coordinates."""
    analysis = experiment_analysis_isolated_station(graph_name, method, station)

    df = analysis.dataframe(metric="validation_mse", mode="min")

    # Select the hyperparameters and performance metric
    cols_to_plot = ["config/batch_size", "config/learning_rate", "config/hidden_size", "config/num_layers"]
    target_col = "validation_mse"
    df = df[cols_to_plot + [target_col]].dropna()

    # Bin into top and others based on performance
    top_threshold = df["validation_mse"].quantile(0.1)
    flop_threshold = df["validation_mse"].quantile(0.9)
    df["performance"] = [
        "top" if mse <= top_threshold else "flop" if mse >= flop_threshold else "rest" for mse in df["validation_mse"]
    ]

    # Normalize hyperparameters to [0, 1] for plotting clarity (optional but helpful)
    normalized_df = df.copy()
    for col in cols_to_plot:
        min_val = df[col].min()
        max_val = df[col].max()
        normalized_df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

    norm = Normalize(vmin=0.3, vmax=1.5)
    cmap = cm.viridis_r

    plt.figure(figsize=(14, 6))
    x = list(range(len(cols_to_plot)))
    for _, row in normalized_df[normalized_df["performance"] == "top"].iterrows():
        y = [row[col] for col in cols_to_plot]
        color = cmap(norm(row[target_col]))
        plt.plot(x, y, color=color, alpha=0.7)
    plt.xticks(x, cols_to_plot, rotation=45)
    # plt.yticks[0, 0.5, 1.0]
    plt.ylabel("Normalized Hyperparameter")
    plt.title("Validation MSE")
    plt.grid(True, axis="y")
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm)
    # cbar.set_label('Validation MSE')
    plt.tight_layout()
    plt.show()
