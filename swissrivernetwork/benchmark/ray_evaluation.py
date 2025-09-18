import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from halo import Halo
from ray.tune import ExperimentAnalysis

from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.test_isolated_station import (
    test_lstm, test_graphlet, test_lstm_embedding, test_transformer_embedding
)
from swissrivernetwork.benchmark.test_single_model import test_stgnn
from swissrivernetwork.benchmark.train_isolated_station import read_stations, extract_neighbors, read_graph
from swissrivernetwork.gbr25.graph_exporter import plot_graph

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green

VERBOSE = False

CUR_ABS_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = (CUR_ABS_DIR / '../../' / 'swissrivernetwork/benchmark/outputs/ray_results/').resolve()
DUMP_DIR = (CUR_ABS_DIR / '../../' / 'swissrivernetwork/benchmark/dump/').resolve()
LOAD_LATEST_RESULTS = True


def compute_stats(df: pd.DataFrame):
    """
    Compute statistics (e.g., mean, median, max, min, std) for RMSE, MAE, NSE columns in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'Station', 'RMSE', 'MAE', 'NSE', and 'N'.

    returns a new DataFrame with the original data and the statistics appended at the bottom.
    """
    stats = {
        'Station': ['Mean', 'Std', 'Median', 'Min', 'Max'],
        'RMSE': [
            df['RMSE'].mean(),
            df['RMSE'].std(),
            df['RMSE'].median(),
            df['RMSE'].min(),
            df['RMSE'].max(),
        ],
        'MAE': [
            df['MAE'].mean(),
            df['MAE'].std(),
            df['MAE'].median(),
            df['MAE'].min(),
            df['MAE'].max(),
        ],
        'NSE': [
            df['NSE'].mean(),
            df['NSE'].std(),
            df['NSE'].median(),
            df['NSE'].min(),
            df['NSE'].max(),
        ],
        'N': [
            df['N'].mean(),
            df['N'].std(),
            df['N'].median(),
            df['N'].min(),
            df['N'].max(),
        ]
    }
    stats_df = pd.DataFrame(stats)
    print(f'\n{INFO_TAG}STATISTICS:')
    print(stats_df.to_string(index=False))
    # Append statistics to the original DataFrame
    df_stats = pd.concat([df, stats_df], ignore_index=True)
    return df_stats


def get_metrics_from_ray_trial(
        trial: "ray.tune.experiment.trial.Trial", anchor_metric: str, metrics_to_extract: list[str], mode: str
) -> (float, dict[str, float | None], int):
    """
    Extract specified metrics from a Ray Tune trial's Trial object with the same training steps as the best
    ``anchor_metric.``

    Args:
        trial: A Ray Tune trial object.
        anchor_metric: The metric used to identify the best trial (e.g., 'validation_mse') and the best training step.
        metrics_to_extract: List of metric names to extract from the trial's metric analysis.
        mode: indicates how to extract the anchor metric. One of ['min', 'max', 'avg', 'last', 'last-5-avg', 'last-10-avg'].

    Returns:
        A dictionary containing the extracted metrics.
    """
    best_anchor_metric = trial.metric_analysis.get(anchor_metric, {}).get(mode, None)
    if best_anchor_metric is None:
        raise ValueError(f"Anchor metric '{anchor_metric}' with mode '{mode}' not found in trial's metric analysis.")

    # Load the results from the trial's log directory directly:
    pgs_file_path = Path(trial.path) / 'progress.csv'
    if not pgs_file_path.exists():
        raise FileNotFoundError(f'Progress file not found at {pgs_file_path}.')

    df = pd.read_csv(pgs_file_path)
    if anchor_metric not in df.columns:
        raise ValueError(f'Anchor metric "{anchor_metric}" not found in progress file columns.')

    best_rows = df[np.abs(df[anchor_metric] - best_anchor_metric) < 1e-9]
    if best_rows.empty:
        raise ValueError(f'No matching rows found for best anchor metric value {best_anchor_metric}.')
    best_row = best_rows.iloc[-1]  # Take the last occurrence if multiple
    best_training_iteration = int(best_row['training_iteration'])
    extracted_metrics = {}
    for metric in metrics_to_extract:
        if metric in df.columns:
            extracted_metrics[metric] = best_row[metric]
        else:
            extracted_metrics[metric] = None

    return best_anchor_metric, extracted_metrics, best_training_iteration

    # # Unfortunately, this does not always work:
    # # Find the training step corresponding to the best anchor metric:
    # metric_n_steps = trial.metric_n_steps
    # anchor_in_steps = metric_n_steps[anchor_metric]
    # best_deque_key, best_step_in_deque = None, None
    # # Seems that the deque with the key "n" correspond to last n training steps.
    # max_deque_len = max([len(v) for v in anchor_in_steps.values()])
    # for deque_key, values in anchor_in_steps.items():
    #     if len(values) == max_deque_len:
    #         best_deque_key = deque_key
    #         best_step_in_deque = np.argwhere(np.abs(np.array(values) - best_anchor_metric) < 10e-9)[0][-1].item()
    #         break
    #
    # extracted_metrics = {}
    # for metric in metrics_to_extract:
    #     if metric in metric_n_steps:
    #         cur_best_metric = metric_n_steps[metric][best_deque_key][best_step_in_deque]
    #         extracted_metrics[metric] = cur_best_metric
    #     else:
    #         extracted_metrics[metric] = None
    #
    # best_training_iteration = metric_n_steps['training_iteration'][best_deque_key][best_step_in_deque]
    # return best_anchor_metric, extracted_metrics, best_training_iteration


def show_best_trial(best_trial):
    best_all = best_trial.get('all', None)
    best_last = best_trial.get('last', None)
    if best_all is not None:
        validation_mse, metrics, best_train_iter = get_metrics_from_ray_trial(
            best_all, anchor_metric='validation_mse',
            metrics_to_extract=['train_loss', 'validation_ave_rmse', 'validation_rmse'],
            mode='min'
        )
        print(f'\n{INFO_TAG}Best All Trial:')
        print(f'  - Trial ID: {best_all.trial_id}')
        print(f'  - Validation MSE: {validation_mse:.6f}' if validation_mse is not None else '  - Validation MSE: N/A')
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                print(f'  - {metric_name.replace("_", " ").title()}: {metric_value:.6f}')
            else:
                print(f'  - {metric_name.replace("_", " ").title()}: N/A')
        print(f'  - Best Training Iteration: {best_train_iter}')
        print(f'  - Epoch: {best_all.last_result["training_iteration"]}')
    if best_last is not None:
        validation_mse, metrics, best_train_iter = get_metrics_from_ray_trial(
            best_last, anchor_metric='validation_mse',
            metrics_to_extract=['train_loss', 'validation_ave_rmse', 'validation_rmse'],
            mode='last'
        )
        print(f'\n{INFO_TAG}Best Last Trial:')
        print(f'  - Trial ID: {best_last.trial_id}')
        print(f'  - Validation MSE: {validation_mse:.6f}' if validation_mse is not None else '  - Validation MSE: N/A')
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                print(f'  - {metric_name.replace("_", " ").title()}: {metric_value:.6f}')
            else:
                print(f'  - {metric_name.replace("_", " ").title()}: N/A')
        print(f'  - Best Training Iteration: {best_train_iter}')
        print(f'  - Epoch: {best_last.last_result["training_iteration"]}')


def plot_diff(
        graph_name, method, epoch_day_list, actuals, predictions, title,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump'
):
    epoch_days = np.sort(np.unique(np.concatenate(epoch_day_list)))
    diffs = np.empty((len(epoch_day_list), len(epoch_days)), dtype=float)
    for idx_station, (ed, a, p) in enumerate(zip(epoch_day_list, actuals, predictions)):
        day_idx_in_full = np.searchsorted(epoch_days, ed)
        diff = np.full((len(epoch_days)), np.nan, dtype=float)
        diff[day_idx_in_full] = np.abs(np.array(a) - np.array(p))
        diffs[idx_station] = diff

    mean_diff = np.nanmean(diffs, axis=0)
    std_diff = np.nanstd(diffs, axis=0)

    plt.figure(figsize=(20, 6))
    plt.plot(
        epoch_days, mean_diff, color='tab:blue', linestyle='-', linewidth=1, label='Difference'
    )
    plt.fill_between(
        epoch_days, mean_diff - std_diff, mean_diff + std_diff, color='tab:blue', alpha=0.3, label='Std Dev'
    )
    # Plot sequence separators as vertical dashed lines:
    day_diff = np.diff(epoch_days)
    breaks = day_diff != 1
    breaks = np.concatenate((np.array([True]), breaks))
    seque_id = np.cumsum(breaks)
    seques = np.argwhere(breaks).flatten()
    unique_ids, counts = np.unique(seque_id, return_counts=True)
    seque_lengths = counts
    starts = epoch_days[seques]
    ends = epoch_days[seques + seque_lengths - 1]
    colors = plt.get_cmap('tab20', len(starts))
    fill_params = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        plt.axvline(x=s, color=colors(i), linestyle='--', alpha=0.5, linewidth=0.5)
        plt.axvline(x=e, color=colors(i), linestyle='--', alpha=0.5, linewidth=0.5)
        fill_params.append({'s': s, 'e': e, 'color': colors(i), 'ylim': plt.ylim()})
    max_ylim = (min([p['ylim'][0] for p in fill_params]), max([p['ylim'][1] for p in fill_params]))
    for param in fill_params:
        plt.fill_betweenx(max_ylim, param['s'], param['e'], color=param['color'], alpha=0.1)
    plt.legend()
    plt.title(title.replace('\t', ' '))
    fig_path = Path(dump_dir) / f'figures/{graph_name}_{method}_difference.png'
    os.makedirs(fig_path.parent, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.show()
    plt.close()


# %%


def experiment_analysis_isolated_station(graph_name, method, station):
    date = None
    if 'lstm' == method and 'swiss-1990' == graph_name:
        date = '05-09_19-27-00'
    if 'lstm' == method and 'swiss-2010' == graph_name:
        date = '05-13_13-32-54'
    if 'lstm' == method and 'zurich' == graph_name:
        # date = '05-27_09-41-13'
        date = '07-24_18-48-21'
    if 'graphlet' == method and 'swiss-1990' == graph_name:
        date = '05-13_16-43-09'
    if 'graphlet' == method and 'swiss-2010' == graph_name:
        date = '05-13_16-59-26'
    if 'graphlet' == method and 'zurich' == graph_name:
        # date = '07-23_11-55-42'
        date = '07-25_09-16-32'

    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item and station in item]
    assert len(matching_items) == 1, 'Identifier is not unique'
    VERBOSE and print(f'{INFO_TAG}~~~ ANALYSIS for {method} at {station} ~~~')
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')


def experiment_analysis_single_model(graph_name, method, output_dir: Path | None = None):
    if method not in ['transformer_embedding', 'lstm_embedding', 'stgnn']:
        raise ValueError(f'Method {method} does not use single model training.')

    if graph_name not in ['swiss-1990', 'swiss-2010', 'zurich']:
        raise ValueError(f'Unknown graph name: {graph_name}.')

    directory = '/home/benjamin/ray_results' if output_dir is None else output_dir

    if LOAD_LATEST_RESULTS:
        path_prefix = f'{method}-{graph_name}-'
        all_paths = sorted(
            [path for path in directory.iterdir() if
             path.is_dir() and path.name.startswith(path_prefix)]
        )
        assert len(all_paths) > 0, f'No previous results found for {method} on {graph_name}.'
        latest_path = all_paths[-1]
        VERBOSE and print(f'{INFO_TAG}Loading latest results from {latest_path}.')
        VERBOSE and print(f'~~~ ANALYSIS for {method} ~~~')
        return ExperimentAnalysis(str(latest_path))

    else:
        date = None
        if 'lstm_embedding' == method and 'swiss-1990' == graph_name:
            date = '05-20_11-35-12'
        if 'lstm_embedding' == method and 'swiss-2010' == graph_name:
            date = '05-20_17-16-23'
        if 'lstm_embedding' == method and 'zurich' == graph_name:
            # date = '05-27_16-43-27'
            date = '07-23_14-51-42'  # updated stations

        if 'stgnn' == method and 'swiss-1990' == graph_name:
            date = 'stgnn-2025-06-16_16-11-22'
        if 'stgnn' == method and 'swiss-2010' == graph_name:
            # date = 'stgnn-2025-06-17_18-58-54'
            date = 'stgnn-2025-06-18_18-48-43'
        if 'stgnn' == method and 'zurich' == graph_name:
            date = 'stgnn-2025-07-24_09-41-05'

        if date is None:
            raise ValueError(
                'Date for transformer_embedding not specified. '
                'Please enable LOAD_LATEST_RESULTS in script `ray_evaluation` to load the latest results.'
            )

        matching_items = [item for item in os.listdir(directory) if date in item and method in item]
        assert len(matching_items) == 1, 'Identifier is not unique'
        VERBOSE and print(f'~~~ ANALYSIS for {method} ~~~')
        return ExperimentAnalysis(f'{directory}/{matching_items[0]}')


def parameter_count(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print('TOTAL MODEL PARAMETERS: ', pytorch_total_params)
    return pytorch_total_params


def evaluate_best_trial_single_model(graph_name, method, output_dir: Path | None = None):
    if 'stgnn' == method:
        analysis = experiment_analysis_single_model(graph_name, method, output_dir=output_dir)

    # Get Best Trial:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")
    VERBOSE and print(f'Best Trial Configuration: {best_config}')

    # Prepare Model
    if 'stgnn' == method:
        input_size = 1
        num_embeddings = len(read_stations(graph_name))
        model = SpatioTemporalEmbeddingModel(
            best_config['gnn_conv'], 1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'],
            best_config['num_layers'], best_config['num_convs']
        )

    # Load Model
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))
    model.eval()

    # Model summary
    total_params = parameter_count(model)

    if 'stgnn' == method:
        return (*test_stgnn(graph_name, model, dump_dir=DUMP_DIR), total_params)


def evaluate_best_trial_isolated_station(
        graph_name, method, station, i, output_dir: Path | None = None, settings: dict = {}
):
    if 'lstm' == method or 'graphlet' == method:
        analysis = experiment_analysis_isolated_station(graph_name, method, station)
    if method in ['lstm_embedding', 'stgnn', 'transformer_embedding']:
        analysis = experiment_analysis_single_model(graph_name, method, output_dir=output_dir)

    # df = analysis.dataframe()
    # Get the best Trial:
    best_trial_all = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_trial_last = analysis.get_best_trial(metric="validation_mse", mode="min", scope="last")
    best_config = best_trial_all.config
    # VERBOSE and print(f"Best trial: {best_trial}")
    VERBOSE and print(f'{INFO_TAG}Best Trial Configuration: {best_config}')

    # Get the best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(best_trial_all, metric="validation_mse", mode="min")

    # Determine input_size:
    if method in ['lstm', 'lstm_embedding', 'transformer_embedding']:
        input_size = 1
    if 'graphlet' == method:
        input_size = 1 + len(extract_neighbors(graph_name, station, 1))

    if method in ['lstm_embedding', 'transformer_embedding']:
        num_embeddings = len(read_stations(graph_name))
        embedding_size = best_config['embedding_size']

    # Create Model:        
    if 'lstm' == method or 'graphlet' == method:
        model = LstmModel(input_size, best_config['hidden_size'], best_config['num_layers'])
    if 'lstm_embedding' == method:
        model = LstmEmbeddingModel(
            input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers']
        )
    if 'transformer_embedding' == method:
        model = TransformerEmbeddingModel(
            input_size, num_embeddings=num_embeddings if best_config['use_station_embedding'] else 0,
            embedding_size=best_config['embedding_size'],
            num_heads=best_config['num_heads'],
            num_layers=best_config['num_layers'],
            dim_feedforward=best_config['dim_feedforward'],
            dropout=best_config['dropout'],
            d_model=best_config['d_model'] if best_config.get('d_model', None) else int(
                best_config['ratio_heads_to_d_model'] * best_config['num_heads']
            ),
            missing_value_method=settings.get('missing_value_method', None)
        )

    # move
    # if 'stgnn' == method:
    #    model = SpatioTemporalEmbeddingModel(best_config['gnn_conv'], input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers'], best_config['num_convs'])
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))
    model.eval()

    # model summary
    total_params = parameter_count(model)

    best_trial = {'all': best_trial_all, 'last': best_trial_last}
    window_len = settings['window_len'] if 'window_len' in settings else best_config.get('window_len', None)

    if 'lstm' == method:
        return (*test_lstm(graph_name, station, model, dump_dir=DUMP_DIR), total_params, best_trial)
    elif 'graphlet' == method:
        return (*test_graphlet(graph_name, station, model, dump_dir=DUMP_DIR), total_params, best_trial)
    elif 'lstm_embedding' == method:
        return (*test_lstm_embedding(
            graph_name, station, i, model,
            window_len=window_len,
            dump_dir=DUMP_DIR
        ), total_params, best_trial)
    elif 'transformer_embedding' == method:
        return (*test_transformer_embedding(
            graph_name, station, i, model,
            window_len=window_len,
            dump_dir=DUMP_DIR
        ), total_params, best_trial)
    # Move
    # elif 'stgnn' == method:
    #    return test_stgnn(graph_name, )
    else:
        raise ValueError(f'Unknown method: {method}')


def process_method(graph_name, method, output_dir: Path | None = None, settings: dict = {}):
    print(f'~~~ Process {method} on {graph_name} ~~~')

    failed_stations = []

    col_station = []
    col_rmse = []
    col_mae = []
    col_nse = []
    col_n = []

    # Setup
    stations = read_stations(graph_name)
    # statistics:
    print(f'{INFO_TAG}Expected Stations: ', len(stations))

    # visualize_all_isolated_experiments(graph_name, method)
    # exit()

    total_params = 0

    if method in ['lstm', 'graphlet', 'lstm_embedding', 'transformer_embedding']:
        spinner = Halo(
            text='Processing Stations ->', spinner='pong', color='cyan', text_color='grey',
            animation='bounce'
        )
        spinner.start()

        actuals = []
        predictions = []
        epoch_day_list = []

        for i, station in enumerate(stations):
            spinner.text = f'Processing Stations: {i + 1}/{len(stations)} ->'

            if station in failed_stations:
                continue  # fix this stations!

            # Visualize station:
            # visualize_isolated_experiment(graph_name, method, station)
            # exit()

            # if True:
            try:
                rmse, mae, nse, n, preds, params, best_trial = evaluate_best_trial_isolated_station(
                    graph_name, method, station, i, output_dir=output_dir, settings=settings
                )
                actual, prediction, epoch_days = preds
                if math.isnan(rmse):
                    failed_stations.append(station)
                    continue

                if method in ['lstm_embedding', 'transformer_embedding']:
                    total_params = params
                else:
                    total_params += params  # lstm and graphlets use different models per station.

                col_station.append(station)
                col_rmse.append(rmse)
                col_mae.append(mae)
                col_nse.append(nse)
                col_n.append(n)

                actuals.append(actual)
                predictions.append(prediction)
                epoch_day_list.append(epoch_days)
            except FileNotFoundError as e:
                raise
            except Exception as e:
                print(f'{ISSUE_TAG}Station {station} failed!')
                print(e)
                raise
                failed_stations.append(station)

        spinner.succeed(f'Processed {len(stations)} stations with {len(failed_stations)} failures.')

        plot_diff(
            graph_name, method, epoch_day_list, actuals, predictions, 'Difference on all stations',
            dump_dir=DUMP_DIR
        )  # fixme: debug

    if 'stgnn' == method:
        # run model:
        rmses, maes, nses, ns, total_params = evaluate_best_trial_single_model(
            graph_name, method, output_dir=output_dir
        )

        # collect per station
        for i, station in enumerate(stations):
            col_station.append(station)
            col_rmse.append(rmses[i])
            col_mae.append(maes[i])
            col_nse.append(nses[i])
            col_n.append(ns[i])

    print('METHOD LEARNABLE PARAMETERS:', total_params)

    print('FAILED_STATIONS:', failed_stations)

    df = pd.DataFrame(
        data={
            'Station': col_station,
            'RMSE': col_rmse,
            'MAE': col_mae,
            'NSE': col_nse,
            'N': col_n
        }
    )

    df = compute_stats(df)

    show_best_trial(best_trial)

    test_dir = DUMP_DIR / 'test_results/'
    test_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(test_dir / f'{graph_name}_{method}.csv', index=False)

    plt.close('all')
    x, e = read_graph(graph_name)
    information = dict()
    color = dict()
    for station, rmse in zip(col_station, col_rmse):
        information[station] = f'{station} - (RMSE={rmse:.3f})'
        color[station] = rmse

    # For zurich:
    plt.figure(figsize=(16, 10), layout='tight')

    plot_graph(x, e, information=information, color=color, vmin=0.5, vmax=1.5)
    plt.savefig(test_dir / f'figure_{graph_name}_{method}.png', dpi=150)


if __name__ == '__main__':

    GRAPH_NAMES = ['swiss-1990', 'swiss-2010', 'zurich']
    METHODS = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn', 'transformer_embedding']

    window_len_map = {  # fixme: experiment
        'lstm': None,
        'graphlet': None,
        'lstm_embedding': 90,
        'stgnn': None,
        'transformer_embedding': 90
    }

    # Single Run
    SINGLE_RUN = False
    if SINGLE_RUN:
        graph_name = GRAPH_NAMES[2]
        method = METHODS[3]
        process_method(
            graph_name, method, output_dir=OUTPUT_DIR, settings={
                'window_len': window_len_map[method]
            }
        )

    # Graph Run
    GRAPH_RUN = True
    if GRAPH_RUN:
        graph_name = GRAPH_NAMES[2]
        for m in METHODS[2:3]:  # fixme: test only [2:3}
            process_method(
                graph_name, m, output_dir=OUTPUT_DIR, settings={
                    'window_len': window_len_map[m]
                }
            )

    # plot graphs:
    plt.show()
