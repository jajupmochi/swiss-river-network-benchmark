import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
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
        'Station': ['Mean', 'Std', 'Median', 'Max', 'Min'],
        'RMSE': [
            df['RMSE'].mean(),
            df['RMSE'].std(),
            df['RMSE'].median(),
            df['RMSE'].max(),
            df['RMSE'].min(),
        ],
        'MAE': [
            df['MAE'].mean(),
            df['MAE'].std(),
            df['MAE'].median(),
            df['MAE'].max(),
            df['MAE'].min(),
        ],
        'NSE': [
            df['NSE'].mean(),
            df['NSE'].std(),
            df['NSE'].median(),
            df['NSE'].max(),
            df['NSE'].min(),
        ],
        'N': [
            df['N'].mean(),
            df['N'].std(),
            df['N'].median(),
            df['N'].max(),
            df['N'].min(),
        ]
    }
    stats_df = pd.DataFrame(stats)
    print(f'\n{INFO_TAG}STATISTICS:')
    print(stats_df.to_string(index=False))
    # Append statistics to the original DataFrame
    df_stats = pd.concat([df, stats_df], ignore_index=True)
    return df_stats


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


def evaluate_best_trial_isolated_station(graph_name, method, station, i, output_dir: Path | None = None):
    if 'lstm' == method or 'graphlet' == method:
        analysis = experiment_analysis_isolated_station(graph_name, method, station)
    if method in ['lstm_embedding', 'stgnn', 'transformer_embedding']:
        analysis = experiment_analysis_single_model(graph_name, method, output_dir=output_dir)

    # df = analysis.dataframe()
    # Get the best Trial:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config
    # VERBOSE and print(f"Best trial: {best_trial}")
    VERBOSE and print(f'{INFO_TAG}Best Trial Configuration: {best_config}')

    # Get the best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")

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
        )

    # move
    # if 'stgnn' == method:
    #    model = SpatioTemporalEmbeddingModel(best_config['gnn_conv'], input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers'], best_config['num_convs'])
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))
    model.eval()

    # model summary
    total_params = parameter_count(model)

    if 'lstm' == method:
        return (*test_lstm(graph_name, station, model, dump_dir=DUMP_DIR), total_params)
    elif 'graphlet' == method:
        return (*test_graphlet(graph_name, station, model, dump_dir=DUMP_DIR), total_params)
    elif 'lstm_embedding' == method:
        return (*test_lstm_embedding(graph_name, station, i, model, dump_dir=DUMP_DIR), total_params)
    elif 'transformer_embedding' == method:
        return (*test_transformer_embedding(graph_name, station, i, model, dump_dir=DUMP_DIR), total_params)
    # Move
    # elif 'stgnn' == method:
    #    return test_stgnn(graph_name, )
    else:
        raise ValueError(f'Unknown method: {method}')


def process_method(graph_name, method, output_dir: Path | None = None):
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

        for i, station in enumerate(stations):
            spinner.text = f'Processing Stations: {i + 1}/{len(stations)} ->'

            if station in failed_stations:
                continue  # fix this stations!

            # Visualize station:
            # visualize_isolated_experiment(graph_name, method, station)
            # exit()

            # if True:
            try:
                rmse, mae, nse, n, params = evaluate_best_trial_isolated_station(
                    graph_name, method, station, i, output_dir=output_dir
                )
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
            except FileNotFoundError as e:
                raise
            except Exception as e:
                print(f'{ISSUE_TAG}Station {station} failed!')
                # print(e)
                # raise
                failed_stations.append(station)

        spinner.succeed(f'Processed {len(stations)} stations with {len(failed_stations)} failures.')

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
    METHODS = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn']

    # Single Run
    SINGLE_RUN = False
    if SINGLE_RUN:
        graph_name = GRAPH_NAMES[2]
        method = METHODS[3]
        process_method(graph_name, method, output_dir=OUTPUT_DIR)

    # Graph Run
    GRAPH_RUN = True
    if GRAPH_RUN:
        graph_name = GRAPH_NAMES[0]
        for m in METHODS[2:3]:  # fixme: test only [2:3}
            process_method(graph_name, m, output_dir=OUTPUT_DIR)

    # plot graphs:
    plt.show()
