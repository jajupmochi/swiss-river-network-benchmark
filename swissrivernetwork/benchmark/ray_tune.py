import argparse
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import ray
from benedict import benedict
from ray.tune import uniform, randint, run, choice, Callback, sample_from
from ray.tune.schedulers import ASHAScheduler

from swissrivernetwork.benchmark.dataset import read_stations
from swissrivernetwork.benchmark.train_isolated_station import (
    train_lstm, train_graphlet, train_transformer, train_transformer_graphlet
)
from swissrivernetwork.benchmark.train_single_model import (
    train_lstm_embedding, train_stgnn, train_transformer_embedding, train_masked_transformer_embedding,
    train_transformer_stgnn
)
from swissrivernetwork.benchmark.util import get_run_name, is_transformer_model, str2bool, get_run_extra_key

CUR_ABS_DIR = Path(__file__).resolve().parent

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green

'''
Run the Ray Tuner to determine best architectures
'''


class MemoryLimitCallback(Callback):
    def __init__(self, mem_limit_mb):
        self.mem_limit_mb = mem_limit_mb


    def on_trial_result(self, iteration, trials, trial, result, **info):
        import psutil
        process = psutil.Process(trial.runner.pid)
        mem = process.memory_info().rss / (1024 ** 2)
        if mem > self.mem_limit_mb:
            print(f"Stopping trial {trial.trial_id} due to high memory usage: {mem} MB")
            trial.stop()


# Use actual working dir
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

search_space_lstm = {
    "batch_size": randint(32, 256 + 1),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.0001, 0.01),
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    "hidden_size": randint(16, 128 + 1),  # 128
    "num_layers": randint(1, 3 + 1)  # more layers!
}

search_space_transformer = {
    "batch_size": randint(32, 256 + 1),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.0001, 0.01),
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    # "hidden_size": randint(16, 128 + 1),  # 128  determined by num_t_heads * ratio_heads_to_d_model
    "num_layers": randint(1, 4 + 1),  # more layers!
    "dropout": uniform(0.0, 0.5),
    "num_t_heads": randint(2, 8 + 1),
    "ratio_heads_to_d_model": choice([4, 8, 16]),  # d_model must be multiple of num_t_heads
    # "d_model": randint(16, 256 + 1),  # 128
    "dim_feedforward": randint(32, 256 + 1),
    # "use_station_embedding": False  # Not used. Hard coded for both training and evaluation.
}

search_space_lstm_embedding = {
    "batch_size": randint(32, 256 + 1),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01),  # 10x less
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    "embedding_size": randint(1, 30 + 1),
    "hidden_size": randint(16, 128 + 1),  # 128
    "num_layers": randint(1, 3 + 1)  # more layers!
}

search_space_transformer_embedding = {
    "batch_size": randint(32, 256 + 1),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01),  # 10x less
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    "embedding_size": randint(1, 30 + 1),
    # "hidden_size": randint(16, 128 + 1),  # 128  determined by num_t_heads * ratio_heads_to_d_model
    "num_layers": randint(1, 4 + 1),  # more layers!
    "dropout": uniform(0.0, 0.5),
    "num_t_heads": randint(2, 8 + 1),
    "ratio_heads_to_d_model": choice([4, 8, 16]),  # d_model must be multiple of num_t_heads
    # "d_model": randint(16, 256 + 1),  # 128
    "dim_feedforward": randint(32, 256 + 1),
    "use_station_embedding": True
}

search_space_stgnn = {
    "batch_size": randint(1, 5),  # Batch size is times |Nodes| 30-50 bigger  (orig: 1 - 10)
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01),  # 10x less
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    "embedding_size": randint(1, 30 + 1),
    "hidden_size": randint(16, 128 + 1),  # 128
    "num_layers": randint(1, 3 + 1),  # more layers!
    # "gnn_conv": choice(['GCN', 'GIN']),
    "gnn_conv": choice(['GCN', 'GIN', 'GraphSAGE', 'MPNN', 'GAT']),
    # "gnn_conv": choice(['GraphSAGE']),
    "num_convs": randint(1, 6 + 1),
    "num_heads": sample_from(lambda spec: randint(1, 8 + 1).sample() if spec['gnn_conv'] == 'GAT' else 0),
    "edge_hidden_size": sample_from(lambda spec: randint(4, 32 + 1).sample() if spec['gnn_conv'] == 'MPNN' else None),
    # "use_station_embedding": True  # Not used. Hard coded for both training and evaluation.
}

search_space_transformer_stgnn = {
    "batch_size": randint(1, 5),  # Batch size is times |Nodes| 30-50 bigger  (orig: 1 - 10)
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01),  # 10x less
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    "embedding_size": randint(1, 30 + 1),
    # "hidden_size": randint(16, 128 + 1),  # 128  determined by num_t_heads * ratio_heads_to_d_model
    "num_layers": randint(1, 3 + 1),  # more layers!
    # "gnn_conv": choice(['GCN', 'GIN']),
    "gnn_conv": choice(['GCN', 'GIN', 'GraphSAGE', 'MPNN', 'GAT']),
    # "gnn_conv": choice(['GraphSAGE']),
    "num_convs": randint(1, 6 + 1),
    "num_heads": sample_from(lambda spec: randint(1, 8 + 1).sample() if spec['gnn_conv'] == 'GAT' else 0),
    "edge_hidden_size": sample_from(lambda spec: randint(4, 32 + 1).sample() if spec['gnn_conv'] == 'MPNN' else None),
    # Transformer specific:
    "dropout": uniform(0.0, 0.5),
    "num_t_heads": randint(1, 8 + 1),
    "ratio_heads_to_d_model": choice([4, 8, 12]),  # d_model must be multiple of num_t_heads
    # "d_model": randint(16, 256 + 1),  # 128
    "dim_feedforward": randint(32, 256 + 1),
    "use_station_embedding": True  # Not used. Hard coded for both training and evaluation.
}


def scheduler():
    return ASHAScheduler(
        max_t=200,  # 100
        grace_period=3,
        reduction_factor=2
    )


# this one is a bit less "aggressive"
def scheduler_soft():
    return ASHAScheduler(
        max_t=200,  # 100
        grace_period=5,  # 5
        reduction_factor=1.5
    )


def scheduler_single_model_soft():
    return ASHAScheduler(
        max_t=500,  # 100
        grace_period=5,
        reduction_factor=1.5
    )


def scheduler_single_model_hard():
    return ASHAScheduler(
        max_t=500,
        grace_period=3,
        reduction_factor=2
    )


def run_experiment(
        method, graph_name, num_samples, storage_path: str | None, config: benedict, verbose: int,
        if_trim_checkpoints: bool = True
):
    """
    Run a Ray Tune experiment for the given method and graph.

    Args:
        method (str): The method to use ('lstm', 'graphlet', 'lstm_embedding', 'stgnn').
        graph_name (str): The name of the graph to use.
        num_samples (int): The number of hyperparameter combinations to sample.
        storage_path (str | None): The path to store results relative to the repository root. If None, defaults to './ray_results'.
        verbose (int): The verbosity level.
    """
    # Each experiment has one time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Use same time for all!
    # now = '2025-09-15_14-53-11'  # debug only
    storage_path = (CUR_ABS_DIR / '../../' / storage_path).resolve() if storage_path else None
    # For transformer_embedding, GPU per epoch is around 50 seconds, CPU around 14-40 minutes
    # Using GPU is much faster. Notice when only one GPU is present, set the `resources_per_trial['gpu']` to a float
    # between 0 and 1 to enable multiple trials on GPU, otherwise it seems that Ray Tune just simply runs each trial
    # until the end.
    resources_per_trial = {
        'cpu': config.get('resources_per_trial.cpu', 1),
        'gpu': config.get('resources_per_trial.gpu', 0)
    }  # debug  transformer_embedding: {'cpu': 1, 'gpu': 0.25}, others: {'cpu': 1, 'gpu': 0}

    num_cpus = None if ('num_cpus' not in config or config.num_cpus is None) else (
        config.num_cpus if config.num_cpus > 0 else os.cpu_count() + config.num_cpus  # todo: check this for clusters
    )
    ray.init(num_cpus=num_cpus)
    print(f'{INFO_TAG}Ray initialized with {num_cpus} CPUs.')
    print(f'{INFO_TAG}Cluster resources detected by Ray: {ray.cluster_resources()}.')
    print(f'{INFO_TAG}Available / Idle resources detected by Ray: {ray.available_resources()}.')

    run_name = get_run_name(method, graph_name, now, config, directory=storage_path)

    # update search space (!)
    if 'lstm' == method or 'graphlet' == method:

        for station in read_stations(graph_name):

            search_space = search_space_lstm.copy()
            search_space['station'] = station
            search_space['graph_name'] = graph_name
            # --- Dataset specific:
            # 'mask_embedding' or 'interpolation' or 'zero' or None:
            search_space['missing_value_method'] = config.missing_value_method
            search_space['short_subsequence_method'] = config.short_subsequence_method  # 'pad' or 'drop'
            search_space['max_mask_consecutive'] = config.max_mask_consecutive  # only used
            search_space['max_mask_ratio'] = config.max_mask_ratio
            # --- For all models:
            search_space['use_current_x'] = config.use_current_x
            search_space['future_steps'] = config.get('future_steps', 1)
            search_space['extrapo_mode'] = config.get('extrapo_mode', None)

            if 'lstm' == method:
                trainer = partial(train_lstm, settings=config, verbose=verbose)
            elif 'graphlet' == method:
                trainer = partial(
                    train_graphlet, settings={**config, 'path_extra_keys': get_run_extra_key(config)}, verbose=verbose
                )
            else:
                raise ValueError(f'Unknown method: {method}.')

            analysis = run(
                trainer,
                name=f'{run_name}/{station}',
                config=search_space,
                # scheduler = scheduler_soft(), # ASHA is quite a speedup
                scheduler=scheduler(),
                num_samples=num_samples,
                metric='validation_mse',
                mode='min',
                storage_path=storage_path,
                resources_per_trial=resources_per_trial,
                resume=config.get('resume', False)
            )

            print(f'\n\n~~~ Analysis of {method} ~~~')
            print('Best config: ', analysis.best_config)
            print(f'Best results: {analysis.best_result}')

    elif method in ['transformer', 'transformer_graphlet']:

        for station in read_stations(graph_name):

            search_space = search_space_transformer.copy()
            search_space['station'] = station
            search_space['graph_name'] = graph_name
            # --- dataset specific:
            # 'mask_embedding' or 'interpolation' or 'zero' or None:
            search_space['missing_value_method'] = config.missing_value_method
            search_space['short_subsequence_method'] = config.short_subsequence_method  # 'pad' or 'drop'
            search_space['max_mask_consecutive'] = config.max_mask_consecutive  # only used
            search_space['max_mask_ratio'] = config.max_mask_ratio
            # --- For all models:
            search_space['use_current_x'] = config.use_current_x
            search_space['future_steps'] = config.get('future_steps', 1)
            # --- Transformer specific:
            search_space['max_len'] = config.max_len
            search_space['positional_encoding'] = config.positional_encoding  # None, 'sinusoidal', 'rope', 'learnable'

            if 'transformer' == method:
                trainer = partial(train_transformer, settings=config, verbose=verbose)
            elif 'transformer_graphlet' == method:
                trainer = partial(
                    train_transformer_graphlet, settings={**config, 'path_extra_keys': get_run_extra_key(config)},
                    verbose=verbose
                )
            else:
                raise ValueError(f'Unknown method: {method}.')

            analysis = run(
                trainer,
                name=f'{run_name}/{station}',
                config=search_space,
                # scheduler = scheduler_soft(), # ASHA is quite a speedup
                scheduler=scheduler(),
                num_samples=num_samples,
                metric='validation_mse',
                mode='min',
                storage_path=storage_path,
                resources_per_trial=resources_per_trial,
                resume=config.get('resume', False)
            )

            print(f'\n\n~~~ Analysis of {method} ~~~')
            print('Best config: ', analysis.best_config)
            print(f'Best results: {analysis.best_result}')

    elif 'lstm_embedding' == method:
        search_space = search_space_lstm_embedding.copy()
        search_space['graph_name'] = graph_name
        # --- dataset specific:
        # 'mask_embedding' or 'interpolation' or 'zero' or None:
        search_space['missing_value_method'] = config.missing_value_method
        search_space['short_subsequence_method'] = config.short_subsequence_method  # 'pad' or 'drop'
        search_space['max_mask_consecutive'] = config.max_mask_consecutive  # only used
        search_space['max_mask_ratio'] = config.max_mask_ratio
        # --- For all models:
        search_space['use_current_x'] = config.use_current_x
        search_space['future_steps'] = config.get('future_steps', 1)
        search_space['extrapo_mode'] = config.get('extrapo_mode', None)
        # --- Embedded models specific:
        search_space['use_station_embedding'] = config.use_station_embedding

        trainer = partial(train_lstm_embedding, settings=config, verbose=verbose)

        analysis = run(
            trainer,
            name=run_name,
            config=search_space,
            scheduler=scheduler_soft(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            storage_path=storage_path,
            resources_per_trial=resources_per_trial,
            resume=config.get('resume', False)
        )

    elif 'transformer_embedding' == method:
        search_space = search_space_transformer_embedding.copy()
        search_space['graph_name'] = graph_name
        # --- dataset specific:
        # 'mask_embedding' or 'interpolation' or 'zero' or None:
        search_space['missing_value_method'] = config.missing_value_method
        search_space['short_subsequence_method'] = config.short_subsequence_method  # 'pad' or 'drop'
        search_space['max_mask_consecutive'] = config.max_mask_consecutive  # only used
        search_space['max_mask_ratio'] = config.max_mask_ratio
        # --- For all models:
        search_space['use_current_x'] = config.use_current_x
        search_space['future_steps'] = config.get('future_steps', 1)
        # --- Transformer specific:
        search_space['max_len'] = config.max_len
        search_space['positional_encoding'] = config.positional_encoding  # None, 'sinusoidal', 'rope', 'learnable'
        # --- Embedded models specific:
        search_space['use_station_embedding'] = config.use_station_embedding

        if config.missing_value_method is None:
            trainer = partial(train_transformer_embedding, settings=config, verbose=verbose)
        elif config.missing_value_method == 'mask_embedding':
            trainer = partial(train_masked_transformer_embedding, settings=config, verbose=verbose)
        else:
            raise NotImplementedError(f'Missing value method {config.missing_value_method} not implemented.')

        analysis = run(
            trainer,
            name=run_name,
            config=search_space,
            scheduler=scheduler_soft(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            storage_path=storage_path,
            resources_per_trial=resources_per_trial,
            resume=config.get('resume', False),
        )

    elif 'stgnn' == method:
        search_space = search_space_stgnn.copy()
        search_space['graph_name'] = graph_name
        # --- dataset specific:
        # 'mask_embedding' or 'interpolation' or 'zero' or None:
        search_space['missing_value_method'] = config.missing_value_method
        search_space['short_subsequence_method'] = config.short_subsequence_method  # 'pad' or 'drop'
        search_space['max_mask_consecutive'] = config.max_mask_consecutive  # only used
        search_space['max_mask_ratio'] = config.max_mask_ratio
        # --- For all models:
        search_space['use_current_x'] = config.use_current_x
        search_space['future_steps'] = config.get('future_steps', 1)
        search_space['extrapo_mode'] = config.get('extrapo_mode', None)
        # --- Embedded models specific:
        search_space['use_station_embedding'] = config.use_station_embedding

        trainer = partial(train_stgnn, settings=config, verbose=verbose)

        analysis = run(
            trainer,
            name=run_name,
            config=search_space,
            scheduler=scheduler_single_model_hard(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            max_concurrent_trials=20,  # Fix memory issues
            storage_path=storage_path,
            resources_per_trial=resources_per_trial,
            resume=config.get('resume', False)
        )

    elif 'transformer_stgnn' == method:
        search_space = search_space_transformer_stgnn.copy()
        search_space['graph_name'] = graph_name
        # --- dataset specific:
        # 'mask_embedding' or 'interpolation' or 'zero' or None:
        search_space['missing_value_method'] = config.missing_value_method
        search_space['short_subsequence_method'] = config.short_subsequence_method  # 'pad' or 'drop'
        search_space['max_mask_consecutive'] = config.max_mask_consecutive  # only used
        search_space['max_mask_ratio'] = config.max_mask_ratio
        # --- For all models:
        search_space['use_current_x'] = config.use_current_x
        search_space['future_steps'] = config.get('future_steps', 1)
        # Transformer specific:
        search_space['max_len'] = config.max_len
        search_space['positional_encoding'] = config.positional_encoding  # None, 'sinusoidal', 'rope', 'learnable'
        # --- Embedded models specific:
        search_space['use_station_embedding'] = config.use_station_embedding

        if config.missing_value_method is None:
            trainer = partial(train_transformer_stgnn, settings=config, verbose=verbose)
        elif config.missing_value_method == 'mask_embedding':
            trainer = partial(train_masked_transformer_stgnn, settings=config, verbose=verbose)
        else:
            raise NotImplementedError(f'Missing value method {config.missing_value_method} not implemented.')

        analysis = run(
            trainer,
            name=run_name,
            config=search_space,
            scheduler=scheduler_single_model_hard(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            max_concurrent_trials=20,  # Fix memory issues
            storage_path=storage_path,
            resources_per_trial=resources_per_trial,
            resume=config.get('resume', False)
        )

    else:
        raise ValueError(f'Unknown method: {method}.')

    if method in ['lstm_embedding', 'stgnn', 'transformer_embedding', 'transformer_stgnn']:
        # print(f'trials: {analysis.trials}')  # debug
        # print(f'last results: {[t.last_result for t in analysis.trials]}')  # debug
        # print(f'configs: {[t.config for t in analysis.trials]}')  # debug
        print(f'\n\n~~~ Analysis of {method} ~~~')
        print('Best config: ', analysis.best_config)
        # print('Best trial: ', analysis.get_best_trial())
        print(f'Best results: {analysis.best_result}')

    if if_trim_checkpoints:
        from swissrivernetwork.benchmark.util import trim_checkpoints
        start_time_trim = time.time()
        trim_checkpoints(
            storage_path / run_name, keep_best_n=10, anchor_metric='validation_mse',
            mode='min', if_trim_best_n=True, keep_best_for_trimmed_trials=True, keep_last_for_trimmed_trials=False,
            verbose=False
        )
        end_time_trim = time.time()
        print(f'{SUCCESS_TAG}Trimmed checkpoints in {end_time_trim - start_time_trim:.2f} seconds.')


def parse_config():
    methods = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn',
               'transformer', 'transformer_graphlet', 'transformer_embedding', 'transformer_stgnn']
    graphs = ['swiss-1990', 'swiss-2010', 'zurich']

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help='Path to config file (YAML)')
    parser.add_argument('-m', '--method', required=False, choices=methods)
    parser.add_argument('-g', '--graph', required=False, choices=graphs)
    parser.add_argument(
        '-wl', '--window_len', required=False, type=int, default=90,
        help='Window length (in days) of historical data used for prediction. Default is 90 days.'
    )
    # Ray Tune specific:
    parser.add_argument(
        '-n', '--num_samples', required=False, type=int,
        help='The amount of hyperparameter combination random search samples for ray tune.', default=200  # 200 debug
    )
    parser.add_argument(
        '-s', '--storage_path', required=False, type=str, help='Path to store results.', default=None
    )
    parser.add_argument(
        '-r', '--resume', required=False, type=str2bool, nargs='?', const=True, default=False,
        help='Whether to resume from previous ray tune results in the storage path.'
    )
    parser.add_argument(
        '-rts', '--resume_timestamp', required=False, type=str, default=None,
        help='The time stamp of the previous run to resume from, e.g., 2024-01-01_12-00-00. If None, resume from the latest run.'
    )
    # Dataset specific:
    parser.add_argument(
        '-mvm', '--missing_value_method', required=False, type=str,
        choices=['mask_embedding', 'interpolation', 'zero', 'none'],
        default='none', help='Method to handle missing values in the input data for transformer models.'
    )
    parser.add_argument(
        '-mmc', '--max_mask_consecutive', required=False, type=int, default=12,
        help='Maximum number of consecutive time steps to mask when using mask_embedding for missing values. Default is 12.'
    )
    parser.add_argument(
        '-mmr', '--max_mask_ratio', required=False, type=float, default=0.5,
        help='Maximum ratio of time steps to mask when using mask_embedding for missing values. Default is 0.5.'
    )
    parser.add_argument(
        '-ssm', '--short_subsequence_method', required=False, type=str, choices=['pad', 'drop'],
        default='pad', help='How to handle short subsequences that are shorter than window_len in transformer models.'
    )
    # For all models:
    parser.add_argument(
        '-ucx', '--use_current_x', required=False, type=str2bool, nargs='?', const=True, default=True,
        help='Whether to use the current time step feature in transformer models. If False, only past features are used, which corresponds to next-token prediction.'
    )
    parser.add_argument(
        '-fs', '--future_steps', required=False, type=int, default=1,
        help='Number of future time steps to predict. Only used when use_current_x is False. Default is 1.'
    )
    parser.add_argument(
        '-em', '--extrapo_mode', required=False, type=str, choices=['none', 'limo', 'future_embedding', 'recursive'],
        default='future_embedding', help='Extrapolation mode for multiple-step forecasting.'
    )
    # Transformers specific:
    parser.add_argument(
        '-pe', '--positional_encoding', required=False, type=str, choices=['none', 'sinusoidal', 'rope', 'learnable'],
        default=None, help='Type of positional encoding to use in the transformer model.'
    )
    parser.add_argument(
        '-ml', '--max_len', required=False, type=int, default=None,
        help='Maximum sequence length for transformer models. If less than window_len, will be set to window_len. Default is 90.'
    )
    # Embedded models specific:
    parser.add_argument(
        '-use', '--use_station_embedding', required=False, type=str2bool, nargs='?', const=True, default=True,
        help='Whether to use station embeddings in models that support it.'
    )
    # General:
    parser.add_argument('-v', '--verbose', required=False, type=int, help='Verbosity level.')
    parser.add_argument(
        '-d', '--dev_run', type=str2bool, nargs='?', const=True, default=False, required=False,
        help='If set, use very small subsets for training and validation to test the pipeline.'
    )
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            cfg_from_file = yaml.safe_load(f)

        for k, v in cfg_from_file.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    # Remove positional encoding if set to None:
    # This means no positional encoding will be used, e.g., for LSTM models. Remove it so that some name generation
    # functions do not consider it.
    if args.positional_encoding is None:
        delattr(args, 'positional_encoding')

    for keys in ['missing_value_method', 'positional_encoding', 'extrapo_mode']:
        if hasattr(args, keys) and getattr(args, keys) == 'none':
            setattr(args, keys, None)

    return args


if __name__ == '__main__':
    debug_mode = False  # fixme: debug

    if debug_mode:
        # Exp1: LSTM with embedding:
        debug_cfg = {
            # 'config': CUR_ABS_DIR / 'configs' / 'lstm.yaml',  # fixme: debug
            # 'config': CUR_ABS_DIR / 'configs' / 'transformer.yaml',
            # 'config': CUR_ABS_DIR / 'configs' / 'graphlet.yaml',
            # 'config': CUR_ABS_DIR / 'configs' / 'transformer_graphlet.yaml',
            'config': CUR_ABS_DIR / 'configs' / 'lstm_embedding.yaml',
            # 'config': CUR_ABS_DIR / 'configs' / 'transformer_embedding.yaml',
            # 'config': CUR_ABS_DIR / 'configs' / 'stgnn.yaml',
            # 'config': CUR_ABS_DIR / 'configs' / 'transformer_stgnn.yaml',
            'graph': 'swiss-1990',  # 'swiss-1990', 'swiss-2010', 'zurich'
            'dev_run': True,  # fixme: debug
            'positional_encoding': 'rope',  # fixme: debug, 'none', transformers only: 'sinusoidal', 'rope', 'learnable'
            'window_len': 90,
            'missing_value_method': 'none',  # 'mask_embedding',  # 'mask_embedding', 'interpolation'
            'short_subsequence_method': 'drop',  # 'pad' or 'drop'
            'use_current_x': True,  # fixme: debug, True or False (future-token prediction)
            'future_steps': 0,  # Only works if 'use_current_x' is False
            'max_mask_consecutive': 12,  # only used when missing_value_method is 'mask_embedding'
            'max_mask_ratio': 0.5,
            'resume': False,  # fixme: debug
            'extrapo_mode': 'future_embedding',  # 'limo', 'future_embedding', 'recursive',
            #  Only valid for models with station embeddings, e.g., lstm_embedding, transformer_embedding, stgnn
            #  and transformer_stgnn:
            'use_station_embedding': False,  # fixme: debug
        }

        if not is_transformer_model(debug_cfg['config'].stem):
            debug_cfg['positional_encoding'] = 'none'

        # Set the cfg as the input args:
        sys.argv = [sys.argv[0]]
        for k, v in debug_cfg.items():
            sys.argv += [f'--{k}', str(v)]

    args = parse_config()
    print(f'{INFO_TAG}Arguments: {args}.')
    run_experiment(
        args.method, args.graph, args.num_samples, args.storage_path, benedict(vars(args)),
        args.verbose
    )
