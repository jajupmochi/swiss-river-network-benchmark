import os
import argparse
from datetime import datetime
from pathlib import Path
from functools import partial
from benedict import benedict

import ray
from ray.tune import uniform, randint, run, choice, Callback
from ray.tune.schedulers import ASHAScheduler

from swissrivernetwork.benchmark.train_isolated_station import train_lstm, read_stations, train_graphlet
from swissrivernetwork.benchmark.train_single_model import train_lstm_embedding, train_stgnn, train_transformer
import sys

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
    # "hidden_size": randint(16, 128 + 1),  # 128
    "num_layers": randint(1, 4 + 1),  # more layers!
    "dropout": uniform(0.0, 0.5),
    "num_heads": randint(2, 8 + 1),
    "ratio_heads_to_d_model": choice([4, 8, 16]),  # d_model must be multiple of num_heads
    # "d_model": randint(16, 256 + 1),  # 128
    "dim_feedforward": randint(32, 256 + 1),
    "use_station_embedding": True
}

search_space_stgnn = {
    "batch_size": randint(1, 10),  # Batch size is times |Nodes| 30-50 bigger
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01),  # 10x less
    # "epochs": randint(30, 50), # 30-50
    "epochs": 30,
    "embedding_size": randint(1, 30 + 1),
    "hidden_size": randint(16, 128 + 1),  # 128
    "num_layers": randint(1, 3 + 1),  # more layers!
    # "gnn_conv": choice(['GCN', 'GIN']),
    "gnn_conv": choice(['GraphSAGE']),
    "num_convs": randint(1, 7 + 1)
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
        grace_period=5,
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


def run_experiment(method, graph_name, num_samples, storage_path: str | None, config: benedict, verbose: int):
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
    # now = '2025-09-05_13-34-45'  # fixme: debug only
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
        config.num_cpus if config.num_cpus > 0 else os.cpu_count() + config.num_cpus
    )
    ray.init(num_cpus=num_cpus)

    # update search space (!)
    if 'lstm' == method or 'graphlet' == method:

        for station in read_stations(graph_name):

            search_space = search_space_lstm.copy()
            search_space['station'] = station
            search_space['graph_name'] = graph_name

            trainer = None
            if 'lstm' == method:
                trainer = train_lstm
            if 'graphlet' == method:
                trainer = train_graphlet

            analysis = run(
                trainer,
                name=f'{method}_{station}-{now}',
                config=search_space,
                # scheduler = scheduler_soft(), # ASHA is quite a speedup
                scheduler=scheduler(),
                num_samples=num_samples,
                metric='validation_mse',
                mode='min',
                storage_path=storage_path,
                resources_per_trial=resources_per_trial
            )

            print(f'\n\n~~~ Analysis of {method} ~~~')
            print('Best config: ', analysis.best_config)

    if 'lstm_embedding' == method:
        search_space = search_space_lstm_embedding.copy()
        search_space['graph_name'] = graph_name

        trainer = partial(train_lstm_embedding, settings=config, verbose=verbose)

        analysis = run(
            trainer,
            name=f'{method}-{graph_name}-{now}',
            config=search_space,
            scheduler=scheduler_soft(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            storage_path=storage_path,
            resources_per_trial=resources_per_trial
        )

    if 'transformer_embedding' == method:
        search_space = search_space_transformer_embedding.copy()
        search_space['graph_name'] = graph_name

        trainer = partial(train_transformer, settings=config, verbose=verbose)

        analysis = run(
            trainer,
            name=f'{method}-{graph_name}-{now}',
            config=search_space,
            scheduler=scheduler_soft(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            storage_path=storage_path,
            resources_per_trial=resources_per_trial,
            # resume=True,  # fixme: debug only
        )

    if 'stgnn' == method:
        search_space = search_space_stgnn.copy()
        search_space['graph_name'] = graph_name

        # add GAT Heads
        search_space['num_heads'] = 0
        if search_space['gnn_conv'] == 'GAT':
            search_space['num_heads'] = randint(1, 8)

        trainer = partial(train_stgnn, settings=config, verbose=verbose)

        analysis = run(
            trainer,
            name=f'{method}-{graph_name}-{now}',
            config=search_space,
            scheduler=scheduler_single_model_hard(),  # ASHA is quite a speedup
            num_samples=num_samples,
            metric='validation_mse',
            mode='min',
            max_concurrent_trials=20,  # Fix memory issues
            storage_path=storage_path,
            resources_per_trial=resources_per_trial
        )

    if method in ['lstm_embedding', 'stgnn', 'transformer_embedding']:
        print(f'\n\n~~~ Analysis of {method} ~~~')
        print('Best config: ', analysis.best_config)
        # print('Best trial: ', analysis.get_best_trial())
        print(f'Best results: {analysis.best_result}')


def parse_config():
    methods = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn', 'transformer_embedding']
    graphs = ['swiss-1990', 'swiss-2010', 'zurich']

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help='Path to config file (YAML)')
    parser.add_argument('-m', '--method', required=False, choices=methods)
    parser.add_argument('-g', '--graph', required=False, choices=graphs)
    parser.add_argument(
        '-n', '--num_samples', required=False, type=int,
        help='The amount of hyperparameter combination random search samples for ray tune.', default=200
    )
    parser.add_argument(
        '-s', '--storage_path', required=False, type=str, help='Path to store results.', default=None
    )
    parser.add_argument('-v', '--verbose', required=False, type=int, help='Verbosity level.')
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            cfg_from_file = yaml.safe_load(f)

        for k, v in cfg_from_file.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    return args


if __name__ == '__main__':
    debug_mode = False  # fixme: debug

    if debug_mode:
        # Exp1: LSTM with embedding:
        debug_cfg = {
            # 'config': CUR_ABS_DIR / 'configs' / 'lstm_embedding.yaml',
            'config': CUR_ABS_DIR / 'configs' / 'transformer_embedding.yaml',
            'graph': 'swiss-2010',  # 'swiss-1990', 'swiss-2010', 'zurich'
        }

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
