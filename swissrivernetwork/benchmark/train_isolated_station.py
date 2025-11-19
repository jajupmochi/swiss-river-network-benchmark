from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import LstmModel, TransformerModel, ExtrapoLstmModel
from swissrivernetwork.benchmark.training import training_loop
from swissrivernetwork.benchmark.util import *

CUR_ABS_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = (CUR_ABS_DIR / '../../' / 'swissrivernetwork/benchmark/outputs/ray_results/').resolve()
DUMP_DIR = (CUR_ABS_DIR / '../../' / 'swissrivernetwork/benchmark/dump/').resolve()


# %% Isolated Station Training:


def train_lstm(config, settings: benedict = benedict({}), verbose: int = 2):
    # config    
    station = config['station']
    graph_name = config['graph_name']

    # Get the train time series
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)

    train_isolated_station(config, 1, df, settings=settings, verbose=verbose)


def train_transformer(config, settings: benedict = benedict({}), verbose: int = 2):
    # config
    station = config['station']
    graph_name = config['graph_name']

    # Get the train time series
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)

    train_isolated_station(config, 1, df, settings=settings, verbose=verbose)


# %% Graphlet Training:


def train_graphlet(config, settings: benedict = benedict({}), verbose: int = 2):
    station = config['station']
    graph_name = config['graph_name']
    num_hops = 1  # use 1-Hop Neighborhood

    # Create DataFrame:
    neighs = extract_neighbors(graph_name, station, num_hops)
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)
    # Run test_lstm first to generate neigh prediction files:
    predict_dump_dir = DUMP_DIR / 'predictions' / settings.get('path_extra_keys', '')
    df_neighs = [read_csv_prediction_train(graph_name, 'lstm', neigh, predict_dump_dir=predict_dump_dir) for neigh in
                 neighs]
    df = merge_graphlet_dfs(df, df_neighs)

    train_isolated_station(config, 1 + len(neighs), df, settings=settings, verbose=verbose)


def train_transformer_graphlet(config, settings: benedict = benedict({}), verbose: int = 2):
    station = config['station']
    graph_name = config['graph_name']
    num_hops = 1  # use 1-Hop Neighborhood

    # Create DataFrame:
    neighs = extract_neighbors(graph_name, station, num_hops)
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)
    # Run test_transformer first to generate neigh prediction files:
    predict_dump_dir = DUMP_DIR / 'predictions' / settings.get('path_extra_keys', '')
    df_neighs = [read_csv_prediction_train(graph_name, 'transformer', neigh, predict_dump_dir=predict_dump_dir) for
                 neigh in neighs]
    df = merge_graphlet_dfs(df, df_neighs)

    train_isolated_station(config, 1 + len(neighs), df, settings=settings, verbose=verbose)


# %% Common Training Function:


def train_isolated_station(config, input_size, df, settings: benedict = benedict({}), verbose: int = 2):
    # Normalize and Split
    df, normalizer_at, normalizer_wt = normalize_isolated_station(df)
    df_train, df_valid = train_valid_split(config, df)

    # Create Data Loaders
    dataset_train = SequenceWindowedDataset(
        config['window_len'], df_train, name=config['station'], dev_run=settings.get('dev_run', False)
    )
    # todo: use windowed dataset for validation too?:
    if settings.get('valid_use_window', True):  # window valid data for transformer
        dataset_valid = SequenceWindowedDataset(
            config['window_len'], df_valid, name=config['station'],
            # dev_run=settings.get('dev_run', False)
        )
    else:
        dataset_valid = SequenceFullDataset(df_valid, name=config['station'])
    dataloader_train = torch.utils.data.DataLoader(
        # ``drop_last`` was True. Changed to default (False) to allow dev run with small dataset:
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=False  # todo: use True?
    )
    batch_size = 1 if isinstance(dataset_valid, SequenceFullDataset) else config['batch_size']
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Setup Model
    if is_transformer_model(settings['method']):
        model = TransformerModel(
            input_size, 0, 0,
            num_heads=config['num_t_heads'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            d_model=config['d_model'] if config.get('d_model', None) else int(
                config['ratio_heads_to_d_model'] * config['num_t_heads']
            ),
            max_len=config['max_len'],
            missing_value_method=config['missing_value_method'],
            use_current_x=config['use_current_x'],
            positional_encoding=config.get('positional_encoding', 'rope'),
            future_steps=config.get('future_steps', 1),
        )
    else:
        if config.get('use_current_x', True):
            model = LstmModel(input_size, config['hidden_size'], config['num_layers'])
        else:
            model = ExtrapoLstmModel(
                input_size, config['hidden_size'], config['num_layers'], future_steps=config['future_steps']
            )

    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=False,
        normalizer_wt={config['station']: normalizer_wt},
        settings=settings, verbose=verbose
    )


if __name__ == '__main__':
    # fix 2010 bug:
    graph_name = 'swiss-1990'  # 'swiss-1990' or 'swiss-2010' or 'zurich'

    method = 'transformer_graphlet'  # 'lstm', 'graphlet', 'transformer', or 'transformer_graphlet'

    station = '2091'  # '2091' for 'swiss-1990', '534' for 'zurich'

    # read stations:
    print(f'{INFO_TAG}Stations in graph {graph_name}:')
    print(read_stations(graph_name))

    config = {
        'station': station,
        'graph_name': graph_name,
        'batch_size': 256,
        'window_len': 90,
        'train_split': 0.8,
        'learning_rate': 0.001,
        'epochs': 30,
        'hidden_size': 32,
        'num_layers': 1,
        # --- Dataset specific:
        # for masked model such as masked_transformer_embedding:
        'max_mask_ratio': 0.5,  # maximum ratio of days to be masked in a window todo: not used yet.
        # maximum number of consecutive days to be masked in a window
        'max_mask_consecutive': 12,  # 12   # fixme: test. based on if win_len == inf:
        # what to do if subsequence is shorter than window_len, 'pad' or 'drop'.
        # Applied on both training and validation sets:
        'short_subsequence_method': 'drop',  # fixme: debug: only 'pad' for win len > 90
        # --- Exp configs used for all models:
        # 'mask_embedding' or 'interpolation' or 'zero' or None
        'missing_value_method': None,  # fixme: test. based on lstm or transformer
        'use_current_x': False,  # whether to use the current day's features as input to predict next day
        'future_steps': 7,  # fixme: days to predict ahead.
    }

    # Extra config:
    settings = {
        'dev_run': True,  # fixme: debug  Set training and validation to very small subsets (4) and disable wandb
        'enable_wandb': True,  # fixme: debug Enable wandb logging
    }

    if method == 'lstm':
        train_lstm(
            config, settings=benedict({**settings, 'method': 'lstm', 'path_extra_keys': get_run_extra_key(config)})
        )
    elif method == 'graphlet':
        train_graphlet(
            config, settings=benedict({**settings, 'method': 'graphlet', 'path_extra_keys': get_run_extra_key(config)})
        )

    if not is_transformer_model(method):
        exit()

    # Transformer specific configs:
    config.update(
        {
            'd_model': 32,
            'num_t_heads': 4,  # temporal attention heads
            'num_layers': 4,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'use_station_embedding': False,  # fixme: remove?
            'max_len': max(500, config['window_len']),  # maximum length of the input sequence (for positional encoding)
            'positional_encoding': 'rope',  # 'sinusoidal' or 'rope' or 'learnable' or None
            # --- Updated for transformer models:
            # 'mask_embedding' or 'interpolation' or 'zero' or None
            'missing_value_method': None,  # fixme: test. based on lstm or transformer fixme: mask_embedding
        }
    )

    if method == 'transformer':
        if config['missing_value_method'] is None:
            train_transformer(
                config,
                settings=benedict({**settings, 'method': 'transformer', 'path_extra_keys': get_run_extra_key(config)})
            )
        else:
            raise NotImplementedError
    elif method == 'transformer_graphlet':
        if config['missing_value_method'] is None:
            train_transformer_graphlet(
                config, settings=benedict(
                    {**settings, 'method': 'transformer_graphlet', 'path_extra_keys': get_run_extra_key(config)}
                )
            )
        else:
            raise NotImplementedError
    else:
        raise ValueError(f'Unknown method: {method}.')
