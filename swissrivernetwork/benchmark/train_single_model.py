from benedict import benedict

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.training import training_loop
from swissrivernetwork.util.scaler import StationSplitScaler

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green


def train_lstm_embedding(config, settings: benedict = benedict({}), verbose: int = 2):
    # Setup Dataset
    graph_name = config['graph_name']
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    df = read_csv_train(graph_name)
    datasets_train = []
    datasets_valid = []
    for i, station in enumerate(stations):
        df_station = select_isolated_station(df, station)
        dataset_train, dataset_valid, normalizer_at, normalizer_wt = create_dataset_embedding(
            config, df_station, i, valid_use_window=False, name=station,
            dev_run=settings.get('dev_run', False)
        )
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(
        # ``drop_last`` was True. Changed to default (False) to be consistent with other models:
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=False
    )
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, shuffle=False, drop_last=False)

    model = LstmEmbeddingModel(1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_layers'])

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True,
        normalizer_at=normalizer_at, normalizer_wt=normalizer_wt, settings=settings, verbose=verbose
    )


def create_dataset_embedding(
        config, df, i, valid_use_window: bool = False, short_subsequence_method: str = 'pad',  # 'pad' or 'drop'
        name: str = '',  # station name
        dev_run: bool = False
):
    # Normalize
    df, normalizer_at, normalizer_wt = normalize_isolated_station(df)

    # Train/Validation split
    df_train, df_valid = train_valid_split(config, df)

    # Create datasets
    dataset_train = SequenceWindowedDataset(
        config['window_len'], df_train, embedding_idx=i, name=name, dev_run=dev_run
    )
    if valid_use_window:
        dataset_valid = SequenceWindowedDataset(
            config['window_len'], df_valid, embedding_idx=i, name=name, dev_run=dev_run
        )
    else:
        dataset_valid = SequenceFullDataset(df_valid, embedding_idx=i, name=name)
    return dataset_train, dataset_valid, normalizer_at, normalizer_wt


def train_stgnn(config, settings: benedict = benedict({}), verbose: int = 2):
    # # test only
    # print(config)
    # print(settings)

    # Setup Dataset
    graph_name = config['graph_name']
    stations = read_stations(graph_name)
    num_embeddings = len(stations)
    _, edges = read_graph(graph_name)

    # Read and prepare data
    df = read_csv_train(graph_name)
    df, normalizer = normalize_columns(df)
    normalizer_wt = StationSplitScaler(normalizer, feat_suffix='_wt')

    # Create Datasets
    df_train, df_valid = train_valid_split(config, df)
    dataset_train = STGNNSequenceWindowedDataset(
        config['window_len'], df_train, stations, dev_run=settings.get('dev_run', False)
    )
    dataset_valid = STGNNSequenceFullDataset(df_valid, stations)
    # dataset_valid = STGNNSequenceWindowedDataset(  # fixme: debug
    #     config['window_len'], df_valid, stations
    # )

    dataloader_train = torch.utils.data.DataLoader(
        # ``drop_last`` was True. Changed to default (False) to allow dev run with small dataset:
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=False  # todo: USE True?
    )
    batch_size = 1 if isinstance(dataset_valid, STGNNSequenceFullDataset) else config['batch_size']
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = SpatioTemporalEmbeddingModel(
        config['gnn_conv'], 1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_layers'],
        config['num_convs'], config['num_heads'], edge_hidden_size=config.get('edge_hidden_size'),
        temporal_func='lstm_embedding',
    )

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=False, edges=edges,
        normalizer_wt=normalizer_wt,
        settings=settings, verbose=verbose
    )


# %% Transformer Embedding:


def train_transformer(config, settings: benedict = benedict({}), verbose: int = 2):
    """
    Train on a vanilla transformer model with / without station embeddings.
    """
    # Setup Dataset
    graph_name = config['graph_name']
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    df = read_csv_train(graph_name)
    datasets_train = []
    datasets_valid = []
    normalizers_at, normalizers_wt = {}, {}
    for i, station in enumerate(stations):
        df_station = select_isolated_station(df, station)
        dataset_train, dataset_valid, normalizer_at, normalizer_wt = create_dataset_embedding(
            config, df_station, i, valid_use_window=True, short_subsequence_method=config['short_subsequence_method'],
            name=station,
            dev_run=settings.get('dev_run', False)
        )
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
        normalizers_at[station] = normalizer_at
        normalizers_wt[station] = normalizer_wt
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=config['batch_size'], shuffle=False, drop_last=False
    )

    # print(f'{INFO_TAG}The positional encoding is set to:', config.get('positional_encoding', 'rope'), 'MUHAHA')
    # exit()  # debug
    model = TransformerEmbeddingModel(
        1, num_embeddings=num_embeddings if config['use_station_embedding'] else 0,
        embedding_size=config['embedding_size'],
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
    )

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True,
        normalizer_at=normalizers_at, normalizer_wt=normalizers_wt, settings=settings, verbose=verbose
    )


def create_masked_dataset_embedding(
        config, df, i, valid_use_window: bool = False, max_mask_ratio: float = 0.25, max_mask_consecutive: int = 10,
        short_subsequence_method: str = 'pad',  # 'pad' or 'drop'
        name: str = '',  # station name
        dev_run: bool = False
):
    # Normalize
    df, normalizer_at, normalizer_wt = normalize_isolated_station(df)

    # Train/Validation split
    df_train, df_valid = train_valid_split(config, df)

    # Create datasets
    dataset_train = SequenceMaskedWindowedDataset(
        config['window_len'], df_train, embedding_idx=i,
        max_mask_ratio=max_mask_ratio, max_mask_consecutive=max_mask_consecutive,
        short_subsequence_method=short_subsequence_method,
        name=name,
        dev_run=dev_run
    )
    if valid_use_window:
        dataset_valid = SequenceMaskedWindowedDataset(
            config['window_len'], df_valid, embedding_idx=i,
            max_mask_ratio=max_mask_ratio, max_mask_consecutive=max_mask_consecutive,
            short_subsequence_method=short_subsequence_method,
            name=name,
            dev_run=dev_run  # debug
        )
    else:
        raise NotImplementedError('Validation without window not implemented for masked dataset.')
        dataset_valid = SequenceFullDataset(df_valid, embedding_idx=i, name=name)
    return dataset_train, dataset_valid, normalizer_at, normalizer_wt


def train_masked_transformer(config, settings: benedict = benedict({}), verbose: int = 2):
    """
    Train on a vanilla masked transformer model with / without station embeddings.
    """
    # Setup Dataset
    graph_name = config['graph_name']
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    df = read_csv_train(graph_name)
    datasets_train = []
    datasets_valid = []
    normalizers_at, normalizers_wt = {}, {}
    for i, station in enumerate(stations):
        df_station = select_isolated_station(df, station)
        dataset_train, dataset_valid, normalizer_at, normalizer_wt = create_masked_dataset_embedding(
            config, df_station, i, valid_use_window=True,
            max_mask_ratio=config['max_mask_ratio'], max_mask_consecutive=config['max_mask_consecutive'],
            short_subsequence_method=config['short_subsequence_method'],
            name=station,
            dev_run=settings.get('dev_run', False)
        )
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
        normalizers_at[station] = normalizer_at
        normalizers_wt[station] = normalizer_wt
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=False
        # fixme: drop was True. Should be False for long win_lens? check the other codes that use this.
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=config['batch_size'], shuffle=False, drop_last=False
    )

    model = TransformerEmbeddingModel(
        1, num_embeddings=num_embeddings if config['use_station_embedding'] else 0,
        embedding_size=config['embedding_size'],
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
    )

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True,
        normalizer_at=normalizers_at, normalizer_wt=normalizers_wt, settings=settings, verbose=verbose
    )


# %% Transformer + STGNN:


def train_transformer_stgnn(config, settings: benedict = benedict({}), valid_use_window: bool = True, verbose: int = 2):
    # # test only
    # print(config)
    # print(settings)

    # Setup Dataset
    graph_name = config['graph_name']
    stations = read_stations(graph_name)
    num_embeddings = len(stations)
    _, edges = read_graph(graph_name)

    # Read and prepare data
    df = read_csv_train(graph_name)
    df, normalizer = normalize_columns(df)
    normalizer_wt = StationSplitScaler(normalizer, feat_suffix='_wt')

    # Create Datasets
    df_train, df_valid = train_valid_split(config, df)
    dataset_train = STGNNSequenceWindowedDataset(
        config['window_len'], df_train, stations, dev_run=settings.get('dev_run', False)
    )
    if valid_use_window:
        dataset_valid = STGNNSequenceWindowedDataset(config['window_len'], df_valid, stations)
    else:
        dataset_valid = STGNNSequenceFullDataset(df_valid, stations)

    dataloader_train = torch.utils.data.DataLoader(
        # ``drop_last`` was True. Changed to default (False) to allow dev run with small dataset:
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=False  # todo: USE True?
    )
    batch_size = 1 if isinstance(dataset_valid, STGNNSequenceFullDataset) else config['batch_size']
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = SpatioTemporalEmbeddingModel(
        config['gnn_conv'], 1, num_embeddings, config['embedding_size'],
        config['hidden_size'] if config.get('hidden_size', None) else int(
            config['ratio_heads_to_d_model'] * config['num_t_heads']
        ),
        config['num_layers'],
        # Graph NN params:
        config['num_convs'], config['num_heads'], edge_hidden_size=config.get('edge_hidden_size'),
        # Transformer params:
        temporal_func='transformer_embedding',
        num_t_heads=config['num_t_heads'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_len=config['max_len'],
        # Dataset related:
        missing_value_method=config['missing_value_method'],
        use_current_x=config['use_current_x'],
        positional_encoding=config.get('positional_encoding', 'rope'),
    )

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=False, edges=edges,
        normalizer_wt=normalizer_wt,
        settings=settings, verbose=verbose
    )


# %%


if __name__ == '__main__':
    # fix 2010 bug:
    # graph_name = 'swiss-2010'
    graph_name = 'swiss-1990'

    method = 'transformer_embedding'  # 'lstm_embedding', 'stgnn', 'transformer_embedding', or 'transformer_stgnn'

    # read stations:
    print(f'{INFO_TAG}Stations in graph {graph_name}:')
    print(read_stations(graph_name))

    config = {
        'graph_name': graph_name,
        'batch_size': 256,  # fixme: default 256, which is too large for stgnn with MPNN conv or transformer_stgnn
        'window_len': 90,  # fixme: test 90, 366, 365+366=731, inf (all past data)
        'train_split': 0.8,
        'learning_rate': 0.001,
        'epochs': 30,
        'embedding_size': 5,
        'hidden_size': 32,
        'edge_hidden_size': 8,  # for gnn edge network only
        'num_layers': 1,
        'gnn_conv': 'GAT',  # GraphSAGE
        'num_convs': 1,
        'num_heads': 4,  # fixme: only used for GAT when using stgnn
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
        'use_current_x': True,  # whether to use the current day's features as input to predict next day
    }

    # Extra config:
    settings = {
        'dev_run': False,  # fixme: debug  Set training and validation to very small subsets (4) and disable wandb
        'enable_wandb': True,  # fixme: debug Enable wandb logging
    }

    if method == 'lstm_embedding':
        train_lstm_embedding(config, settings=benedict({**settings, 'method': 'lstm_embedding'}))
    elif method == 'stgnn':
        train_stgnn(config, settings=benedict({**settings, 'method': 'stgnn'}))

    if not method.startswith('transformer'):
        exit()

    # Transformer specific:
    config.update(
        {
            'd_model': 32,
            'num_t_heads': 4,  # temporal attention heads
            'num_layers': 4,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'use_station_embedding': True,
            'max_len': max(500, config['window_len']),  # maximum length of the input sequence (for positional encoding)
            # 'mask_embedding' or 'interpolation' or 'zero' or None
            'missing_value_method': None,  # fixme: test. based on lstm or transformer fixme: mask_embedding
            'use_current_x': True,  # whether to use the current day's features as input to predict next day
            'positional_encoding': 'rope',  # 'sinusoidal' or 'rope' or 'learnable' or None

        }
    )

    if method == 'transformer_embedding':
        if config['missing_value_method'] is None:
            train_transformer(config, settings=benedict({**settings, 'method': 'transformer_embedding'}))
        elif config['missing_value_method'] in ['mask_embedding', 'interpolation', 'zero']:
            train_masked_transformer(config, settings=benedict({**settings, 'method': 'masked_transformer_embedding'}))
        else:
            raise NotImplementedError(f'Missing value method {config["missing_value_method"]} not implemented.')

    elif method == 'transformer_stgnn':
        # STGNN-Transformer:
        if config['missing_value_method'] is None:
            train_transformer_stgnn(config, settings=benedict({**settings, 'method': 'transformer_stgnn'}))
        elif config['missing_value_method'] in ['mask_embedding', 'interpolation', 'zero']:
            train_masked_transformer_stgnn(
                config, settings=benedict({**settings, 'method': 'masked_transformer_stgnn'})
            )
        else:
            raise NotImplementedError(f'Missing value method {config["missing_value_method"]} not implemented.')

# %%

# # # this is the best config for swiss-1990 with transformer sinusoidal:
# # config.update(
# #     {
# #         'batch_size': 160,
# #         'window_len': 90,
# #         'learning_rate': 0.003399,
# #         'epochs': 30,
# #         'embedding_size': 10,
# #         'd_model': 32,
# #         'num_heads': 4,
# #         'num_layers': 3,
# #         'dim_feedforward': 116,
# #         'dropout': 0.0899,
# #         'use_station_embedding': True,
# #         'max_len': max(500, config['window_len']),  # maximum length of the input sequence (for positional encoding)
# #         # 'mask_embedding' or 'interpolation' or 'zero' or None
# #         'missing_value_method': 'mask_embedding',
# #         'use_current_x': True,  # whether to use the current day's features as input to predict next day
# #         'positional_encoding': 'sinusoidal',  # 'sinusoidal' or 'rope' or 'learnable' or None
# #
# #     }
# # )
