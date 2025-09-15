from benedict import benedict

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.training import training_loop

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
        dataset_train, dataset_valid, normalizer_at, normalizer_wt = create_dataset_embedding(config, df_station, i)
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True
    )
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    model = LstmEmbeddingModel(1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_layers'])

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True,
        normalizer_at=normalizer_at, normalizer_wt=normalizer_wt, settings=settings, verbose=verbose
    )


def create_dataset_embedding(config, df, i, valid_use_window: bool = False):
    # Normalize
    df, normalizer_at, normalizer_wt = normalize_isolated_station(df)

    # Train/Validation split
    df_train, df_valid = train_valid_split(config, df)

    # Create datasets
    dataset_train = SequenceWindowedDataset(
        config['window_len'], df_train, embedding_idx=i, dev_run=config.get('dev_run', False)
    )
    if valid_use_window:
        dataset_valid = SequenceWindowedDataset(
            config['window_len'], df_valid, embedding_idx=i, dev_run=config.get('dev_run', False)
        )
    else:
        dataset_valid = SequenceFullDataset(df_valid, embedding_idx=i)
    return dataset_train, dataset_valid, normalizer_at, normalizer_wt


def train_stgnn(config, settings: benedict = benedict({}), verbose: int = 2):
    # Setup Dataset
    graph_name = config['graph_name']
    stations = read_stations(graph_name)
    num_embeddings = len(stations)
    _, edges = read_graph(graph_name)

    # Read and prepare data
    df = read_csv_train(graph_name)
    df = normalize_columns(df)

    # Create Datasets
    df_train, df_valid = train_valid_split(config, df)
    dataset_train = STGNNSequenceWindowedDataset(
        config['window_len'], df_train, stations, dev_run=config.get('dev_run', False)
    )
    dataset_valid = STGNNSequenceFullDataset(df_valid, stations)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True
    )
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    model = SpatioTemporalEmbeddingModel(
        config['gnn_conv'], 1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_layers'],
        config['num_convs'], config['num_heads']
    )

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=False, edges=edges,
        settings=settings, verbose=verbose
    )


def train_transformer(config, settings: benedict = benedict({}), verbose: int = 2):
    """
    Train on a pure transformer model without station embeddings.
    """
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
            config, df_station, i, valid_use_window=True
        )
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=config['batch_size'], shuffle=False, drop_last=False
    )

    model = TransformerEmbeddingModel(
        1, num_embeddings=num_embeddings if config['use_station_embedding'] else 0,
        embedding_size=config['embedding_size'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        d_model=config['d_model'] if config.get('d_model', None) else int(
            config['ratio_heads_to_d_model'] * config['num_heads']
        ),
    )

    # Run Training Loop!
    training_loop(
        config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True,
        normalizer_at=normalizer_at, normalizer_wt=normalizer_wt, settings=settings, verbose=verbose
    )


if __name__ == '__main__':
    # fix 2010 bug:
    # graph_name = 'swiss-2010'
    graph_name = 'swiss-1990'

    # read stations:
    print(f'{INFO_TAG}Stations in graph {graph_name}:')
    print(read_stations(graph_name))

    config = {
        'graph_name': graph_name,
        'batch_size': 256,
        'window_len': 90,
        'train_split': 0.8,
        'learning_rate': 0.001,
        'epochs': 30,
        'embedding_size': 5,
        'hidden_size': 32,
        'num_layers': 1,
        'gnn_conv': 'GraphSAGE',
        'num_convs': 1,
        'num_heads': 0,
    }

    # Extra config:
    config.update(
        {
            'dev_run': False,  # fixme: debug  Set training and validation to very small subsets (4) and disable wandb
        }
    )

    # train_lstm_embedding(config, settings=benedict({'enable_wandb': True, 'method': 'lstm_embedding'}))
    # # # train_stgnn(config)

    # Transformer specific:
    config.update(
        {
            'd_model': 32,
            'num_heads': 4,
            'num_layers': 4,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'use_station_embedding': True
        }
    )
    train_transformer(config, settings=benedict({'enable_wandb': True, 'method': 'transformer_embedding'}))
