
import torch

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.training import training_loop

def train_lstm_embedding(config):

    # Setup Dataset
    graph_name = config['graph_name']    
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    df = read_csv_train(graph_name)
    datasets_train = []
    datasets_valid = []
    for i,station in enumerate(stations):
        df_station = select_isolated_station(df, station)
        dataset_train,dataset_valid = create_dataset_embedding(config, df_station, i)
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    model = LstmEmbeddingModel(1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_layers'])

    # Run Training Loop!
    training_loop(config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True)
    

def create_dataset_embedding(config, df, i):
    # Normalize
    df = normalize_isolated_station(df)   

    # Train/Validation split
    df_train, df_valid = train_valid_split(config, df)

    # Create datasets
    dataset_train = SequenceWindowedDataset(config['window_len'], df_train, embedding_idx=i)
    dataset_valid = SequenceFullDataset(df_valid, embedding_idx=i)
    return dataset_train, dataset_valid

def train_stgnn(config):

    # Setup Dataset
    graph_name = config['graph_name']    
    stations = read_stations(graph_name)
    num_embeddings = len(stations)
    _,edges = read_graph(graph_name)

    # Read and prepare data
    df = read_csv_train(graph_name)
    df = normalize_columns(df)

    # Create Datasets
    df_train, df_valid = train_valid_split(config, df)
    dataset_train = STGNNSequenceWindowedDataset(config['window_len'], df_train, stations)
    dataset_valid = STGNNSequenceFullDataset(df_valid, stations)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    model = SpatioTemporalEmbeddingModel(config['gnn_conv'], 1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_layers'], config['num_convs'], config['num_heads'])
    
    # Run Training Loop!
    training_loop(config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=False, edges=edges)



if __name__ == '__main__':

    # fix 2010 bug:
    #graph_name = 'swiss-2010'
    graph_name = 'swiss-1990'    

    # read stations:
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
        'num_heads': 0
    }
        
    #train_lstm_embedding(config)
    train_stgnn(config)
