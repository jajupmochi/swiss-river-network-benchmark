import os

import torch
import torch.nn as nn

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import LstmModel
from swissrivernetwork.benchmark.util import *
from swissrivernetwork.benchmark.training import training_loop

def train_lstm(config):
    # config    
    station = config['station']
    graph_name = config['graph_name']    

    # Get the train time series
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)    

    train_isolated_station(config, 1, df)

def train_graphlet(config):
    station = config['station']
    graph_name = config['graph_name']
    num_hops = 1 # use 1-Hop Neighborhood

    # Create DataFrame:
    neighs = extract_neighbors(graph_name, station, num_hops)
    df = read_csv_train(graph_name)
    df = select_isolated_station(df, station)
    df_neighs = [read_csv_prediction_train(graph_name, neigh) for neigh in neighs]
    df = merge_graphlet_dfs(df, df_neighs)
        
    train_isolated_station(config, 1+len(neighs), df)

def train_isolated_station(config, input_size, df):
    # Normalize and Split
    df = normalize_isolated_station(df)
    df_train,df_valid = train_valid_split(config, df)    
    
    # Create Data Loaders
    dataset_train = SequenceWindowedDataset(config['window_len'], df_train)
    dataset_valid = SequenceFullDataset(df_valid)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    # Setup Model
    model = LstmModel(input_size, config['hidden_size'], config['num_layers'])

    training_loop(config, dataloader_train, dataloader_valid, model, len(dataset_valid), False)   

   

if __name__ == '__main__':

    # fix 2010 bug:
    graph_name = 'swiss-2010'
    #graph_name = 'swiss-1990'
    station = '2091'

    # read stations:    
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
        'num_layers': 1
    }
    
    train_lstm(config)
    #train_graphlet(config)

                

