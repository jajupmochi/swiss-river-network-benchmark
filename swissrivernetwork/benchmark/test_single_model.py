import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ray.tune import ExperimentAnalysis
from sklearn.preprocessing import MinMaxScaler

# from swissrivernetwork.experiment.error import Error
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.util import *

from swissrivernetwork.benchmark.test_isolated_station import plot, summary
from swissrivernetwork.experiment.error import compute_errors


def fit_column_normalizers(df):
    normalizers = dict()
    columns = df.columns
    for col in columns:
        df_col = df.loc[:, col]
        normalizer = MinMaxScaler()
        normalizer.fit(df[col].values.reshape(-1, 1))
        normalizers[col] = normalizer
    return normalizers


def test_stgnn(graph_name, model, dump_dir: Path | str = 'swissrivernetwork/journal/dump'):
    # TODO:
    # Read Train Data for normalizer
    df_train = read_csv_train(graph_name)
    df_train = df_train.loc[:, ~df_train.columns.isin(['epoch_day', 'has_nan'])]
    normalizers = fit_column_normalizers(df_train)

    # Read Test Data
    stations = read_stations(graph_name)
    n_stations = len(stations)
    num_embeddings = len(stations)
    _, edges = read_graph(graph_name)
    df = read_csv_test(graph_name)

    # Normalize Input Values:
    for station in stations:
        df[f'{station}_at'] = normalizers[f'{station}_at'].transform(df[f'{station}_at'].values.reshape(-1, 1))
    # TODO: test if equal to column wise normalizer.. (but should)

    # Create Dataset
    dataset = STGNNSequenceFullDataset(df, stations)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Compute this:
    # epoch_days, prediction_norm, mask, actual, prediction
    epoch_days = [[] for _ in range(n_stations)]
    prediction_norm = [[] for _ in range(n_stations)]
    masks = [[] for _ in range(n_stations)]
    actual = [[] for _ in range(n_stations)]
    prediction = [[] for _ in range(n_stations)]
    with torch.no_grad():
        model.eval()
        for (t, e, x, y) in dataloader:

            # Run Model
            out = model(x, edges)

            # Check for only one batch:
            assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Split the predictions per station:
            mask = ~torch.isnan(y)
            # y = y[mask]
            # out_masked = out[mask]
            for i, station in enumerate(stations):

                # Store epoch_days and prediction_norm on all days:
                epoch_days[i].append(t[0, i].detach().numpy())
                prediction_norm[i].append(out[0, i].detach().numpy())  # store normalized predictions

                # mask values:             
                mask_i = mask[0, i]
                masks[i].append(mask_i)
                if mask_i.sum() == 0:
                    continue  # skip if all is masked

                # Denormalize (masked output)
                ys = y[0, i][mask_i]
                outs = out[0, i][mask_i]
                outs = normalizers[f'{station}_wt'].inverse_transform(outs.detach().numpy().reshape(-1, 1))

                # Store values            
                actual[i].append(ys.detach().numpy())  # Original
                prediction[i].append(outs.flatten())  # Denormalized Prediction

    # Combine arrays:
    for i in range(n_stations):
        epoch_days[i] = np.concatenate(epoch_days[i], axis=0).flatten()
        prediction_norm[i] = np.concatenate(prediction_norm[i], axis=0).flatten()
        masks[i] = np.concatenate(masks[i], axis=0).flatten()
        actual[i] = np.concatenate(actual[i], axis=0).flatten()
        prediction[i] = np.concatenate(prediction[i], axis=0).flatten()

    # Compute errors
    rmses = []
    maes = []
    nses = []
    ns = []
    for i, station in enumerate(stations):
        rmse, mae, nse = compute_errors(actual[i], prediction[i])
        title = summary(station, rmse, mae, nse)
        print(title)

        # Plot Figure of Test Data
        plot(graph_name, 'stgnn', station, epoch_days[i][masks[i]], actual[i], prediction[i], title, dump_dir=dump_dir)

        rmses.append(rmse)
        maes.append(mae)
        nses.append(nse)
        ns.append(len(prediction))

    return rmses, maes, nses, ns


if __name__ == '__main__':
    method = 'stgnn'
    graph_name = 'swiss-1990'

    # Read statistics
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    # Load a model from a config:
    analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/stgnn-2025-06-16_16-11-22')

    # COPY CODE
    # Get best trial and load model:
    # This is probably not correct! (see ray evaluate)
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")
    print('use best config:', best_config)

    # COPY CODE:
    # Create Model
    model = SpatioTemporalEmbeddingModel(
        best_config['gnn_conv'], 1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'],
        best_config['num_layers'], best_config['num_convs']
    )
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))

    test_stgnn(graph_name, model)
