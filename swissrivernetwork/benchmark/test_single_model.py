from ray.tune import ExperimentAnalysis

from swissrivernetwork.benchmark.dataset import *
# from swissrivernetwork.experiment.error import Error
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.test_isolated_station import summary
from swissrivernetwork.benchmark.training import check_is_aggregation_needed
from swissrivernetwork.benchmark.util import *
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


def test_stgnn(
        graph_name, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump', verbose: int = 2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

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

    edges = edges.to(device)

    # Normalize Input Values:
    for station in stations:
        df[f'{station}_at'] = normalizers[f'{station}_at'].transform(df[f'{station}_at'].values.reshape(-1, 1))
    # TODO: test if equal to column wise normalizer.. (but should)

    # Create Dataset
    # dataset = STGNNSequenceFullDataset(df, stations)   # debug
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
    if window_len is None:
        dataset = STGNNSequenceFullDataset(df, stations)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)  # fixme: drop_last? and the other locations?
    else:
        dataset = STGNNSequenceWindowedDataset(window_len, df, stations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)

    # Compute this:
    # epoch_days, prediction_norm, mask, actual, prediction
    epoch_days, prediction_norm, actual, masks = [], [], [], []
    with torch.no_grad():
        model.eval()
        for (t, e, x, y) in dataloader:
            t, e, x, y = t.to(device), e.to(device), x.to(device), y.to(device)

            # Run Model
            out = model(x, edges)

            # # Check for only one batch:
            # assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Split the predictions per station:
            mask = ~torch.isnan(y)
            # y = y[mask]
            # out_masked = out[mask]

            epoch_days.append(t.cpu().detach().numpy())
            prediction_norm.append(out.cpu().detach().numpy())
            actual.append(y.cpu().detach().numpy())
            masks.append(mask.cpu().detach().numpy())

    # Shape before: [[B, n_stations, seq_len / win_len, 1], ...]
    # Shape after:  [[n_stations, seq_len / win_len, 1], ...] (n_subsequences)
    epoch_days = [i for t in epoch_days for i in t]
    prediction_norm = [i for p in prediction_norm for i in p]
    actual = [i for a in actual for i in a]
    masks = [i for m in masks for i in m]

    # Combine arrays:
    all_epoch_days, all_actual, all_prediction, all_masks = [], [], [], []
    for i, station in enumerate(stations):
        station_epoch_days = np.concatenate([e[i] for e in epoch_days], axis=0).flatten()
        station_prediction_norm = np.concatenate([p[i] for p in prediction_norm], axis=0).flatten()
        station_masks = np.concatenate([m[i] for m in masks], axis=0).flatten()
        station_actual = np.concatenate([a[i] for a in actual], axis=0).flatten()

        if check_is_aggregation_needed(dataloader):
            station_epoch_days, aggregated_dict = aggregate_day_predictions(
                station_epoch_days,
                {
                    'prediction_norm': station_prediction_norm,
                    'mask': station_masks,
                    'actual': station_actual,
                },
                method='longest_history'
            )
            station_masks = aggregated_dict['mask']
            station_prediction_norm = aggregated_dict['prediction_norm'][station_masks]
            station_actual = aggregated_dict['actual'][station_masks]
        else:
            station_prediction_norm = station_prediction_norm[station_masks]
            station_actual = station_actual[station_masks]

        # Denormalize predictions
        station_prediction = normalizers[f'{station}_wt'].inverse_transform(
            station_prediction_norm.reshape(-1, 1)
        ).flatten()

        all_epoch_days.append(station_epoch_days[station_masks])  # masked
        all_actual.append(station_actual)  # masked
        all_prediction.append(station_prediction)  # masked

    # Compute errors
    rmses = []
    maes = []
    nses = []
    ns = []
    for i, station in enumerate(stations):
        rmse, mae, nse = compute_errors(all_actual[i], all_prediction[i])
        title = summary(station, rmse, mae, nse)
        verbose > 1 and print(title)

        # # Plot Figure of Test Data  # fixme: enable plotting as needed
        # plot(
        #     graph_name, 'stgnn', station, all_epoch_days[i], all_actual[i], all_prediction[i], title,
        #     plot_diff=True, # debug
        #     dump_dir=dump_dir
        # )

        rmses.append(rmse)
        maes.append(mae)
        nses.append(nse)
        ns.append(len(all_prediction[i]))

    return rmses, maes, nses, ns, (all_actual, all_prediction, all_epoch_days)


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
