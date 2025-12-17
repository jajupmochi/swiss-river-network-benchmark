import time

from ray.tune import ExperimentAnalysis
from sklearn.preprocessing import MinMaxScaler

from swissrivernetwork.benchmark.dataset import STGNNSequenceFullDataset, STGNNSequenceWindowedDataset
from swissrivernetwork.benchmark.dataset import read_stations, read_csv_train, read_csv_test
# from swissrivernetwork.experiment.error import Error
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.test_isolated_station import compute_metrics
from swissrivernetwork.benchmark.util import *


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
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump', method: str | None = None,
        config: dict = {},
        verbose: int = 2
):
    infer_time_total = time.time()

    # Set up configurations:
    use_current_x = config.get('use_current_x', True)
    extrapo_mode = config.get('extrapo_mode', None)
    noise_settings = {k: v for k, v in config.items() if k.startswith('noise_')}

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
    if window_len is None:
        dataset = STGNNSequenceFullDataset(df, stations, **noise_settings)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)  # fixme: drop_last? and the other locations?
    else:
        dataset = STGNNSequenceWindowedDataset(window_len, df, stations, **noise_settings)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=get_proper_infer_batchsize(method, graph_name), shuffle=False, drop_last=False
        )

    infer_start_gpu_time = torch.cuda.Event(enable_timing=True)
    infer_end_gpu_time = torch.cuda.Event(enable_timing=True)
    infer_start_gpu_time.record()

    # Compute this:
    # epoch_days, prediction_norm, mask, actual, prediction
    epoch_days, prediction_norm, actual, masks = [], [], [], []
    # This can be "history_epoch_days" for LIMO mode or "full_epoch_days" for future_embedding mode:
    epoch_days_to_record, preds_to_record = [], []  # for timewise extrapolation
    with torch.no_grad():
        model.eval()
        for (t, e, x, y) in dataloader:
            t, e, x, y = t.to(device), e.to(device), x.to(device), y.to(device)

            # Run Model
            out = model(x, edges)

            # # Check for only one batch:
            # assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Remove historical inputs and masks for forecasting:
            if not use_current_x:
                future_steps = config['future_steps']
                # For future embedding extrapolation, "out" includes all predictions including historical ones:
                epoch_days_to_record.append(t.cpu().detach().numpy())
                preds_to_record.append(out.cpu().detach().numpy())
                out = out[..., -future_steps:, :]
                # todo: this may be incorrect if using limo mode
                # if extrapo_mode == 'future_embedding':
                # else:
                #     raise NotImplementedError('Only future_embedding extrapo_mode is supported for LSTM now!')
                #     # For timewise extrapolation, "out" includes both predictions and history hidden states (saved for
                #     # future use such as graphlet):
                #     epoch_days_to_record.append(t[..., :-future_steps, :].cpu().detach().numpy())
                #     preds_to_record.append(out[1].cpu().detach().numpy())
                #     out = out[0]
                y = y[..., -future_steps:, :]
                t = t[..., -future_steps:, :]

            # Split the predictions per station:
            mask = ~torch.isnan(y)
            # y = y[mask]
            # out_masked = out[mask]

            epoch_days.append(t.cpu().detach().numpy())
            prediction_norm.append(out.cpu().detach().numpy())
            actual.append(y.cpu().detach().numpy())
            masks.append(mask.cpu().detach().numpy())

    infer_end_gpu_time.record()
    torch.cuda.synchronize()
    infer_time_gpu_total = infer_start_gpu_time.elapsed_time(infer_end_gpu_time) / 1000.0  # in seconds

    # Shape before: [[B, n_stations, seq_len / win_len, 1], ...]
    # Shape after:  [[n_stations, seq_len / win_len, 1], ...] (n_subsequences)
    epoch_days = [i for t in epoch_days for i in t]
    prediction_norm = [i for p in prediction_norm for i in p]
    actual = [i for a in actual for i in a]
    masks = [i for m in masks for i in m]
    if not use_current_x:
        epoch_days_to_record = [i for t in epoch_days_to_record for i in t]
        preds_to_record = [i for p in preds_to_record for i in p]
        # todo: this may be incorrect if using limo mode
        # if extrapo_mode == 'future_embedding':
        # else:
        #     raise NotImplementedError('Only future_embedding extrapo_mode is supported for LSTM now!')
        #     epoch_days_to_record = np.concatenate([i.flatten() for i in epoch_days_to_record], axis=0).flatten()
        #     # Items in history_hiddens have shape (B, L_hist, D), we need to flatten all but the last dimension:
        #     preds_to_record = np.concatenate([i.reshape(-1, i.shape[-1]) for i in preds_to_record], axis=0)

    # fixme: check if it is correct for full sequences:
    n_total_time_steps = np.sum([p.shape[0] * p.shape[1] for p in prediction_norm])

    # Combine arrays:
    all_epoch_days, all_actual, all_prediction, all_masks = [], [], [], []
    all_epoch_days_to_record, all_preds_to_record, all_forcast_time_steps = [], [], []  # for extrapolation only
    for i, station in enumerate(stations):
        station_epoch_days = np.concatenate([e[i] for e in epoch_days], axis=0).flatten()
        station_prediction_norm = np.concatenate([p[i] for p in prediction_norm], axis=0).flatten()
        station_masks = np.concatenate([m[i] for m in masks], axis=0).flatten()
        station_actual = np.concatenate([a[i] for a in actual], axis=0).flatten()
        if not use_current_x:
            # and extrapo_mode == 'future_embedding': todo: this may be incorrect if using limo mode
            station_epoch_days_to_record = np.concatenate([e[i] for e in epoch_days_to_record], axis=0).flatten()
            station_preds_to_record = np.concatenate([p[i] for p in preds_to_record], axis=0).flatten()

        # Only aggregate epoch_days_to_record and preds_to_record for timewise extrapolation for dumping:
        if not use_current_x:
            station_epoch_days_to_record, aggregated_dict = aggregate_day_predictions(
                station_epoch_days_to_record,
                {
                    'preds_to_record': station_preds_to_record,
                },
                method='longest_history'
            )
            station_preds_to_record = aggregated_dict['preds_to_record']
            all_epoch_days_to_record.append(station_epoch_days_to_record)
            all_preds_to_record.append(station_preds_to_record)

            # Create a time step array as a reference for forecasting:
            n_future_steps = config['future_steps']
            assert station_actual.size % n_future_steps == 0, 'Forcasting: total size must be divisible by n_future_steps.'
            n_samples = station_actual.size // n_future_steps
            forcast_time_steps = np.tile(np.arange(1, n_future_steps + 1), n_samples)
            all_forcast_time_steps.append(forcast_time_steps[station_masks])  # masked

        elif check_is_aggregation_needed(dataloader):
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
            station_prediction_norm = aggregated_dict['prediction_norm']
            station_actual = aggregated_dict['actual']

        station_prediction_norm = station_prediction_norm[station_masks]
        station_actual = station_actual[station_masks]

        # Denormalize predictions
        station_prediction = normalizers[f'{station}_wt'].inverse_transform(
            station_prediction_norm.reshape(-1, 1)
        ).flatten()

        all_epoch_days.append(station_epoch_days[station_masks])  # masked
        all_actual.append(station_actual)  # masked
        all_prediction.append(station_prediction)  # masked

    infer_time_total = time.time() - infer_time_total
    extra_resu = {
        'infer_time_total': [infer_time_total / n_stations] * n_stations,
        'infer_time_gpu_total': [infer_time_gpu_total / n_stations] * n_stations,
        'infer_time_per_time_step': [infer_time_total / n_total_time_steps] * n_stations,
        'n_total_time_steps': [n_total_time_steps / n_stations] * n_stations,
    }
    if not use_current_x:
        extra_resu['forcast_time_steps'] = all_forcast_time_steps
        extra_resu['full_epoch_days'] = all_epoch_days_to_record
        extra_resu['full_prediction_norm'] = all_preds_to_record
        #  todo: this may be incorrect if using limo mode
        # if extrapo_mode == 'future_embedding':
        # else:
        #     raise NotImplementedError('Only future_embedding extrapo_mode is supported for LSTM now!')
        #     # extra_resu['history_epoch_days'] = epoch_days_to_record
        #     # extra_resu['history_hiddens'] = preds_to_record

    # Compute errors
    rmses = []
    maes = []
    nses = []
    ns = []
    for i, station in enumerate(stations):
        cur_extra_resu = {k: v[i] for k, v in extra_resu.items()}
        rmse, mae, nse, len_pred, title = compute_metrics(
            all_actual[i], all_prediction[i], cur_extra_resu, config, station, verbose
        )

        # # Plot Figure of Test Data  # fixme: enable plotting as needed
        # plot(
        #     graph_name, 'stgnn', station, all_epoch_days[i], all_actual[i], all_prediction[i], title,
        #     plot_diff=True, # debug
        #     dump_dir=dump_dir
        # )

        rmses.append(rmse)
        maes.append(mae)
        nses.append(nse)
        ns.append(len_pred)

    return rmses, maes, nses, ns, (all_actual, all_prediction, all_epoch_days), extra_resu


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
