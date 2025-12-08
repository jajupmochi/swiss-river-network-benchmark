import time

import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import LstmModel, LstmEmbeddingModel
from swissrivernetwork.benchmark.util import *
from swissrivernetwork.experiment.error import compute_errors

SHOW_PLOT = False


# %% Model Inference Processes:


def run_lstm_model(
        model, df, normalizer_at, normalizer_wt, embedding_idx=None, use_embedding=False, window_len: int | None = None,
        config: dict = {}
):
    infer_time_total = time.time()

    # Set up configurations:
    use_current_x = config.get('use_current_x', True)
    extrapo_mode = config.get('extrapo_mode', None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # print(f'Using device: {next(model.parameters()).device}.\n')

    # Predict test data:
    df['air_temperature'] = normalizer_at.transform(df['air_temperature'].values.reshape(-1, 1))

    # Handle NaN in air temperature:
    assert df['air_temperature'].isna().sum() == 0, 'No NaN in Input please!'
    # exit()
    # print('[DATA PREPARATION] counted NaN values in input:', df['air_temperature'].isna().sum())
    # df['air_temperature'] = df['air_temperature'].fillna(-1)

    if window_len is None:
        dataset = SequenceFullDataset(df, embedding_idx)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
    else:
        dataset = SequenceWindowedDataset(window_len, df, embedding_idx=embedding_idx)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)

    infer_start_gpu_time = torch.cuda.Event(enable_timing=True)
    infer_end_gpu_time = torch.cuda.Event(enable_timing=True)
    infer_start_gpu_time.record()

    epoch_days = []
    prediction_norm = []
    masks = []
    actual = []
    # This can be "history_epoch_days" for LIMO mode or "full_epoch_days" for future_embedding mode:
    epoch_days_to_record, preds_to_record = [], []  # for timewise extrapolation
    with torch.no_grad():
        model.eval()
        for (t, e, x, y) in dataloader:
            t, e, x, y = t.to(device), e.to(device), x.to(device), y.to(device)
            if use_embedding:
                out = model(e, x)
            else:
                out = model(x)

            # Check for only one batch:
            if window_len is None:
                assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Remove historical inputs and masks for forecasting:
            if not use_current_x:
                future_steps = config['future_steps']
                if extrapo_mode == 'future_embedding':
                    # For future embedding extrapolation, "out" includes all predictions including historical ones:
                    epoch_days_to_record.append(t.cpu().detach().numpy())
                    preds_to_record.append(out.cpu().detach().numpy())
                    out = out[..., -future_steps:, :]
                else:
                    raise NotImplementedError('Only future_embedding extrapo_mode is supported for LSTM now!')
                    # For timewise extrapolation, "out" includes both predictions and history hidden states (saved for
                    # future use such as graphlet):
                    epoch_days_to_record.append(t[..., :-future_steps, :].cpu().detach().numpy())
                    preds_to_record.append(out[1].cpu().detach().numpy())
                    out = out[0]
                y = y[..., -future_steps:, :]
                t = t[..., -future_steps:, :]

            # Store epoch_days and prediction_norm on all days:
            epoch_days.append(t.cpu().detach().numpy())
            prediction_norm.append(out.cpu().detach().numpy())  # store normalized predictions

            # mask values:
            mask = ~torch.isnan(y)
            masks.append(mask.cpu().detach().numpy())
            # if mask.sum() == 0:
            #     continue  # skip if all is masked

            # Store values
            actual.append(y.cpu().detach().numpy())  # original

    infer_end_gpu_time.record()
    torch.cuda.synchronize()
    infer_time_gpu_total = infer_start_gpu_time.elapsed_time(infer_end_gpu_time) / 1000.0  # in seconds

    # combine arrays. We can simply combine everything together as there is only one station:
    epoch_days = np.concatenate([i.flatten() for i in epoch_days], axis=0).flatten()
    prediction_norm = np.concatenate([i.flatten() for i in prediction_norm], axis=0).flatten()
    masks = np.concatenate([i.flatten() for i in masks], axis=0).flatten()
    actual = np.concatenate([i.flatten() for i in actual], axis=0).flatten()
    if not use_current_x:
        if extrapo_mode == 'future_embedding':
            epoch_days_to_record = np.concatenate([i.flatten() for i in epoch_days_to_record], axis=0).flatten()
            preds_to_record = np.concatenate([i.flatten() for i in preds_to_record], axis=0).flatten()
        else:
            raise NotImplementedError('Only future_embedding extrapo_mode is supported for LSTM now!')
            epoch_days_to_record = np.concatenate([i.flatten() for i in epoch_days_to_record], axis=0).flatten()
            # Items in history_hiddens have shape (B, L_hist, D), we need to flatten all but the last dimension:
            preds_to_record = np.concatenate([i.reshape(-1, i.shape[-1]) for i in preds_to_record], axis=0)

    n_total_time_steps = len(prediction_norm)  # fixme: check if it is correct for win_len == inf and single models.

    # Only aggregate epoch_days_to_record and preds_to_record for timewise extrapolation for dumping:
    if not use_current_x:
        unique_epoch_days, aggregated_dict = aggregate_day_predictions(
            epoch_days_to_record,
            {
                'preds_to_record': preds_to_record,
            },
            method='longest_history'
        )
        # Store epoch_days, prediction_norm and masks on all days:
        epoch_days_to_record = unique_epoch_days
        preds_to_record = aggregated_dict['preds_to_record']
    elif window_len is not None:
        # Notice that `last` or `longest_history` do not work if dataloader is shuffled!
        unique_epoch_days, aggregated_dict = aggregate_day_predictions(
            epoch_days,
            {
                'prediction_norm': prediction_norm, 'mask': masks, 'actual': actual,
            },
            method='longest_history'
        )
        # Store epoch_days, prediction_norm and masks on all days:
        epoch_days = unique_epoch_days
        prediction_norm = aggregated_dict['prediction_norm']
        masks = aggregated_dict['mask']
        actual = aggregated_dict['actual']

    prediction = normalizer_wt.inverse_transform(prediction_norm.reshape(-1, 1)).flatten()

    if not use_current_x:
        # Create a time step array as a reference for forecasting:
        n_future_steps = config['future_steps']
        assert actual.size % n_future_steps == 0, 'Forcasting: total size must be divisible by n_future_steps.'
        n_samples = actual.size // n_future_steps
        forcast_time_steps = np.tile(np.arange(1, n_future_steps + 1), n_samples)
        forcast_time_steps = forcast_time_steps[masks]  # masked

    # Apply mask on actual and prediction:
    actual = actual[masks]
    prediction = prediction[masks]

    infer_time_total = time.time() - infer_time_total
    extra_resu = {
        'infer_time_total': infer_time_total,
        'infer_time_gpu_total': infer_time_gpu_total,
        'infer_time_per_time_step': infer_time_total / n_total_time_steps,
        'n_total_time_steps': n_total_time_steps
    }
    if not use_current_x:
        extra_resu['forcast_time_steps'] = forcast_time_steps
        if extrapo_mode == 'future_embedding':
            extra_resu['full_epoch_days'] = epoch_days_to_record
            extra_resu['full_prediction_norm'] = preds_to_record
        else:
            raise NotImplementedError('Only future_embedding extrapo_mode is supported for LSTM now!')
            # extra_resu['history_epoch_days'] = epoch_days_to_record
            # extra_resu['history_hiddens'] = preds_to_record

    return epoch_days, prediction_norm, masks, actual, prediction, extra_resu


def run_transformer_model(
        model, df, normalizer_at, normalizer_wt, embedding_idx=None, use_embedding=False, window_len: int | None = None,
        graph_name: str | None = None,
        method: str | None = None,
        config: dict = {},
):
    """
    The returned epoch_days, prediction_norm, masks, and possible full_epoch_days and full_prediction_norm include all
    time steps, while actual and prediction only include masked (valid) time steps.
    """
    infer_time_total = time.time()

    # Set up configurations:
    use_current_x = config.get('use_current_x', True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # print(f'Using device: {next(model.parameters()).device}.\n')

    # Predict test data:
    df['air_temperature'] = normalizer_at.transform(df['air_temperature'].values.reshape(-1, 1))

    # Handle NaN in air temperature:
    assert df['air_temperature'].isna().sum() == 0, 'No NaN in Input please!'
    # exit()
    # print('[DATA PREPARATION] counted NaN values in input:', df['air_temperature'].isna().sum())
    # df['air_temperature'] = df['air_temperature'].fillna(-1)

    if window_len is None:
        dataset = SequenceFullDataset(df, embedding_idx)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
    else:
        dataset = SequenceWindowedDataset(window_len, df, embedding_idx=embedding_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=get_proper_infer_batchsize(method, graph_name), shuffle=False, drop_last=False
        )

    infer_start_gpu_time = torch.cuda.Event(enable_timing=True)
    infer_end_gpu_time = torch.cuda.Event(enable_timing=True)
    infer_start_gpu_time.record()

    epoch_days = []
    prediction_norm = []
    masks = []
    actual = []
    full_epoch_days, full_prediction_norm = [], []  # for timewise extrapolation
    with torch.no_grad():
        model.eval()
        for (t, e, x, y) in dataloader:
            t, e, x, y = t.to(device), e.to(device), x.to(device), y.to(device)
            if use_embedding:
                out = model(e, x)
            else:
                out = model(x)

            # # Check for only one batch:
            # assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Remove historical inputs and masks for forecasting:
            if not use_current_x:
                full_epoch_days.append(t.cpu().detach().numpy())
                full_prediction_norm.append(out.cpu().detach().numpy())
                future_steps = config['future_steps']
                # Here "out" contains predictions for all time steps including historical ones:
                out = out[..., -future_steps:, :]
                y = y[..., -future_steps:, :]
                t = t[..., -future_steps:, :]

            # Store epoch_days and prediction_norm on all days:
            epoch_days.append(t.cpu().detach().numpy())
            prediction_norm.append(out.cpu().detach().numpy())  # store normalized predictions

            # mask values:
            mask = ~torch.isnan(y)
            masks.append(mask.cpu().detach().numpy())
            # if mask.sum() == 0:
            #     continue  # skip if all is masked

            # Store values
            actual.append(y.cpu().detach().numpy())  # original

    infer_end_gpu_time.record()
    torch.cuda.synchronize()
    infer_time_gpu_total = infer_start_gpu_time.elapsed_time(infer_end_gpu_time) / 1000.0  # in seconds

    # combine arrays. We can simply combine everything together as there is only one station:
    epoch_days = np.concatenate([i.flatten() for i in epoch_days], axis=0).flatten()
    prediction_norm = np.concatenate([i.flatten() for i in prediction_norm], axis=0).flatten()
    masks = np.concatenate([i.flatten() for i in masks], axis=0).flatten()
    actual = np.concatenate([i.flatten() for i in actual], axis=0).flatten()
    if not use_current_x:
        full_epoch_days = np.concatenate([i.flatten() for i in full_epoch_days], axis=0).flatten()
        full_prediction_norm = np.concatenate([i.flatten() for i in full_prediction_norm], axis=0).flatten()

    n_total_time_steps = len(prediction_norm)  # fixme: check if it is correct for full sequences.

    if use_current_x:
        # Notice that `last` or `longest_history` do not work if dataloader is shuffled!
        unique_epoch_days, aggregated_dict = aggregate_day_predictions(
            epoch_days,
            {
                'prediction_norm': prediction_norm, 'mask': masks, 'actual': actual,
            },
            method='longest_history'
        )
        # Store epoch_days, prediction_norm and masks on all days:
        epoch_days = unique_epoch_days
        prediction_norm = aggregated_dict['prediction_norm']
        masks = aggregated_dict['mask']
        actual = aggregated_dict['actual']
    else:  # Only aggregate full_epoch_days and full_prediction_norm for timewise extrapolation for dumping:
        unique_epoch_days, aggregated_dict = aggregate_day_predictions(
            full_epoch_days,
            {
                'prediction_norm': full_prediction_norm,
            },
            method='longest_history'
        )
        # Store epoch_days, prediction_norm and masks on all days:
        full_epoch_days = unique_epoch_days
        full_prediction_norm = aggregated_dict['prediction_norm']

    prediction = normalizer_wt.inverse_transform(prediction_norm.reshape(-1, 1)).flatten()

    if not use_current_x:
        # Create a time step array as a reference for forecasting:
        n_future_steps = config['future_steps']
        assert actual.size % n_future_steps == 0, 'Forcasting: total size must be divisible by n_future_steps.'
        n_samples = actual.size // n_future_steps
        forcast_time_steps = np.tile(np.arange(1, n_future_steps + 1), n_samples)

    # Apply mask on actual and prediction:
    actual = actual[masks]
    prediction = prediction[masks]

    infer_time_total = time.time() - infer_time_total
    extra_resu = {
        'infer_time_total': infer_time_total,
        'infer_time_gpu_total': infer_time_gpu_total,
        'infer_time_per_time_step': infer_time_total / n_total_time_steps,
        'n_total_time_steps': n_total_time_steps
    }
    if not use_current_x:
        extra_resu['forcast_time_steps'] = forcast_time_steps[masks]
        extra_resu['full_epoch_days'] = full_epoch_days
        extra_resu['full_prediction_norm'] = full_prediction_norm

    return epoch_days, prediction_norm, masks, actual, prediction, extra_resu


# %% Utility Functions:


def dump_predictions(
        graph_name: str,
        model: str,
        station: str,
        suffix: str,
        epoch_days: np.ndarray,
        prediction: np.ndarray,
        dump_dir: Path | str = 'swissrivernetwork/journal/dump'
):
    assert len(epoch_days) == len(prediction), 'not same amount'

    if prediction.ndim == 1:
        df = pd.DataFrame(
            data={
                'epoch_day': epoch_days,
                f'{station}_wt_hat': prediction
            }
        )
    # This is the case for e.g., timewise extrapolation where we dump historical hidden states, e.g., for lstm:
    elif prediction.ndim == 2:
        data_dict = {'epoch_day': epoch_days}
        for i in range(prediction.shape[1]):
            data_dict[f'{station}_wt_hat_{i}'] = prediction[:, i]
        df = pd.DataFrame(data=data_dict)
    else:
        raise ValueError('prediction must be 1D or 2D array!')

    pred_path = Path(dump_dir) / f'{graph_name}_{model}_{station}_{suffix}.csv'
    os.makedirs(pred_path.parent, exist_ok=True)
    df.to_csv(pred_path, index=False)


def fit_normalizers(df):
    normalizer_at = MinMaxScaler().fit(df['air_temperature'].values.reshape(-1, 1))
    normalizer_wt = MinMaxScaler().fit(df['water_temperature'].values.reshape(-1, 1))
    return normalizer_at, normalizer_wt


def summary(station, rmse, mae, nse):
    return f'Station {station} --\tRMSE: {rmse:.3f}\tMAE: {mae:.3f}\tNSE: {nse:.3f}'


def plot(
        graph_name, method, station, epoch_days, actual, prediction, title,
        plot_ys: bool = True,
        plot_diff: bool = False,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump'
):
    # plt.figure(figsize=(10, 6))
    # plt.plot(epoch_days, actual, label='Actual')
    # plt.plot(epoch_days, prediction, label='Prediction')
    # if plot_diff:
    #     ax1 = plt.gca()
    #     ax2 = ax1.twinx()
    #     ax2.plot(epoch_days, np.abs(np.array(actual) - np.array(prediction)), color='tab:red', label='Difference')
    #     ax2.set_ylabel('Difference')
    #     ax2.legend(loc='upper right')
    # plt.legend(loc='upper left')
    # plt.title(title.replace('\t', ' '))
    # fig_path = Path(dump_dir) / f'figures/{graph_name}_{method}_{station}.png'
    # os.makedirs(fig_path.parent, exist_ok=True)
    # plt.savefig(fig_path, dpi=300)
    # SHOW_PLOT and plt.show()
    # plt.close()
    plt.figure(figsize=(20, 6))
    # plt.figure(figsize=(10, 6))
    if plot_ys:
        plt.plot(epoch_days, actual, label='Actual')
        plt.plot(epoch_days, prediction, label='Prediction')
    if plot_diff:
        plt.plot(
            epoch_days, np.abs(np.array(actual) - np.array(prediction)), color='tab:red', linestyle='-', linewidth=0.5,
            label='Difference'
        )
        # Plot sequence separators as vertical dashed lines:
        day_diff = np.diff(epoch_days)
        breaks = day_diff != 1
        breaks = np.concatenate((np.array([True]), breaks))
        seque_id = np.cumsum(breaks)
        seques = np.argwhere(breaks).flatten()
        unique_ids, counts = np.unique(seque_id, return_counts=True)
        seque_lengths = counts
        starts = epoch_days[seques]
        ends = epoch_days[seques + seque_lengths - 1]
        colors = plt.cm.get_cmap('tab20', len(starts))
        fill_params = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            plt.axvline(x=s, color=colors(i), linestyle='--', alpha=0.5, linewidth=0.5)
            plt.axvline(x=e, color=colors(i), linestyle='--', alpha=0.5, linewidth=0.5)
            fill_params.append({'s': s, 'e': e, 'color': colors(i), 'ylim': plt.ylim()})
        max_ylim = (min([p['ylim'][0] for p in fill_params]), max([p['ylim'][1] for p in fill_params]))
        for param in fill_params:
            plt.fill_betweenx(max_ylim, param['s'], param['e'], color=param['color'], alpha=0.1)
    plt.legend()
    plt.title(title.replace('\t', ' '))
    fig_path = Path(dump_dir) / f'figures/{graph_name}_{method}_{station}.png'
    os.makedirs(fig_path.parent, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    SHOW_PLOT and plt.show()
    plt.close()


def compute_metrics(
        actual: np.ndarray, prediction: np.ndarray, extra_resu: dict, config: dict, station: str,
        verbose: int
):  # -> tuple[Any, floating[Any], float | ndarray[tuple[Any, ...], dtype[Any]] | Any, str, int]:
    # Compute errors:
    if config.get('use_current_x', True):
        rmse, mae, nse = compute_errors(actual, prediction)
        title = summary(station, rmse, mae, nse)
        verbose > 1 and print(title)
        len_pred = len(prediction)
    else:
        # For forecasting, compute errors for each forecast step:
        n_future_steps = config['future_steps']
        rmse = []
        mae = []
        nse = []
        len_pred = []
        for step in range(n_future_steps):
            mask_step = extra_resu['forcast_time_steps'] == (step + 1)
            actual_step = actual[mask_step]
            prediction_step = prediction[mask_step]
            rmse_step, mae_step, nse_step = compute_errors(actual_step, prediction_step)
            rmse.append(rmse_step)
            mae.append(mae_step)
            nse.append(nse_step)
            # verbose > 1 and print(
            #     f'Station {station} -- Forecast Step {step + 1} --\tRMSE: {rmse_step:.3f}\tMAE: {mae_step:.3f}\tNSE: {nse_step:.3f}'
            # )
            len_pred.append(len(prediction_step))
        title = f'Station {station} -- Overall Forecast --\tRMSE: {np.mean(rmse):.3f}\tMAE: {np.mean(mae):.3f}\tNSE: {np.mean(nse):.3f}'
        verbose > 1 and print(title)
    return rmse, mae, nse, len_pred, title


# %% Main test processes:


def test_graphlet(
        graph_name, station, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump',
        predict_dump_dir: Path | str | None = None,
        config: dict = {},
        verbose: int = 2
):
    # load normalizers
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)

    # prepare test data (using neighbors)
    num_hops = 1
    neighs = extract_neighbors(graph_name, station, num_hops)
    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    predict_dump_dir = dump_dir if predict_dump_dir is None else predict_dump_dir
    df_neighs = [read_csv_prediction_test(graph_name, 'lstm', neigh, predict_dump_dir=predict_dump_dir) for neigh in
                 neighs]
    df = merge_graphlet_dfs(df, df_neighs)

    # run lstm model on it
    epoch_days, prediction_norm, mask, actual, prediction, extra_resu = run_lstm_model(
        model, df, normalizer_at, normalizer_wt, use_embedding=False, window_len=window_len, config=config
    )

    # Compute errors:
    rmse, mae, nse, len_pred, title = compute_metrics(actual, prediction, extra_resu, config, station, verbose)

    # Plot Figure of Test Data
    if config.get('use_current_x', True):  # fixme: test
        plot(
            graph_name, 'graphlet', station, epoch_days[mask], actual, prediction, title,
            plot_diff=True,  # debug
            dump_dir=dump_dir
        )

    return rmse, mae, nse, len_pred, (actual, prediction, epoch_days[mask]), extra_resu


def test_transformer_graphlet(
        graph_name, station, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump',
        predict_dump_dir: Path | str | None = None,
        method: str | None = None,
        config: dict = {},
        verbose: int = 2
):
    # load normalizers
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)

    # prepare test data (using neighbors)
    num_hops = 1
    neighs = extract_neighbors(graph_name, station, num_hops)
    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    predict_dump_dir = dump_dir if predict_dump_dir is None else predict_dump_dir
    df_neighs = [read_csv_prediction_test(graph_name, 'transformer', neigh, predict_dump_dir=predict_dump_dir)
                 for neigh in neighs]
    df = merge_graphlet_dfs(df, df_neighs)

    # run lstm model on it
    epoch_days, prediction_norm, mask, actual, prediction, extra_resu = run_transformer_model(
        model, df, normalizer_at, normalizer_wt, use_embedding=False, window_len=window_len, config=config,
        graph_name=graph_name, method=method
    )

    # Compute errors:
    rmse, mae, nse, len_pred, title = compute_metrics(actual, prediction, extra_resu, config, station, verbose)

    # create graphs
    if config.get('use_current_x', True):  # fixme: test
        plot(
            graph_name, 'transformer_graphlet', station, epoch_days[mask], actual, prediction, title,
            plot_diff=True,  # debug
            dump_dir=dump_dir
        )
    return rmse, mae, nse, len_pred, (actual, prediction, epoch_days[mask]), extra_resu


def test_lstm(
        graph_name, station, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump',
        predict_dump_dir: Path | str | None = None,
        config: dict = {},
        verbose: int = 2
):
    model_name = trim_lstm_model_name(model.__class__.__name__)

    # Prepare normalizers:
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)
    # What if we load normalizers from checkpoint?!

    # Prepare test data:
    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    epoch_days, prediction_norm, mask, actual, prediction, extra_resu = run_lstm_model(
        model, df, normalizer_at, normalizer_wt, use_embedding=False, window_len=window_len, config=config
    )
    dump_predictions(
        graph_name, model_name, station, 'test',
        extra_resu.get('full_epoch_days', epoch_days),
        extra_resu.get('full_prediction_norm', prediction_norm),
        # extra_resu.get('history_epoch_days', epoch_days),
        # extra_resu.get('history_hiddens', prediction_norm),
        dump_dir=(dump_dir if predict_dump_dir is None else predict_dump_dir)
    )

    # Run on Train data as well:
    (
        epoch_days_train, prediction_norm_train, mask_train, actual_train, prediction_train, extra_resu_train
    ) = run_lstm_model(
        model, df_train, normalizer_at, normalizer_wt, use_embedding=False, window_len=window_len, config=config
    )  # do not denormalize
    dump_predictions(
        graph_name, model_name, station, 'train',
        extra_resu_train.get('full_epoch_days', epoch_days_train),
        extra_resu_train.get('full_prediction_norm', prediction_norm_train),
        # extra_resu_train.get('history_epoch_days', epoch_days_train),
        # extra_resu_train.get('history_hiddens', prediction_norm_train),
        dump_dir=(dump_dir if predict_dump_dir is None else predict_dump_dir)
    )

    # Compute errors:
    rmse, mae, nse, len_pred, title = compute_metrics(actual, prediction, extra_resu, config, station, verbose)

    # Plot Figure of Test Data
    if config.get('use_current_x', True):  # fixme: test
        plot(
            graph_name, 'lstm', station, epoch_days[mask], actual, prediction, title,
            plot_diff=True,  # debug
            dump_dir=dump_dir
        )

    return rmse, mae, nse, len_pred, (actual, prediction, epoch_days[mask]), extra_resu


def test_transformer(
        graph_name, station, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump',
        predict_dump_dir: Path | str | None = None,
        config: dict = {},
        verbose: int = 2
):
    model_name = model.__class__.__name__.lower().removesuffix('model')

    # Prepare normalizers:
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)
    # What if we load normalizers from checkpoint?!

    # Prepare test data:
    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    epoch_days, prediction_norm, mask, actual, prediction, extra_resu = run_transformer_model(
        model, df, normalizer_at, normalizer_wt, use_embedding=False, window_len=window_len, config=config
    )
    dump_predictions(
        graph_name, model_name, station, 'test',
        extra_resu.get('full_epoch_days', epoch_days),
        extra_resu.get('full_prediction_norm', prediction_norm),
        dump_dir=(dump_dir if predict_dump_dir is None else predict_dump_dir)
    )

    # Run on Train data as well:
    (
        epoch_days_train, prediction_norm_train, mask_train, actual_train, prediction_train, extra_resu_train
    ) = run_transformer_model(
        model, df_train, normalizer_at, normalizer_wt, use_embedding=False, window_len=window_len, config=config
    )  # do not denormalize
    dump_predictions(
        graph_name, model_name, station, 'train',
        extra_resu_train.get('full_epoch_days', epoch_days_train),
        extra_resu_train.get('full_prediction_norm', prediction_norm_train),
        dump_dir=(dump_dir if predict_dump_dir is None else predict_dump_dir)
    )

    # Compute errors:
    rmse, mae, nse, len_pred, title = compute_metrics(actual, prediction, extra_resu, config, station, verbose)

    # Plot Figure of Test Data
    if config.get('use_current_x', True):  # fixme: test
        plot(
            graph_name, 'transformer', station, epoch_days[mask], actual, prediction, title,
            plot_diff=True,  # debug
            dump_dir=dump_dir
        )

    return rmse, mae, nse, len_pred, (actual, prediction, epoch_days[mask]), extra_resu


def test_lstm_embedding(
        graph_name, station, i, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump',
        config: dict = {},
        verbose: int = 2
):
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)

    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    epoch_days, prediction_norm, mask, actual, prediction, extra_resu = run_lstm_model(
        model, df, normalizer_at, normalizer_wt, embedding_idx=i, use_embedding=True, window_len=window_len,
        config=config
    )

    # Compute errors:
    rmse, mae, nse, len_pred, title = compute_metrics(actual, prediction, extra_resu, config, station, verbose)

    # Plot Figure of Test Data
    if config.get('use_current_x', True):  # fixme: test
        plot(
            graph_name, 'lstm_embedding', station, epoch_days[mask], actual, prediction, title,
            plot_diff=True,  # debug
            dump_dir=dump_dir
        )

    return rmse, mae, nse, len_pred, (actual, prediction, epoch_days[mask]), extra_resu


def test_transformer_embedding(
        graph_name, station, i, model, window_len: int | None = None,
        dump_dir: Path | str = 'swissrivernetwork/benckmark/dump',
        config: dict = {},
        verbose: int = 2
):
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)

    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    epoch_days, prediction_norm, mask, actual, prediction, extra_resu = run_transformer_model(
        model, df, normalizer_at, normalizer_wt, embedding_idx=i, use_embedding=True, window_len=window_len,
        config=config
    )

    # Compute errors:
    rmse, mae, nse, len_pred, title = compute_metrics(actual, prediction, extra_resu, config, station, verbose)

    # Plot Figure of Test Data
    if config.get('use_current_x', True):  # fixme: test
        plot(
            graph_name, 'transformer_embedding', station, epoch_days[mask], actual, prediction, title,
            plot_diff=True,  # debug
            dump_dir=dump_dir
        )

    return rmse, mae, nse, len_pred, (actual, prediction, epoch_days[mask]), extra_resu


if __name__ == '__main__':

    graph_name = 'swiss-1990'
    station = '2091'
    method = 'lstm_embedding'

    for i, s in enumerate(read_stations(graph_name)):
        if s == station:
            break  # set i

    # Select best model:
    if 'lstm' == method:
        # LSTM Model    
        analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/{method}_{station}-2025-05-13_13-32-54')
        input_size = 1
    if 'graphlet' == method:
        analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/{method}_{station}-2025-05-13_16-59-26')
        input_size = 1 + len(extract_neighbors(graph_name, station, 1))
    if 'lstm_embedding' == method:
        analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/{method}-2025-05-20_11-35-12')
        input_size = 1

    # Get best trial and load model:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")

    if 'lstm_embedding' == method:
        num_embeddings = len(read_stations(graph_name))
        embedding_size = best_config['embedding_size']

        # Create Model
    if 'lstm' == method or 'graphlet' == method:
        model = LstmModel(input_size, best_config['hidden_size'], best_config['num_layers'])
    if 'lstm_embedding' == method:
        model = LstmEmbeddingModel(
            input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers']
        )
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))

    if 'lstm' == method:
        test_lstm(graph_name, station, model)
    if 'graphlet' == method:
        test_graphlet(graph_name, station, model)
    if 'lstm_embedding' == method:
        test_lstm_embedding(graph_name, station, i, model)
    plt.show()
