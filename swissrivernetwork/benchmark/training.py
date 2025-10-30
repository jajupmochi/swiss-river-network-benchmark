import sys
import tempfile
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import wandb
from benedict import benedict
from ray.tune import Checkpoint, report
from sklearn.base import BaseEstimator
from torchinfo import summary
from tqdm import tqdm

from swissrivernetwork.benchmark.util import save, aggregate_day_predictions, safe_get_ray_trial_id
from swissrivernetwork.experiment.error import Error
from swissrivernetwork.util.scaler import StationSplitScaler

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green


def training_loop(
        config, dataloader_train, dataloader_valid, model, n_valid, use_embedding, edges=None,
        normalizer_at: list[BaseEstimator] | StationSplitScaler | dict[str, BaseEstimator] = None,
        normalizer_wt: list[BaseEstimator] | StationSplitScaler | dict[str, BaseEstimator] = None,
        wandb_project: str | None = 'swissrivernetwork', settings: benedict = benedict({}),
        verbose: int = 2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if edges is not None:
        edges = edges.to(device)

    if verbose >= 2:
        # Print data info:
        print(f'{INFO_TAG}Training sample size: {len(dataloader_train.dataset)}.')
        if isinstance(dataloader_train.dataset, torch.utils.data.ConcatDataset):
            for ds in dataloader_train.dataset.datasets:
                print(f'  - Station {ds.embedding_idx}: {len(ds)} samples, sequence lengths: {ds.sequence_lengths}')
        else:
            print(f'  Sequence lengths: {dataloader_train.dataset.sequence_lengths}')
        print(f'{INFO_TAG}Validation samples size: {len(dataloader_valid.dataset)}.')
        if isinstance(dataloader_valid.dataset, torch.utils.data.ConcatDataset):
            for ds in dataloader_valid.dataset.datasets:
                print(f'  - Station {ds.embedding_idx}: {len(ds)} samples, sequence lengths: {ds.sequence_lengths}')
        else:
            print(f'  Sequence lengths: {dataloader_valid.dataset.sequence_lengths}')

        # Print model summary:
        print(f'{INFO_TAG}Model Summary:')
        summary(model)
        print(f'{INFO_TAG}GPU available: {torch.cuda.is_available()}.')
        print(f'{INFO_TAG}Using device: {next(model.parameters()).device}.\n')

    # Login via command line: `wandb login <your_api_key>`
    disable_wandb = settings.get('dev_run', False) or wandb_project is None or not settings.get('enable_wandb', False)
    name = f'{config["graph_name"]}_{model.__class__.__name__}'
    ray_trial_id = safe_get_ray_trial_id()
    if ray_trial_id:
        name += f'_{ray_trial_id}'
    wandb.init(
        project=wandb_project,
        name=name,
        config=config,  # save hyperparameters
        mode='disabled' if disable_wandb else None,
        # finish_previous=True  # each Ray Tune trial should create a separate wandb run automatically
    )

    try:
        # Run the Training loop on the Model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()
        validation_criterion = nn.MSELoss(reduction='mean')  # weight all samples equally

        for epoch in range(config['epochs']):
            metrics_to_report = {}

            model.train()
            losses = []

            # halo = EpochHalo(
            #     total_steps=len(dataloader_train), epoch=epoch + 1, total_epochs=config['epochs'],
            #     spinner_style='dots', prefix='Training',
            #     start_color=(255, 165, 0), end_color=(0, 0, 255)
            # )

            if verbose >= 2:
                iterator = tqdm(
                    dataloader_train, desc=f'Epoch {epoch + 1}/{config["epochs"]}', file=sys.stdout, colour='green'
                )
            else:
                iterator = dataloader_train

            # The training loss MSE is computed for each sub-sequence, then averaged over all sub-sequences.
            # When window_len remains the same and no masking is applied, this is equivalent to computing
            # the MSE over all samples.
            # Notice that shuffling is applied for training, so the order of samples is different from
            # the original time order.
            for step, content in enumerate(iterator):
                if len(content) == 4:  # fixme: revise other places too. search for "e, x, y"
                    _, e, x, y = content
                    time_masks, pad_masks = None, None
                    kwargs = {}
                elif len(content) == 5:
                    _, e, x, y, time_masks = content
                    time_masks, pad_masks = time_masks.to(device), None
                    kwargs = {'time_masks': time_masks}
                elif len(content) == 6:
                    _, e, x, y, time_masks, pad_masks = content
                    time_masks, pad_masks = time_masks.to(device), pad_masks.to(device)
                    kwargs = {'time_masks': time_masks, 'pad_masks': pad_masks}
                else:
                    raise ValueError('The dataloader must return (t, e, x, y, [time_masks], [pad_masks])!')

                e, x, y = e.to(device), x.to(device), y.to(device)
                optimizer.zero_grad()
                if edges is not None:
                    out = model(x, edges, **kwargs)
                elif use_embedding:
                    out = model(e, x, **kwargs)
                else:
                    out = model(x, **kwargs)
                mask = ~torch.isnan(y)  # mask NaNs.
                # Mask also the padded time steps. This was not done in the Transformer outputs:
                if time_masks is not None:
                    mask &= (~time_masks).unsqueeze(-1)  # True for valid values
                if pad_masks is not None:
                    mask &= (~pad_masks).unsqueeze(-1)  # True for valid values

                # Shapes of ``out``, ``y``, ``mask`` are as follows:
                # - For SeqWindowed[Masked]Dataset: [B, win_len, 1]
                # - For STGNNSequenceWindowed[Masked}Dataset: [B, n_stations, win_len, 1].
                loss = criterion(out[mask], y[mask])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if verbose >= 2:
                    iterator.set_postfix(loss=loss.item())
                # halo.update(step, loss.item())

            train_loss = sum(losses) / len(losses)
            metrics_to_report['train_loss'] = train_loss

            # halo.succeed(f'Epoch {epoch + 1}: Avg Loss = {sum(losses) / len(losses):.4f}.')
            # print(f'Epoch {epoch + 1}: Avg Train Loss = {train_loss:.4f}.')

            model.eval()
            epoch_days, preds, targets, masks = [], [], [], []
            with torch.no_grad():
                if verbose >= 2:
                    iterator_valid = tqdm(
                        dataloader_valid, desc=f'  Validation', file=sys.stdout,
                        colour='blue'
                    )
                else:
                    iterator_valid = dataloader_valid

                for content in iterator_valid:
                    if len(content) == 4:
                        t, e, x, y = content
                        time_masks, pad_masks = None, None
                        kwargs = {}
                    elif len(content) == 5:
                        t, e, x, y, time_masks = content
                        time_masks, pad_masks = time_masks.to(device), None
                        kwargs = {'time_masks': time_masks}
                    elif len(content) == 6:
                        t, e, x, y, time_masks, pad_masks = content
                        time_masks, pad_masks = time_masks.to(device), pad_masks.to(device)
                        kwargs = {'time_masks': time_masks, 'pad_masks': pad_masks}
                    else:
                        raise ValueError('The dataloader must return (t, e, x, y, [time_masks], [pad_masks])!')

                    t, e, x, y = t.to(device), e.to(device), x.to(device), y.to(device)
                    if edges is not None:
                        out = model(x, edges, **kwargs)
                    elif use_embedding:
                        out = model(e, x, **kwargs)
                    else:
                        out = model(x, **kwargs)
                    mask = ~torch.isnan(y)  # mask NaNs. True for valid values.
                    if time_masks is not None:
                        # also mask the padded time steps. This was not done in the Transformer outputs:
                        mask &= (~time_masks).unsqueeze(-1)
                    if pad_masks is not None:
                        mask &= (~pad_masks).unsqueeze(-1)

                    # Record everything (not masked) for possible aggregation later:
                    epoch_days.append(t)
                    masks.append(mask)
                    preds.append(out)  # Both preds and targets are normalized
                    targets.append(y)

            validation_mse, validation_ave_rmse, validation_rmse = compute_all_metrics(
                epoch_days, masks, preds, targets, dataloader_valid, normalizer_wt, validation_criterion,
                is_stg=dataloader_valid.dataset.__class__.__name__.startswith('STGNN')  # todo: startswith or include?
            )

            # Log everything:
            metrics_to_report['validation_mse'] = validation_mse
            metrics_to_report['validation_ave_rmse'] = validation_ave_rmse
            metrics_to_report['validation_rmse'] = validation_rmse

            # Register Ray Checkpoint
            checkpoint_dir = tempfile.mkdtemp()
            save(model.state_dict(), checkpoint_dir, f'{settings.get("method", "model_epoch")}_{epoch + 1}.pth')
            # save(normalizer_at, checkpoint_dir, 'normalizer_at.pth')
            # save(normalizer_wt, checkpoint_dir, 'normalizer_wt.pth')
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # report epoch loss. The reported metrics for Ray Tune must be a number, it can not be a numpy.ndarray!!
            if ray_trial_id:
                report(metrics_to_report, checkpoint=checkpoint)
            # print(
            #     f'{SUCCESS_TAG}I have reported to Ray!, where validation_mse is {metrics_to_report["validation_mse"]}'
            #     f' of type {type(metrics_to_report["validation_mse"])}.'
            # )  # debug

            for k, v in metrics_to_report.items():
                wandb.log({'epoch': epoch + 1, k: v})

            metric_str = ', '.join([f'{k} = {v:.5f}' for k, v in metrics_to_report.items()])
            print(f'{INFO_TAG}End of Epoch {epoch + 1}: {metric_str}.')

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f'{ISSUE_TAG}WARNING: ran out of memory, skipping this trial!')
            report(done=True, status="OOM")
        else:
            raise

    wandb.finish()


def compute_all_metrics(
        epoch_days: list[torch.Tensor],
        masks: list[torch.Tensor],
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_valid: torch.utils.data.DataLoader,
        normalizer_wt: list[BaseEstimator] | StationSplitScaler,
        validation_criterion: nn.Module,
        is_stg: bool
):
    # Shapes of items in input lists ``epoch_days``, ``masks``, ``preds``, ``targets`` are as follows:
    # - For SeqFull[Masked]Dataset: [B, seq_len, 1]. ``seq_len`` may vary.
    # - For SeqWindowed[Masked]Dataset: [B, win_len, 1]. ``win_len`` is fixed.
    # - For STGNNSequenceFullDataset: [B, n_stations, seq_len, 1]. ``seq_len`` may vary.
    # - For STGNNSequenceWindowedDataset: [B, n_stations, win_len, 1]. ``win_len`` is fixed.
    epoch_days = [i for t in epoch_days for i in t]  # Can not torch.cat here as sequence lengths may vary
    masks = [i for m in masks for i in m]
    preds = [i for p in preds for i in p]
    targets = [i for tg in targets for i in tg]

    if is_stg:
        return compute_all_metrics_stg(
            epoch_days, masks, preds, targets, dataloader_valid, normalizer_wt, validation_criterion
        )
    else:
        return compute_all_metrics_isolated_station(
            epoch_days, masks, preds, targets, dataloader_valid, normalizer_wt, validation_criterion
        )


def compute_all_metrics_stg(
        epoch_days: list[torch.Tensor],
        masks: list[torch.Tensor],
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_valid: torch.utils.data.DataLoader,
        normalizer_wt: list[BaseEstimator] | StationSplitScaler,
        validation_criterion: nn.Module,
):
    def station_data_extractor(input, iter_idx):
        return input[:, iter_idx, :].flatten()


    # For STGNNs. Shapes after concatenation are all [n_samples_per_station / total_days, n_stations, 1]:
    epoch_days = torch.concat([i.transpose(0, 1) for i in epoch_days], dim=0)
    masks = torch.concat([i.transpose(0, 1) for i in masks], dim=0)
    preds = torch.concat([i.transpose(0, 1) for i in preds], dim=0)
    targets = torch.concat([i.transpose(0, 1) for i in targets], dim=0)

    station_iterator = zip(dataloader_valid.dataset.stations, range(preds.shape[1]))

    metrics = compute_all_metrics_unified(
        epoch_days, masks, preds, targets, dataloader_valid, normalizer_wt, validation_criterion,
        station_iterator=station_iterator,
        station_data_extractor=station_data_extractor
    )
    return metrics


def compute_all_metrics_isolated_station(
        epoch_days: list[torch.Tensor],
        masks: list[torch.Tensor],
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_valid: torch.utils.data.DataLoader,
        normalizer_wt: list[BaseEstimator] | StationSplitScaler,
        validation_criterion: nn.Module,
):
    def station_data_extractor(input, iter_idx):
        start, end = iter_idx
        return torch.cat(input[start:end], dim=0).flatten()


    def construct_station_iterator():
        if hasattr(dataloader_valid.dataset, 'cumulative_sizes'):
            # ConcatDataset, e.g., for lstm_embedding and transformer_embedding methods:
            assert len(dataloader_valid.dataset.datasets) == len(dataloader_valid.dataset.cumulative_sizes), (
                'Mismatch between number of stations and cumulative sizes!'
            )
            station_names = [ds.name for ds in dataloader_valid.dataset.datasets]
            cumulative_sizes = [0] + dataloader_valid.dataset.cumulative_sizes  # cumulate # of sub-sequences per station
            return zip(station_names, zip(cumulative_sizes[:-1], cumulative_sizes[1:]))
        else:
            # Single station dataset, e.g., for lstm and transformer methods:
            station_name = dataloader_valid.dataset.name
            return [(station_name, (0, len(dataloader_valid.dataset)))]


    station_iterator = construct_station_iterator()

    metrics = compute_all_metrics_unified(
        epoch_days, masks, preds, targets, dataloader_valid, normalizer_wt, validation_criterion,
        station_iterator=station_iterator,
        station_data_extractor=station_data_extractor
    )
    if hasattr(dataloader_valid.dataset, 'cumulative_sizes'):
        print(f'{INFO_TAG}cumulative_sizes: {[0] + dataloader_valid.dataset.cumulative_sizes}')
    return metrics


def compute_all_metrics_unified(
        epoch_days: list[torch.Tensor],
        masks: list[torch.Tensor],
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        dataloader_valid: torch.utils.data.DataLoader,
        normalizer_wt: list[BaseEstimator] | StationSplitScaler,
        validation_criterion: nn.Module,
        station_iterator: Iterable,
        station_data_extractor: Callable,

):
    """
    A unified function to compute validation metrics for both isolated stations and STGNNs.

    Args:
        epoch_days, masks, preds, targets (list[torch.Tensor]): Lists of tensors containing epoch days, masks, predictions,
            and targets for the validation set.
            Shapes of items in lists ``epoch_days``, ``masks``, ``preds``, ``targets`` are as follows:
            - For SeqFull[Masked]Dataset: [seq_len, 1]. ``seq_len`` may vary.
            - For SeqWindowed[Masked]Dataset: [win_len, 1]. ``win_len`` is fixed.
            - For STGNNSequenceFullDataset: [total_days, n_stations, 1]. ``total_days`` is the same for each station.
            - For STGNNSequenceWindowedDataset: [n_samples_per_station, n_stations, 1]. ``n_samples_per_station`` is the
                same for each station.

        station_iterator (Iterable): An iterable that yields station indices.

        station_data_extractor (Callable): A function that takes in epoch_days, masks, preds, or targets and returns
            the corresponding data for the given station index.
    """
    # preds, targets, masks, etc. are lists of tensors in shape [B, sub_seq_len, 1].
    # For SequenceFullDataset, each tensor corresponds to one station;
    # For SequenceWindowedDataset, each tensor corresponds to one sub-sequence of a station.
    # - For valid loss, we concatenate all samples and compute the total mse over all samples. **It is the same as
    #   the training loss** when window_len keeps unchanged and no masking is applied for training.
    #   Notice this may favor stations with more samples.
    # - For valid evaluation, we compute two metrics:
    #   (1) RMSE over all samples (**same as valid loss**).
    #   (2) Averaged RMSE over stations (to treat each station equally). For SequenceWindowedDataset,
    #       this requires to aggregate predictions for each station first, e.g., by the 'longest_history' method,
    #       so that each day has only one prediction per station, the same as using SequenceFullDataset.
    #       Then compute the RMSE for each station, and finally average the RMSEs over all stations.
    #       **This is the same metric as in Ray tuner evaluation for test sets**.4950
    all_masks, all_preds, all_targets, all_preds_norm, all_targets_norm = [], [], [], [], []
    valid_ave_rmses = []

    for i_station, iter_idx in station_iterator:
        station_epoch_days = station_data_extractor(epoch_days, iter_idx)
        station_masks = station_data_extractor(masks, iter_idx)
        station_preds_norm = station_data_extractor(preds, iter_idx)
        station_targets_norm = station_data_extractor(targets, iter_idx)

        if check_is_aggregation_needed(dataloader_valid):  # windowed dataset:
            unique_epoch_days, aggregated_dict = aggregate_day_predictions(
                station_epoch_days,
                {
                    'masks': station_masks, 'preds_norm': station_preds_norm,
                    'targets_norm': station_targets_norm
                },
                method='longest_history'
            )
            station_masks = aggregated_dict['masks']
            station_preds_norm = aggregated_dict['preds_norm'][station_masks]  # masked tensor
            station_targets_norm = aggregated_dict['targets_norm'][station_masks]  # masked tensor
        else:  # Full sequence dataset:
            station_preds_norm = station_preds_norm[station_masks]
            station_targets_norm = station_targets_norm[station_masks]

        station_preds = normalizer_wt[i_station].inverse_transform(
            station_preds_norm.cpu().numpy().reshape(-1, 1)
        ).flatten()  # masked array
        station_targets = normalizer_wt[i_station].inverse_transform(
            station_targets_norm.cpu().numpy().reshape(-1, 1)
        ).flatten()  # masked array

        valid_ave_rmse = Error.rmse(station_preds, station_targets)
        valid_ave_rmses.append(valid_ave_rmse)

        all_preds_norm.append(station_preds_norm)
        all_targets_norm.append(station_targets_norm)
        all_masks.append(station_masks.cpu().numpy())  # array
        all_preds.append(station_preds)
        all_targets.append(station_targets)

    all_preds_norm = torch.concatenate(all_preds_norm, dim=0)
    all_targets_norm = torch.concatenate(all_targets_norm, dim=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    n_valid = np.sum(all_masks)
    if n_valid == 0:
        print(f'{ISSUE_TAG}Warning: No valid samples in validation set after masking NaNs!')
        raise StopIteration

    validation_mse = validation_criterion(all_preds_norm, all_targets_norm).cpu().numpy().item()
    validation_rmse = Error.rmse(all_preds, all_targets)
    validation_ave_rmse = np.mean(valid_ave_rmses)

    # # This is another way to compute the same metrics for SequenceFullDataset: (needs repair)
    # all_masks = torch.cat(masks, dim=0).flatten()  # `dim` considers first dim as batch dim
    # all_preds_norm = torch.cat(preds, dim=0).flatten()
    # all_targets_norm = torch.cat(targets, dim=0).flatten()
    # n_valid = all_masks.sum().item()
    # if n_valid == 0:
    #     print(f'{ISSUE_TAG}Warning: No valid samples in validation set after masking NaNs!')
    #     raise StopIteration
    # validation_mse = validation_criterion(all_preds_norm[all_masks], all_targets_norm[all_masks]).cpu().numpy()
    #
    # all_preds = normalizer_wt.inverse_transform(all_preds_norm[all_masks].cpu().numpy().reshape(-1, 1).flatten())
    # all_targets = normalizer_wt.inverse_transform(all_targets_norm[all_masks].cpu().numpy().reshape(-1, 1).flatten())
    # validation_rmse = Error.rmse(all_preds, all_targets)
    # valid_ave_rmses = []
    #
    # cumulative_sizes = [0] + dataloader_valid.dataset.cumulative_sizes  # cumulat # of sub-sequences per station
    # for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:]):
    #     station_epoch_days = torch.cat(epoch_days[start:end], dim=0).flatten()
    #     station_masks = torch.cat(masks[start:end], dim=0).flatten()
    #     station_preds = torch.cat(preds[start:end], dim=0).flatten()
    #     station_targets = torch.cat(targets[start:end], dim=0).flatten()
    #     valid_ave_rmse = Error.rmse(
    #         station_preds[station_masks].cpu().numpy(), station_targets[station_masks].cpu().numpy()
    #     )
    #     valid_ave_rmses.append(valid_ave_rmse)
    # validation_ave_rmse = np.mean(valid_ave_rmses)

    print(f'{INFO_TAG}len(validation_ave_rmses): {len(valid_ave_rmses)}')
    print(f'{INFO_TAG}n_valid: {n_valid}')

    return validation_mse, validation_ave_rmse, validation_rmse


def check_is_aggregation_needed(dataloader: torch.utils.data.DataLoader) -> bool:
    # todo: upgrade with aggregation method in config or full/windowed dataset
    if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset):
        # if settings.method in ['transformer_embedding']:
        # dataloader_valid.dataset (ConcatDataset) -> datasets (list) ->
        # datasets[0] (SequenceWindowedDataset, SequenceFullDataset, etc)
        return dataloader.dataset.datasets[0].window_len > 0
    else:
        return dataloader.dataset.window_len > 0
