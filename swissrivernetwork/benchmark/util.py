import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from benedict import benedict
from torch_geometric.utils import k_hop_subgraph, to_undirected

from swissrivernetwork.benchmark.dataset import read_graph

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green


def save(object, checkpoint_dir, name):
    path = os.path.join(checkpoint_dir, name)
    torch.save(object, path)


def extract_neighbors(graph_name, station, num_hops):
    # Use undirected edges:
    x, e = read_graph(graph_name)
    e = to_undirected(e)

    # Extract k-hop neighborhood
    idx = (x[:, 2] == int(station)).nonzero().item()
    x_sub_idx, e, _, _ = k_hop_subgraph(idx, num_hops, e, relabel_nodes=True)
    x = x[x_sub_idx]
    neighs = [str(s.item()) for s in x[:, 2]]
    neighs.remove(station)  # remove target station
    return neighs


def merge_graphlet_dfs(df, df_neighs):
    for df_neigh in df_neighs:
        df = pd.merge(df, df_neigh, on='epoch_day', how='outer')
        # fill NaN:
        col = df_neigh.columns[1]
        # df[col] = df[col].fillna(-1)
        assert df[col].isna().sum() == 0, f'NaN in neigh {col} detected!'
    has_any_nan = df.drop(columns=['air_temperature', 'water_temperature']).isna().any().any()
    assert not has_any_nan, 'There is a NaN in your data!'
    return df


# %% Prediction aggregation utils:


def aggregate_day_predictions(
        epoch_days: np.ndarray | torch.Tensor,
        dict_to_aggregate: dict[str, np.ndarray | torch.Tensor], method: str | None = 'longest_history'
) -> (np.ndarray | torch.Tensor, dict[str, np.ndarray | torch.Tensor]):
    """
    Aggregate day predictions from multiple windows.
    Input:
        epoch_days: np.ndarray or torch.Tensor of shape (total_length,). Make sure that the days are in time order if using
            'longest_history' or 'last' methods.
        dict_to_aggregate: dict of np.ndarray or torch.Tensor of shape (total_length,). Each array must align with `epoch_days`.
        method: aggregation method, one of [None, 'last', 'mean', 'median', 'longest_history']. Default: 'longest_history'.
    Output:
        unique_epoch_days: np.ndarray or torch.Tensor of shape (num_unique_days,)
        aggregated_dict: dict of np.ndarray or torch.Tensor of shape (num_unique_days, ...)
    """
    if method is None:
        return epoch_days, dict_to_aggregate

    if method == 'longest_history':
        # For each given day, this method chooses the predictions for that day who has the longest history.
        # Here we assume that the input `epoch_days` is organized as the concatenation of multiple sliding windows,
        # (e.g., the dataloader with batch_size=1 and shuffle=False). Thus, for each day, the first occurrence of that
        # day in `epoch_days` corresponds to the window with the longest history.
        if isinstance(epoch_days, torch.Tensor):
            unique_epoch_days, day_indices = torch_unique_as_numpy(epoch_days, return_index=True)
            aggregated_dict = {}
            for key, values in dict_to_aggregate.items():
                aggregated_dict[key] = values[day_indices]
            return unique_epoch_days, aggregated_dict
        else:
            unique_epoch_days, day_indices = np.unique(epoch_days, return_index=True)
            aggregated_dict = {}
            for key, values in dict_to_aggregate.items():
                aggregated_dict[key] = values[day_indices]
            return unique_epoch_days, aggregated_dict

    elif method == 'last':
        # Here "last" means that for the prediction of a given day, we take the prediction from the last window:
        if isinstance(epoch_days, torch.Tensor):
            raise NotImplementedError('last method not implemented for torch.Tensor yet.')
        else:
            unique_epoch_days, idx = np.unique(epoch_days[::-1], return_index=True)
            unique_epoch_days = unique_epoch_days[::-1]
            last_indices = len(epoch_days) - 1 - idx[::-1]
            aggregated_dict = {}
            for key, values in dict_to_aggregate.items():
                aggregated_dict[key] = values[last_indices]
            return unique_epoch_days, aggregated_dict

    elif method == 'mean':
        if isinstance(epoch_days, torch.Tensor):
            raise NotImplementedError('mean method not implemented for torch.Tensor yet.')
        else:
            unique_epoch_days = np.unique(epoch_days)
            aggregated_dict = {}
            for key, values in dict_to_aggregate.items():  # todo: use np.bincount for speed
                agg_values = []
                for day in unique_epoch_days:
                    day_values = values[epoch_days == day]
                    agg_values.append(np.mean(day_values, axis=0))
                aggregated_dict[key] = np.array(agg_values)
            return unique_epoch_days, aggregated_dict

    elif method == 'median':
        if isinstance(epoch_days, torch.Tensor):
            raise NotImplementedError('median method not implemented for torch.Tensor yet.')
        else:
            unique_epoch_days = np.unique(epoch_days)
            aggregated_dict = {}
            for key, values in dict_to_aggregate.items():
                agg_values = []
                for day in unique_epoch_days:
                    day_values = values[epoch_days == day]
                    agg_values.append(np.median(day_values, axis=0))
                aggregated_dict[key] = np.array(agg_values)
            return unique_epoch_days, aggregated_dict

    else:
        raise ValueError(f'Unknown aggregation method: {method}.')


def torch_unique_as_numpy(t: torch.Tensor, return_index: bool = False) -> torch.Tensor:
    """
    Mimic the behavior of `np.unique` for torch.Tensor.
    This function returns the unique values in the order of their first occurrence in the input tensor. The unique values
    are sorted in ascending order by default.

    Notice that `torch.unique` always sorts the unique values, while `np.unique` does not sort by default.

    Input:
        t: torch.Tensor of any shape.
        return_index: whether to return the indices of the first occurrences of the unique values.

    Output:
        unique_vals: torch.Tensor of shape (num_unique_vals,)
        first_indices (optional): torch.Tensor of shape (num_unique_vals,)
    """
    # Notice `torch.unique` always sorts no matter to what `sorted` is set.
    # `reverse_indices` gives the indices to reconstruct the original tensor from the unique tensor:
    unique_vals, inverse_indices, counts = torch.unique(t, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(inverse_indices, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    if return_index:
        return unique_vals, first_indices
    else:
        return unique_vals


def check_is_aggregation_needed(dataloader: torch.utils.data.DataLoader, is_extrapolation: bool = True) -> bool:
    # todo: upgrade with aggregation method in config or full/windowed dataset
    if is_extrapolation:  # Return all values (corresponding to future prediction) without aggregation:
        return False
    if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset):
        # if settings.method in ['transformer_embedding']:
        # dataloader_valid.dataset (ConcatDataset) -> datasets (list) ->
        # datasets[0] (SequenceWindowedDataset, SequenceFullDataset, etc)
        return dataloader.dataset.datasets[0].window_len > 0
    else:
        return dataloader.dataset.window_len > 0


# %% Ray Tune related utils:


def safe_get_ray_trial_id():
    from ray.air import session
    sess = session.get_session()
    if sess is not None and session.get_trial_id():
        return session.get_trial_id()
    return None


def get_run_name(method: str, graph_name: str, now: str, config: benedict | dict, directory: Path | None = None) -> str:
    run_name = f'{method}-{graph_name}'
    extra_keys = get_run_extra_key(config)
    run_name += extra_keys
    resume = config.get('resume', False)
    if resume:
        if config.get('resume_timestamp', None) is not None:
            run_name += f'-{config.resume_time_stamp}'
        else:
            # If time stamp is None, use the latest run:
            directory = Path(directory) if directory is not None else Path.cwd()
            path_prefix = Path(run_name).name + ('-' if extra_keys else '')
            run_name = get_latest_run_path(directory, path_prefix=path_prefix, verbose=False)
            run_name = run_name.relative_to(directory)
        print(f'{INFO_TAG}Resuming from previous run: {run_name}.')
    else:
        run_name += f'-{now}'

    return run_name


def get_run_extra_key(config: benedict | dict) -> str:
    if isinstance(config, dict):
        config = benedict(config)
    extra_key = ''
    if 'use_current_x' in config and config.use_current_x is not None and not config.use_current_x:
        extra_key += f'-fs{config.future_steps}'
    if 'window_len' in config and config.window_len is not None:
        extra_key += f'-wl{config.window_len}'
    if 'missing_value_method' in config and config.missing_value_method is not None:
        extra_key += f'-{config.missing_value_method}'
    if 'positional_encoding' in config:
        extra_key += f'-{config.positional_encoding}'.lower()  # None -> none
    return extra_key


def get_latest_run_path(directory: Path, path_prefix: str = '', verbose: bool = False) -> Path:
    all_paths = sorted(
        [path for path in directory.iterdir() if
         path.is_dir() and path.name.startswith(path_prefix) and is_valid_datetime(path.name[len(path_prefix):])]
    )
    assert len(all_paths) > 0, f'No previous results found. Path prefix {path_prefix}.'
    latest_path = all_paths[-1]
    verbose and print(f'{INFO_TAG}Loading latest results from {latest_path}.')
    return latest_path


def trim_checkpoints(
        root_path: Path, keep_best_n: int = 10, anchor_metric: str = 'validation_mse', mode: str = 'min',
        if_trim_best_n: bool = True,
        keep_best_for_trimmed_trials: bool = True, keep_last_for_trimmed_trials: bool = False,
        remove_seperated_marker_files: bool = False,
        cur_depth: int = 0,
        verbose: bool = True
):
    """Keep only the best n checkpoints based on the given metric.

    Consider two cases: the first is that root_path contains multiple trials, and each trial contains multiple
    checkpoints; the second is that root_path contains multiple experiments, each experiment contains multiple trials.

    This can save around 80% disk space in some of our experiments.

    Input:
        root_path: Path to the root directory of the experiment(s).
        keep_best_n: Number of best trials to keep.
        anchor_metric: Metric used to determine the best trials.
        mode: 'min' or 'max', whether to minimize or maximize the anchor_metric.
        if_trim_best_n: Whether to trim the best n trials as well. Always keep the best and last checkpoints for the
            best n trials.
        keep_best_for_trimmed_trials: Whether to keep the best checkpoint for the trimmed trials. This does not apply
            to the best n trials if `if_trim_best_n` is True.
        keep_last_for_trimmed_trials: Whether to keep the last checkpoint for the trimmed trials. This does not apply
            to the best n trials if `if_trim_best_n` is True.
        remove_seperated_marker_files: Whether to remove the seperated marker files (e.g., checkpoint_000000-REMOVED).
    """


    def trim_one_trial(trial, analysis, keep_best_checkpoint, keep_last_checkpoint):
        trial_id = trial.trial_id  # e.g., trial_id: 8d0b9_00000
        # e.g., trial_path: path-to-root/train_transformer_8d0b9_00000_0_batch_size=189,..._2025-10-01_16-04-31:
        trial_path = Path(trial.path)

        trial_dataframe = analysis.trial_dataframes[trial_id]
        checkpoint_dir_names = trial_dataframe['checkpoint_dir_name'].tolist()

        n_files_removed = 0
        disk_space_freed = 0
        n_marker_files_removed = 0
        files_trimmed = []

        dir_names_ignored = set()
        if keep_last_checkpoint:
            # Keep the last best checkpoint for this trial:
            last_ck_dir_name = Path(analysis.get_last_checkpoint(trial).path).name
            dir_names_ignored.add(last_ck_dir_name)
        if keep_best_checkpoint:
            # Keep the best checkpoint for this trial:
            best_ck_dir_name = Path(analysis.get_best_checkpoint(trial, metric=anchor_metric, mode=mode).path).name
            dir_names_ignored.add(best_ck_dir_name)
        if dir_names_ignored:
            ck_dir_names_to_remove = sorted(list(set(checkpoint_dir_names) - dir_names_ignored))
        else:
            ck_dir_names_to_remove = checkpoint_dir_names

        for ck_dir_name in ck_dir_names_to_remove:  # e.g., ck_dir_name: checkpoint_000000
            ck_path = trial_path / ck_dir_name
            for child in ck_path.iterdir():
                if child.is_file() and child.name.endswith('.pth'):
                    # Remove model file:
                    file_size = child.stat().st_size
                    disk_space_freed += file_size
                    n_files_removed += 1

                    child.unlink()
                    # # Create a dummy file to indicate that this checkpoint has been removed:
                    # This will generate too many files, which may use up quota on cluster file systems.
                    # (ck_path / f'{child.name}-REMOVED').touch()
                    files_trimmed.append(str(ck_path.relative_to(root_path) / f'{child.name}'))

                # Remove marker files. This is to curate the previous behavior where we create seperated marker files
                # to indicate removed checkpoints:
                if remove_seperated_marker_files:
                    if child.is_file() and child.name.endswith('.pth-REMOVED'):
                        child.unlink()
                        # remove '-REMOVED' suffix:
                        file_trimmed = str(ck_path.relative_to(root_path) / f'{child.name[:-8]}')
                        files_trimmed.append(file_trimmed)
                        n_marker_files_removed += 1
            # checkpoint_path.rmdir()

        if verbose:
            if n_files_removed > 0:
                print(
                    f'  {INFO_TAG}Removed {n_files_removed} model files from trial {trial_id}. '
                    f'Freed {disk_space_freed / (1024 ** 2):.2f} MB disk space.'
                )
            # else:
            #     print(f'  {INFO_TAG}No model files removed from trial {trial_id}.')

            if n_marker_files_removed > 0:
                print(
                    f'  {INFO_TAG}Removed {n_marker_files_removed} seperated marker files from trial {trial_id}.'
                )

        return files_trimmed, disk_space_freed


    children = list(root_path.iterdir())
    if any(
            child.is_file() and child.name.startswith('experiment_state-') and child.name.endswith('.json') for child in
            children
    ):
        # case 1 (single experiment):
        from ray.tune import ExperimentAnalysis
        analysis = ExperimentAnalysis(root_path)
        trial_ids = [t.trial_id for t in analysis.trials]
    elif cur_depth == 0:
        # case 2 (multiple experiments):
        for child in children:
            if child.is_dir():
                trim_checkpoints(
                    child, keep_best_n=keep_best_n, anchor_metric=anchor_metric, mode=mode,
                    if_trim_best_n=if_trim_best_n,
                    keep_best_for_trimmed_trials=keep_best_for_trimmed_trials,
                    keep_last_for_trimmed_trials=keep_last_for_trimmed_trials,
                    cur_depth=cur_depth + 1,
                    remove_seperated_marker_files=remove_seperated_marker_files,
                    verbose=verbose
                )
        return
    else:
        raise ValueError(f'Invalid root_path: {root_path}.')

    if len(trial_ids) <= keep_best_n:
        print(
            f'{INFO_TAG}Number of trials ({len(trial_ids)}) <= keep_best_n ({keep_best_n}). '
            f'No trimming needed for this experiment.'
        )
        return

    # Get best n trials:
    best_anchor_metrics = [t.metric_analysis[anchor_metric][mode] for t in analysis.trials]
    idx_sorted = np.argsort(best_anchor_metrics)
    if mode == 'max':
        idx_sorted = idx_sorted[::-1]
    elif mode != 'min':
        raise ValueError(f'Unknown mode: {mode}.')
    idx_best_n = idx_sorted[:keep_best_n]
    idx_to_trim = idx_sorted[keep_best_n:]
    best_n_trials = [analysis.trials[i] for i in idx_best_n]
    trials_to_trim = [analysis.trials[i] for i in sorted(idx_to_trim)]
    trial_dataframes = analysis.trial_dataframes

    total_n_files_removed = 0
    total_disk_space_freed = 0
    files_trimmed = []

    for trial in trials_to_trim:
        files_removed, disk_space_freed = trim_one_trial(
            trial, analysis,
            keep_best_checkpoint=keep_best_for_trimmed_trials, keep_last_checkpoint=keep_last_for_trimmed_trials
        )
        total_n_files_removed += len(files_removed)
        total_disk_space_freed += disk_space_freed
        files_trimmed.extend(files_removed)

    if if_trim_best_n:
        for trial in best_n_trials:
            files_removed, disk_space_freed = trim_one_trial(
                trial, analysis, keep_best_checkpoint=True, keep_last_checkpoint=True
            )
            total_n_files_removed += len(files_removed)
            total_disk_space_freed += disk_space_freed
            files_trimmed.extend(files_removed)

    if len(files_trimmed) > 0:
        # Save the list of trimmed files. Append if the file already exists.
        trimmed_files_path = root_path / f'trimmed_files.txt'
        with open(trimmed_files_path, 'a') as f:
            for file_path in files_trimmed:
                f.write(f'{file_path}\n')

    if verbose:
        print(
            f'{SUCCESS_TAG}Total removed {total_n_files_removed} files for this experiment. '
            f'Total freed {total_disk_space_freed / (1024 ** 2):.2f} MB disk space.'
        )


# %% Date time related utils:


def is_valid_datetime(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
        return True
    except ValueError:
        return False


# %% Misc utils:

def is_transformer_model(method: str) -> bool:
    return 'transformer' in method.lower()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true',):
        return True
    elif v.lower() in ('false',):
        return False
    else:
        from argparse import ArgumentTypeError
        raise ArgumentTypeError('Boolean value expected.')


def get_proper_infer_batchsize(method: str, graph_name: str) -> int:
    """
    The inferences for the LSTMs v.s. Transformers experiments were carried out on a PC with a NVIDIA GeForce RTX 3070
    GPU with 8 GB memory. The batch sizes here are chosen to fully utilize the GPU memory without OOM.
    """
    print(method, graph_name)
    if method == 'stgnn':
        if graph_name in ['swiss-1990']:
            return 128
    return 256
