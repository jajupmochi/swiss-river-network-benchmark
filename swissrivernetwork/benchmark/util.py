import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from benedict import benedict
from torch_geometric.utils import k_hop_subgraph, to_undirected

from swissrivernetwork.benchmark.dataset import read_graph


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


# %% Ray Tune related utils:


def safe_get_ray_trial_id():
    from ray.air import session
    sess = session.get_session()
    if sess is not None and session.get_trial_id():
        return session.get_trial_id()
    return None


def get_run_name(method: str, graph_name: str, now: str, config: benedict | dict) -> str:
    run_name = f'{method}-{graph_name}'
    run_name += get_run_name(config)
    run_name += f'-{now}'

    return run_name


def get_run_extra_key(config: benedict | dict):
    if isinstance(config, dict):
        config = benedict(config)
    extra_key = ''
    if 'use_current_x' in config and config.use_current_x is not None and not config.use_current_x:
        extra_key += f'-next_tokens'
    if 'window_len' in config and config.window_len is not None:
        extra_key += f'-wl{config.window_len}'
    if 'missing_value_method' in config and config.missing_value_method is not None:
        extra_key += f'-{config.missing_value_method}'
    if 'positional_encoding' in config:
        extra_key += f'-{config.positional_encoding}'.lower()  # None -> none
    return extra_key


# %% Date time related utils:


def is_valid_datetime(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
        return True
    except ValueError:
        return False
