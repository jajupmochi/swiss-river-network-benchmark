"""
scaler



@Author: linlin
@Date: Oct 09 2025
"""
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


class DimSplitScaler(BaseEstimator, TransformerMixin):
    """
    Single feature scaler
    """


    def __init__(self, global_scaler: Union[MinMaxScaler, StandardScaler], dims_kept: np.ndarray[int] | None = None):
        try:
            check_is_fitted(global_scaler)
        except ValueError as e:
            raise ValueError(
                f'``global_scaler`` must be fitted before initializing DimSplitScaler.\nOriginal error: {e}'
            )

        if dims_kept is None:
            dims_kept = np.arange(global_scaler.n_features_in_)
        if np.any(dims_kept < 0) or np.any(dims_kept >= global_scaler.n_features_in_):
            raise ValueError(f'``dims_kept`` must be between 0 and {global_scaler.n_features_in_ - 1}.')
        self.dims_kept = dims_kept
        self.n_dim = len(dims_kept)

        self.scalers = [self.build_single_dim_scaler(global_scaler, dim) for dim in dims_kept]

        self.cur_dim = None


    @staticmethod
    def build_single_dim_scaler(global_scaler, dim):
        """
        Build a scaler for a single dimension using the parameters from the global scaler.

        Args:
            global_scaler (Union[MinMaxScaler, StandardScaler]): The fitted global scaler.
            dim (int): The dimension index to build the single dimension scaler for.
        """
        if isinstance(global_scaler, MinMaxScaler):
            single_dim_scaler = MinMaxScaler()
        elif isinstance(global_scaler, StandardScaler):
            single_dim_scaler = StandardScaler()
        else:
            raise ValueError('Unsupported scaler type. Only MinMaxScaler and StandardScaler are supported.')

        # Manually set the parameters for the single dimension scaler:
        single_dim_scaler.min_ = np.array([global_scaler.min_[dim]]) if hasattr(global_scaler, 'min_') else None
        single_dim_scaler.scale_ = np.array([global_scaler.scale_[dim]]) if hasattr(global_scaler, 'scale_') else None
        single_dim_scaler.data_min_ = np.array([global_scaler.data_min_[dim]]) if hasattr(
            global_scaler, 'data_min_'
        ) else None
        single_dim_scaler.data_max_ = np.array([global_scaler.data_max_[dim]]) if hasattr(
            global_scaler, 'data_max_'
        ) else None
        single_dim_scaler.data_range_ = np.array([global_scaler.data_range_[dim]]) if hasattr(
            global_scaler, 'data_range_'
        ) else None
        single_dim_scaler.feature_names_in_ = np.array([global_scaler.feature_names_in_[dim]]) if hasattr(
            global_scaler, 'feature_names_in_'
        ) else None
        single_dim_scaler.n_features_in_ = 1
        single_dim_scaler.n_samples_seen_ = global_scaler.n_samples_seen_ if hasattr(
            global_scaler, 'n_samples_seen_'
        ) else None
        single_dim_scaler.feature_range_ = global_scaler.feature_range if hasattr(
            global_scaler, 'feature_range'
        ) else None
        return single_dim_scaler


    def __getitem__(self, item: int):
        """
        Set the current dimension to be transformed.

        Args:
            item (int): The dimension index to set as current. Notice this index corresponds to ``self.scalers``,
                        not the original data dimension in ``global_scaler``.
        """
        if not isinstance(item, int):
            raise TypeError('Index must be an integer.')
        if item < 0 or item >= self.n_dim:
            raise IndexError(f'Index out of range. Must be between 0 and {self.n_dim - 1}.')
        self.cur_dim = item
        return self


    def transform(self, x: np.ndarray):
        """
        Transform a single dimensional input array using the scaler fitted on the full data.

        Args:
            x (np.ndarray): 1D array of shape (n_samples, 1)
        """
        # Validate input (must be single feature):
        if not x.ndim == 2 or not x.shape[1] == 1:
            raise ValueError('Input data must be a 2D array with a single feature (shape: (n_samples, 1)).')

        # check_is_fitted(self.global_scaler)
        if self.cur_dim is None:
            raise ValueError('cur_dim is not set. Please use indexing to set the dimension first.')

        # Transformer the input at the dimension cur_dim:
        return self.scalers[self.cur_dim].transform(x)


    def inverse_transform(self, x_scaled):
        """
        Inverse transform a single dimensional input array using the scaler fitted on the full data.

        Args:
            x_scaled (np.ndarray): 1D array of shape (n_samples, 1)
        """
        # Validate input (must be single feature):
        if not x_scaled.ndim == 2 or not x_scaled.shape[1] == 1:
            raise ValueError('Input data must be a 2D array with a single feature (shape: (n_samples, 1)).')

        # check_is_fitted(self.global_scaler)
        if self.cur_dim is None:
            raise ValueError('cur_dim is not set. Please use indexing to set the dimension first.')

        # Inverse transformer the input at the dimension cur_dim:
        return self.scalers[self.cur_dim].inverse_transform(x_scaled)
