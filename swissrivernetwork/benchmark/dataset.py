from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

CUR_ABS_DIR = Path(__file__).parent.resolve()
PROJ_DIR = (CUR_ABS_DIR / '../../').resolve()

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green


# Utility functions

def read_stations(graph_name, base_dir: str | Path = PROJ_DIR):
    x, _ = read_graph(graph_name, base_dir=base_dir)
    return [str(i) for i in x[:, 2].numpy()]


def read_graph(graph_name, base_dir: str | Path = PROJ_DIR):
    return torch.load(f'{str(base_dir)}/swissrivernetwork/benchmark/dump/graph_{graph_name}.pth')


def read_csv_train(graph_name, base_dir: str | Path = PROJ_DIR):
    return pd.read_csv(f'{base_dir}/swissrivernetwork/benchmark/dump/{graph_name}_train.csv')  # DUPLICATE?!


def read_csv_prediction_train(graph_name, station, base_dir: str | Path = PROJ_DIR):
    return pd.read_csv(f'{base_dir}/swissrivernetwork/benchmark/dump/prediction/{graph_name}_lstm_{station}_train.csv')


def select_isolated_station(df, station):
    return df[['epoch_day', f'{station}_wt', f'{station}_at']].rename(
        columns={f'{station}_wt': 'water_temperature', f'{station}_at': 'air_temperature'}
    )


def read_csv_test(graph_name, base_dir: str | Path = PROJ_DIR):
    return pd.read_csv(f'{base_dir}/swissrivernetwork/benchmark/dump/{graph_name}_test.csv')


def read_csv_prediction_test(
        graph_name, station, predict_dump_dir: str | Path | None = None, based_dir: str | Path = PROJ_DIR
):
    predict_dump_dir = f'{based_dir}/swissrivernetwork/benchmark/dump/prediction' if predict_dump_dir is None else predict_dump_dir
    return pd.read_csv(f'{predict_dump_dir}/{graph_name}_lstm_{station}_test.csv')


def read_station_data_from_df(df_all: pd.DataFrame, station: str):
    return df_all[['epoch_day', f'{station}_wt', f'{station}_at']].rename(
        columns={f'{station}_wt': f'{station}_wt_hat', f'{station}_at': f'{station}_at_hat'}
    )


def normalize_isolated_station(df):
    # Normalize air temperature, water temperature
    normalizer_at = MinMaxScaler()
    df['air_temperature'] = normalizer_at.fit_transform(df['air_temperature'].values.reshape(-1, 1))

    # Handle NaN in air temperature:
    if df['air_temperature'].isna().sum() > 0:
        print('[DATA PREPARATION] counted NaN values in input:', df['air_temperature'].isna().sum())
        assert False, 'We can not handle NaN in Input!'
        # df['air_temperature'] = df['air_temperature'].fillna(-1)

    # NaN in Output will be masked
    normalizer_wt = MinMaxScaler()
    df['water_temperature'] = normalizer_wt.fit_transform(df['water_temperature'].values.reshape(-1, 1))
    if df['water_temperature'].isna().sum() > 0:
        print('[DATA PREPARATION] counted NaN values in output:', df['water_temperature'].isna().sum())

    return df, normalizer_at, normalizer_wt


def normalize_columns(df):
    normalizer = MinMaxScaler()
    cols = df.columns.difference(['epoch_day', 'has_nan'])
    df_normalized = df.copy()
    df_normalized[cols] = pd.DataFrame(normalizer.fit_transform(df[cols]), columns=cols)
    return df_normalized, normalizer


def train_valid_split(config, df):
    # Split into Train and Validation:
    train_size = int(config['train_split'] * len(df))
    df_train = df.iloc[:train_size].reset_index(drop=True)
    df_valid = df.iloc[train_size:].reset_index(drop=True)
    return df_train, df_valid


# Dataset classes

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
            self, window_len, df, embedding_idx,
            short_subsequence_method: str = 'drop',  # 'pad' or 'drop'
            name: str = ''
    ):
        self.name = name
        self.window_len = window_len
        self.df = df
        self.embedding_idx = embedding_idx
        self.short_subsequence_method = short_subsequence_method

        self.stations = None
        self.sequences = []
        self.sequence_lengths = []
        self.extract_sequences()


    def extract_sequences(self):
        day_diff = self.df['epoch_day'].diff()
        breaks = day_diff != 1
        sequence_id = breaks.cumsum()
        sequences = self.df.index[breaks].values
        if self.window_len <= 0:  # Full sequences
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values
        else:
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values - self.window_len + 1

        df, sequences, sequence_lengths = self.process_short_subsequences(
            self.df, sequences, sequence_lengths, self.window_len, self.short_subsequence_method
        )
        self.df = df
        self.sequences = sequences
        self.sequence_lengths = sequence_lengths


    @staticmethod
    def process_short_subsequences(
            df, sequences: np.ndarray[int], sequence_lengths: np.ndarray[int], window_len: int,
            short_subsequence_method: str,
    ):
        """
        Process sequences that are shorter than the window length.

        Args:
            sequences (list[int]): List of subsequences, each represented by its start index at df.
        """
        if short_subsequence_method == 'drop':
            return SequenceDataset.process_short_subsequences_drop(df, sequences, sequence_lengths, window_len)
        elif short_subsequence_method == 'pad':
            return SequenceDataset.process_short_subsequences_pad(df, sequences, sequence_lengths, window_len)
        else:
            raise ValueError(f'Unknown short_subsequence_method: {short_subsequence_method}.')


    @staticmethod
    def process_short_subsequences_drop(
            df, sequences: np.ndarray[int], sequence_lengths: np.ndarray[int], window_len: int
    ):
        # remove short sequences
        drop_sequences = sequence_lengths < 0
        sequences = sequences[~drop_sequences]
        sequence_lengths = sequence_lengths[~drop_sequences]
        return df, sequences, sequence_lengths


    @staticmethod
    def process_short_subsequences_pad(
            df, sequences: np.ndarray[int], sequence_lengths: np.ndarray[int], window_len: int
    ):
        """
        Pad short sequences with nan values at the end to reach the desired window length. (Add and) revise the
        ``pad_mask`` column in the dataframe accordingly.

        Notice that ``pad_mask`` is separate from ``time_mask`` because some mask embeddings (e.g., for transformers)
        may be used for ``time_mask``.

        Args:
            df (pd.DataFrame): The original dataframe.
            sequences (list[int]): List of subsequences, each represented by its start index at df.
            sequence_lengths (list[int]): List of lengths of the subsequences. Notice if window_len > 0, these lengths
                are already adjusted to be `length - window_len + 1`.
        """
        drop_sequences = sequence_lengths < 0
        if not drop_sequences.any():  # No short sequences to pad
            return df, sequences, sequence_lengths

        if window_len > 0:
            sequence_lengths = sequence_lengths - 1 + window_len

        # Initialize padded df:
        dtype_dict = {col: df[col].dtype for col in df.columns}
        if 'pad_mask' in dtype_dict:
            columns = df.columns.tolist()
        else:
            columns = df.columns.tolist() + ['pad_mask']
            dtype_dict['pad_mask'] = bool
        df_padded = pd.DataFrame(columns=columns).astype(dtype_dict)

        # Pad with nan rows at the end of each short sequence:
        starts = sequences  # idx list
        ends = sequences + sequence_lengths  # idx list, exclusive
        padded_lens = []
        for i_seq, (start, end) in enumerate(zip(starts, ends)):
            if end - start < 5:
                print(f'{ISSUE_TAG}Removing sequence of length {ends - start} < 5 starting at index {start}!')
                # continue  todo: or raise error?
                raise ValueError('Sequence too short!')

            df_sequence = df.iloc[start:end].copy()
            if 'pad_mask' not in df_sequence.columns:
                df_sequence['pad_mask'] = False  # False means we have a value, not a mask
            seq_len = len(df_sequence)
            if seq_len < window_len:
                # pad with 0 rows at the end. This may be better than nan for MLP and attention steps:
                df_pad = pd.DataFrame(0, index=range(window_len - seq_len), columns=df.columns)
                df_pad['epoch_day'] = -1  # Use -1 instead of nan, because epoch_day is int type.
                # Don't mask padded rows in time_mask, because these masks may be used for mask embeddings:
                df_pad['time_mask'] = False
                df_pad['pad_mask'] = True  # True means we have a mask / missing value to be ignored
                df_sequence = pd.concat([df_sequence, df_pad], ignore_index=True)
                padded_lens.append(len(df_pad))
            else:
                padded_lens.append(0)  # Keep original length if the sequence is long enough
            df_padded = pd.concat([df_padded, df_sequence], ignore_index=True)

        df_padded = df_padded.reset_index(drop=True)
        sequences = sequences + np.concat(([0], np.cumsum(padded_lens)))[:-1]  # adjust start indices
        sequence_lengths = sequence_lengths + np.array(padded_lens)  # adjust lengths
        if window_len > 0:
            sequence_lengths = sequence_lengths - window_len + 1
        return df_padded, sequences, sequence_lengths


    def as_tensors(self, df):
        t = torch.IntTensor(df['epoch_day'].values).unsqueeze(-1)
        x = torch.FloatTensor(df['air_temperature'].values).unsqueeze(-1)
        y = torch.FloatTensor(df['water_temperature'].values).unsqueeze(-1)

        # check for predictions (wt_hat) (for graphlet models):
        matches = df.columns.str.contains(r'_wt_hat$')
        matching_columns = df.columns[matches]
        neighs = [torch.FloatTensor(df[col].values).unsqueeze(-1) for col in matching_columns]
        for neigh in neighs:
            x = torch.cat((x, neigh), dim=1)

        # check for embedding_idx:
        embs = torch.zeros((x.shape[0]), dtype=torch.long)
        if self.embedding_idx is not None:
            embs = torch.LongTensor([self.embedding_idx] * x.shape[0])

        if t is None or embs is None or x is None or y is None:
            print('haaaaaalt!')

        return (t, embs, x, y)


    def as_stgnn_tensors(self, df):
        ts = []
        xs = []
        ys = []
        for station in self.stations:
            ts.append(torch.IntTensor(df['epoch_day'].values).unsqueeze(-1))
            xs.append(torch.FloatTensor(df[f'{station}_at'].values).unsqueeze(-1))
            ys.append(torch.FloatTensor(df[f'{station}_wt'].values).unsqueeze(-1))

        t = torch.stack(ts, dim=0)
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)

        # add empty embeddings
        embs = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.long)

        if t is None or embs is None or x is None or y is None:
            print('haaaaaalt!')

        return (t, embs, x, y)


class SequenceFullDataset(SequenceDataset):
    '''
    Returns the full available sequence (no windowing)
    '''


    def __init__(self, df, embedding_idx=None, name: str = ''):
        super().__init__(0, df, embedding_idx, name=name)  # window_len=0 means full sequences


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        start = self.sequences[idx]
        length = self.sequence_lengths[idx]

        df = self.df.iloc[start:start + length]
        return self.as_tensors(df)


class SequenceWindowedDataset(SequenceDataset):

    def __init__(self, window_len, df, embedding_idx=None, name: str = '', dev_run: bool = False):
        super().__init__(window_len, df, embedding_idx, name=name)
        self.dev_run = dev_run


    def __len__(self):
        if self.dev_run:
            return min(10, np.sum(self.sequence_lengths))
        else:
            return np.sum(self.sequence_lengths)


    def __getitem__(self, idx):
        for i, length in enumerate(self.sequence_lengths):
            # This length already excludes windows that are too short. See `SequenceDataset.extract_sequences()`.
            if idx >= length:
                idx -= length
                continue

            # idx is now in sequence:
            start = self.sequences[i] + idx

            df = self.df.iloc[start:start + self.window_len]  # Windowed DF
            return self.as_tensors(df)


# %% Masked Datasets without station connections


class SequenceMaskedDataset(SequenceDataset):

    def __init__(
            self, window_len, df, embedding_idx, max_mask_ratio: float = 0.25, max_mask_consecutive: int = 10,
            short_subsequence_method: str = 'pad',  # 'pad' or 'drop'
            name: str = ''
    ):
        self.max_mask_ratio = max_mask_ratio
        self.max_mask_consecutive = max_mask_consecutive
        # self.time_masks = None  # will be set in extract_sequences
        super().__init__(window_len, df, embedding_idx, short_subsequence_method=short_subsequence_method, name=name)


    def extract_sequences(self):
        day_diff = self.df['epoch_day'].diff()
        if (day_diff < 0).any():
            raise ValueError('DataFrame must be sorted by epoch_day in ascending order!')
        if (day_diff[1:].isna()).any():
            raise ValueError('DataFrame must not contain NaN in ``epoch_day``!')

        large_breaks = (day_diff > self.max_mask_consecutive + 1)

        # Pad with 0 at row with small day gaps and add time_mask column:
        dtype_dict = {col: self.df[col].dtype for col in self.df.columns}
        dtype_dict['time_mask'] = bool
        df_padded = pd.DataFrame(columns=self.df.columns.tolist() + ['time_mask']).astype(dtype_dict)
        breaks = day_diff != 1  # bool
        sequence_id = breaks.cumsum()  # int, starting from 1
        sequences = self.df.index[breaks].values  # idx list
        sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values  # int list
        starts = sequences  # idx list
        ends = sequences + sequence_lengths  # idx list, exclusive
        for i_seq, (start, end) in enumerate(zip(starts, ends)):
            df_sequence = self.df.iloc[start:end].copy()
            df_sequence['time_mask'] = False  # False means we have a value, not a mask
            df_padded = pd.concat([df_padded, df_sequence], ignore_index=True)
            if end >= len(self.df):
                break
            cur_day_diff = int(self.df.iloc[end]['epoch_day'] - self.df.iloc[end - 1]['epoch_day'])
            if cur_day_diff <= self.max_mask_consecutive + 1:
                # missing days to pad (if padding with nan, transformer QK matrix will corrupt even with mask):
                # todo: check how it affects lstms
                df_gaps = pd.DataFrame(0, index=range(cur_day_diff - 1), columns=self.df.columns)
                df_gaps['epoch_day'] = range(
                    int(self.df.iloc[end - 1]['epoch_day'] + 1), int(self.df.iloc[end]['epoch_day'])
                )
                df_gaps['time_mask'] = True  # True means we have a mask / missing value to be ignored
                df_padded = pd.concat([df_padded, df_gaps], ignore_index=True)

        self.df = df_padded.reset_index(drop=True)

        # Then, rerun the sequence extraction on the padded df:
        day_diff = self.df['epoch_day'].diff()
        breaks = day_diff != 1  # bool
        assert (breaks.sum() == large_breaks.sum() + 1), 'Large breaks mismatch!'

        sequence_id = breaks.cumsum()  # int, starting from 1
        sequences = self.df.index[breaks].values  # idx list
        if self.window_len <= 0:  # full sequences
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values
        else:
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values - self.window_len + 1

        df, sequences, sequence_lengths = self.process_short_subsequences(
            self.df, sequences, sequence_lengths, self.window_len, self.short_subsequence_method
        )
        self.df = df
        self.sequences = sequences
        self.sequence_lengths = sequence_lengths
        # todo: Consider max_mask_ratio here
        # drop_sequences = drop_sequences | (self.masks.groupby(sequence_id).sum().values > self.max_mask_ratio * sequence_lengths)


    def as_tensors(self, df: pd.DataFrame):
        t = torch.IntTensor(df['epoch_day'].values).unsqueeze(-1)
        x = torch.FloatTensor(df['air_temperature'].values).unsqueeze(-1)
        y = torch.FloatTensor(df['water_temperature'].values).unsqueeze(-1)
        time_masks = torch.BoolTensor(df['time_mask'].values)  # [seq_len], True means to ignore / to mask

        # check for predictions (wt_hat):  fixme validate this part:
        matches = df.columns.str.contains(r'_wt_hat$')
        matching_columns = df.columns[matches]
        neighs = [torch.FloatTensor(df[col].values).unsqueeze(-1) for col in matching_columns]
        for neigh in neighs:
            x = torch.cat((x, neigh), dim=1)

        # check for embedding_idx:
        embs = torch.zeros((x.shape[0]), dtype=torch.long)
        if self.embedding_idx is not None:
            embs = torch.LongTensor([self.embedding_idx] * x.shape[0])

        if t is None or embs is None or x is None or y is None:
            print('haaaaaalt!')

        if 'pad_mask' in df.columns:
            pad_masks = torch.BoolTensor(df['pad_mask'].values)  # [seq_len], True means to ignore / to mask
            return t, embs, x, y, time_masks, pad_masks
        else:
            return t, embs, x, y, time_masks


    def as_stgnn_tensors(self, df):
        ts = []
        xs = []
        ys = []
        for station in self.stations:
            ts.append(torch.IntTensor(df['epoch_day'].values).unsqueeze(-1))
            xs.append(torch.FloatTensor(df[f'{station}_at'].values).unsqueeze(-1))
            ys.append(torch.FloatTensor(df[f'{station}_wt'].values).unsqueeze(-1))

        t = torch.stack(ts, dim=0)
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)

        # add empty embeddings
        embs = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.long)

        if t is None or embs is None or x is None or y is None:
            print('haaaaaalt!')

        return (t, embs, x, y)


class SequenceMaskedWindowedDataset(SequenceMaskedDataset):

    def __init__(
            self, window_len, df, embedding_idx=None,
            max_mask_ratio: float = 0.25, max_mask_consecutive: int = 10,
            short_subsequence_method: str = 'pad',  # 'pad' or 'drop'
            name: str = '',
            dev_run: bool = False
    ):
        super().__init__(
            window_len, df, embedding_idx, max_mask_ratio=max_mask_ratio, max_mask_consecutive=max_mask_consecutive,
            short_subsequence_method=short_subsequence_method,
            name=name
        )
        self.dev_run = dev_run


    def __len__(self):
        if self.dev_run:
            return min(10, np.sum(self.sequence_lengths))
        else:
            return np.sum(self.sequence_lengths)


    def __getitem__(self, idx: int):
        for i, length in enumerate(self.sequence_lengths):
            # This length already excludes windows that are too short. See `SequenceMaskedDataset.extract_sequences()`.
            if idx >= length:
                idx -= length
                continue

            # idx is now in sequence:
            start = self.sequences[i] + idx

            df = self.df.iloc[start:start + self.window_len]  # Windowed DF
            return self.as_tensors(df)


# %% STGNN Datasets


class STGNNSequenceFullDataset(SequenceFullDataset):

    def __init__(self, df, stations):
        super().__init__(df)
        self.stations = stations


    def as_tensors(self, df):
        return super().as_stgnn_tensors(df)


class STGNNSequenceWindowedDataset(SequenceWindowedDataset):

    def __init__(self, window_len, df, stations, dev_run: bool = False):
        super().__init__(window_len, df, dev_run=dev_run)
        self.stations = stations


    def as_tensors(self, df):
        return super().as_stgnn_tensors(df)
