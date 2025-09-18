from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

CUR_ABS_DIR = Path(__file__).parent.resolve()
PROJ_DIR = (CUR_ABS_DIR / '../../').resolve()


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


def read_csv_prediction_test(graph_name, station, base_dir: str | Path = PROJ_DIR):
    return pd.read_csv(f'{base_dir}/swissrivernetwork/benchmark/dump/prediction/{graph_name}_lstm_{station}_test.csv')


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
    return df_normalized


def train_valid_split(config, df):
    # Split into Train and Validation:
    train_size = int(config['train_split'] * len(df))
    df_train = df.iloc[:train_size].reset_index(drop=True)
    df_valid = df.iloc[train_size:].reset_index(drop=True)
    return df_train, df_valid


# Dataset classes

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, window_len, df, embedding_idx):
        self.window_len = window_len
        self.df = df
        self.embedding_idx = embedding_idx

        self.stations = None
        self.sequences = []
        self.sequence_lengths = []
        self.extract_sequences()


    def extract_sequences(self):
        day_diff = self.df['epoch_day'].diff()
        breaks = day_diff != 1
        sequence_id = breaks.cumsum()
        sequences = self.df.index[breaks].values
        if self.window_len <= 0:
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values
        else:
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values - self.window_len + 1

        # remove short sequences
        drop_sequences = sequence_lengths < 0
        self.sequences = sequences[~drop_sequences]
        self.sequence_lengths = sequence_lengths[~drop_sequences]


    def as_tensors(self, df):
        t = torch.IntTensor(df['epoch_day'].values).unsqueeze(-1)
        x = torch.FloatTensor(df['air_temperature'].values).unsqueeze(-1)
        y = torch.FloatTensor(df['water_temperature'].values).unsqueeze(-1)

        # check for predictions (wt_hat):
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


    def __init__(self, df, embedding_idx=None):
        super().__init__(0, df, embedding_idx)


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        start = self.sequences[idx]
        length = self.sequence_lengths[idx]

        df = self.df.iloc[start:start + length]
        return self.as_tensors(df)


class SequenceWindowedDataset(SequenceDataset):

    def __init__(self, window_len, df, embedding_idx=None, dev_run: bool = False):
        super().__init__(window_len, df, embedding_idx)
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


# %%


class SequenceMaskedDataset(SequenceDataset):

    def __init__(self, window_len, df, embedding_idx, max_mask_ratio: float = 0.25, max_mask_consecutive: int = 10):
        self.max_mask_ratio = max_mask_ratio
        self.max_mask_consecutive = max_mask_consecutive
        # self.time_masks = None  # will be set in extract_sequences
        super().__init__(window_len, df, embedding_idx)


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
        if self.window_len <= 0:
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values
        else:
            sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values - self.window_len + 1

        # remove short sequences:
        drop_sequences = sequence_lengths < 0
        # todo: Consider max_mask_ratio here
        # drop_sequences = drop_sequences | (self.masks.groupby(sequence_id).sum().values > self.max_mask_ratio * sequence_lengths)
        self.sequences = sequences[~drop_sequences]
        self.sequence_lengths = sequence_lengths[~drop_sequences]


    def as_tensors(self, df: pd.DataFrame):
        t = torch.IntTensor(df['epoch_day'].values).unsqueeze(-1)
        x = torch.FloatTensor(df['air_temperature'].values).unsqueeze(-1)
        y = torch.FloatTensor(df['water_temperature'].values).unsqueeze(-1)
        time_masks = torch.BoolTensor(df['time_mask'].values)  # [seq_len], True means to ignore / to mask

        # check for predictions (wt_hat):
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

        return (t, embs, x, y, time_masks)


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
            dev_run: bool = False
    ):
        super().__init__(
            window_len, df, embedding_idx, max_mask_ratio=max_mask_ratio, max_mask_consecutive=max_mask_consecutive
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


class STGNNSequenceFullDataset(SequenceFullDataset):

    def __init__(self, df, stations):
        super().__init__(df)
        self.stations = stations


    def as_tensors(self, df):
        return super().as_stgnn_tensors(df)


class STGNNSequenceWindowedDataset(SequenceWindowedDataset):

    def __init__(self, window_len, df, stations):
        super().__init__(window_len, df)
        self.stations = stations


    def as_tensors(self, df):
        return super().as_stgnn_tensors(df)
