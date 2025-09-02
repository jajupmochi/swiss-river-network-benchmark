
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

CUR_ABS_DIR = Path(__file__).parent.resolve()
PROJ_DIR = (CUR_ABS_DIR / '../../').resolve()

# Utility functions

def read_stations(graph_name):
    x,_ = read_graph(graph_name)
    return [str(i) for i in x[:, 2].numpy()]

def read_graph(graph_name, base_dir: str | Path = PROJ_DIR):
    return torch.load(f'{str(base_dir)}/swissrivernetwork/benchmark/dump/graph_{graph_name}.pth')

def read_csv_train(graph_name, base_dir: str | Path = PROJ_DIR):
    return pd.read_csv(f'{base_dir}/swissrivernetwork/benchmark/dump/{graph_name}_train.csv') # DUPLICATE?!

def read_csv_prediction_train(graph_name, station, base_dir: str | Path = PROJ_DIR):
    return pd.read_csv(f'{base_dir}/swissrivernetwork/benchmark/dump/prediction/{graph_name}_lstm_{station}_train.csv')

def select_isolated_station(df, station):
    return df[['epoch_day', f'{station}_wt', f'{station}_at']].rename(columns={f'{station}_wt':'water_temperature', f'{station}_at':'air_temperature'})

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
        #df['air_temperature'] = df['air_temperature'].fillna(-1)

    # NaN in Output will be masekd
    normalizer_wt = MinMaxScaler()
    df['water_temperature'] = normalizer_wt.fit_transform(df['water_temperature'].values.reshape(-1, 1))
    if df['water_temperature'].isna().sum() > 0:
        print('[DATA PREPARATION] counted NaN values in output:', df['water_temperature'].isna().sum())

    return df

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
        self.embedding_idx=embedding_idx

        self.stations = None
        self.sequences = []
        self.sequence_lenghts = []
        self.extract_sequences()

    def extract_sequences(self):
        day_diff = self.df['epoch_day'].diff()
        breaks = day_diff != 1
        sequence_id = breaks.cumsum()
        sequences = self.df.index[breaks].values
        sequence_lengths = sequence_id.value_counts(sort=False).sort_index().values-self.window_len

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
            embs = torch.LongTensor([self.embedding_idx]*x.shape[0])

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

class SequenceFullDataset (SequenceDataset):
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

        df = self.df.iloc[start:start+length]
        return self.as_tensors(df)

class SequenceWindowedDataset (SequenceDataset):

    def __init__(self, window_len, df, embedding_idx=None):
        super().__init__(window_len, df, embedding_idx)

    def __len__(self):
        return np.sum(self.sequence_lengths)

    def __getitem__(self, idx):
        for i,length in enumerate(self.sequence_lengths):
            if idx > length:
                idx -= length
                continue

            # idx is now in sequence:
            start = self.sequences[i] + idx

            df = self.df.iloc[start:start+self.window_len] # Windowed DF
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







