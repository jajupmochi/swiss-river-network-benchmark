import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn


class LstmModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, x):
        out, hidden = self.lstm(x)  # x in [batch x sequence x features]
        target = self.linear(out)  # expect [batch x sequence x features_out(1)]
        return target


class LstmEmbeddingModel(nn.Module):

    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(
            input_size=input_size + embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, e, x):
        emb = self.embedding(e)  # e in [batch x sequence x station_id]
        x = torch.cat((emb, x), 2)  # x in [batch x sequence x features]
        out, hidden = self.lstm(x)
        target = self.linear(out)
        return target


class TransformerEmbeddingModel(nn.Module):
    def __init__(
            self, input_size: int, num_embeddings: int, embedding_size: int, num_heads: int, num_layers: int,
            dim_feedforward: int, dropout: float = 0.1,
            d_model: int | None = None, ratio_heads_to_d_model: int = 8,
            use_current_x: bool = True
    ):
        super().__init__()
        self.use_current_x = use_current_x

        # Optional station embedding:
        self.embedding = nn.Embedding(num_embeddings, embedding_size) if num_embeddings > 0 else None

        # Project input to d_model:
        self.input_proj = nn.Linear(input_size + (embedding_size if self.embedding else 0), d_model)

        # Positional Encoding:
        self.pos_embedding = nn.Parameter(torch.zeros(500, d_model))

        # Transformer Encoder:
        if d_model is not None:
            assert d_model % num_heads == 0, 'd_model must be multiple of num_heads.'
        else:
            assert d_model is None and ratio_heads_to_d_model is not None
            d_model = int(num_heads * ratio_heads_to_d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer to predict:
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )


    def forward(self, e, x):
        """
        e: [batch, seq_len] (station id)
        x: [batch, seq_len, input_size] (features per time step)
        """
        if self.embedding:
            emb = self.embedding(e)  # [batch, seq_len, embedding_size]
            x = torch.cat((emb, x), dim=-1)  # fuse embedding
        else:
            pass  # no embedding

        x = self.input_proj(x)  # [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pos_embedding[:seq_len]

        # Mask the future positions (causal):
        if self.use_current_x:
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        else:
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=0).bool()

        out = self.transformer(x, mask=mask)  # [batch, seq_len, d_model]
        target = self.linear(out)  # [batch, seq_len, 1]
        return target


class SpatioTemporalEmbeddingModel(nn.Module):

    def __init__(
            self, method, input_size, num_embeddings, embedding_size, hidden_size, num_layers, num_convs, num_heads
    ):
        super().__init__()
        self.method = method
        # self.window_len = window_len # TODO: for what?
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.stations = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_convs = num_convs
        self.num_heads = num_heads

        # Validate input        
        assert self.num_convs > 0

        # Temporal Module: based on an LSTMEmbeddingModel per Node
        self.temporal = LstmEmbeddingModel(input_size, num_embeddings, embedding_size, hidden_size, num_layers)
        self.temporal.linear = nn.Identity()  # remove linear layer

        # predefine linear layer
        # self.linear = nn.Sequential(nn.ReLU(),nn.Linear(hidden_size, 1))
        self.linear = nn.Linear(hidden_size, 1)

        if self.method == 'GCN':
            self.gconvs = nn.ModuleList(
                [gnn.GCNConv(hidden_size, hidden_size, normalize=True, add_self_loops=True) for _ in range(num_convs)]
            )

        if self.method == 'GIN':
            nn_gin = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            self.gconvs = nn.ModuleList([gnn.GINConv(nn_gin) for _ in range(num_convs)])

        if self.method == 'GAT':
            assert False, 'GAT is not supported'
            convs = []
            for i in range(num_convs):
                concat = (i != num_convs - 1)  # concat last layer
                if i == 0:
                    # First Layer
                    convs.append(gnn.GATConv(hidden_size, hidden_size, heads=num_heads, concat=concat))
                else:
                    # Mid Layer
                    convs.append(gnn.GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, concat=concat))
            self.gconvs = nn.ModuleList(convs)
            # self.linear = nn.Sequential(nn.ReLU(),nn.Linear(hidden_size*num_heads, 1)) # fix linear layer
            self.linear = nn.Linear(hidden_size * num_heads, 1)  # fix linear layer

        if self.method == 'GraphSAGE':
            self.gconvs = nn.ModuleList([gnn.SAGEConv(hidden_size, hidden_size) for _ in range(num_convs)])


    def apply_temporal_model(self, x):
        # input: batch x nodes x sequence x feature
        # output: batch x nodes x sequence x hidden_size
        hs = []
        for i in range(self.stations):
            x_node = x[:, i, :, :]
            e = torch.full((x_node.shape[0], x_node.shape[1]), i, dtype=torch.long)
            out_node = self.temporal(e, x_node)
            hs.append(out_node)
        return torch.stack(hs, dim=1)  # [batch x node x sequence x latent]


    def forward(self, x, edge_index):
        '''
        x are features in [batch x nodes x window x parameter (at)]
        '''

        # apply temporal lstms:
        hs_lstm = self.apply_temporal_model(x)

        # bring time before node (Apply GNN at each timestep):
        hs = torch.transpose(hs_lstm, 1, 2)

        # Use undirected edges
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
        # TODO: what about selfloops?

        # print('[DEBUG]: ', hs.shape, hs.dtype)
        # print('[DEBUG]: ', edge_index.shape, edge_index.dtype)

        for g in range(0, self.num_convs):
            hs = F.relu(hs)
            hs = self.gconvs[g](hs, edge_index)  # GAT

        # Restore dimensions:
        hs = torch.transpose(hs, 1, 2)

        # Predict water temperatures
        target = self.linear(hs)

        return target
