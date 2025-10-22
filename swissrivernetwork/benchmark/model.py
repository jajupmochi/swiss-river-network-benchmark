from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn

from swissrivernetwork.benchmark.nn import TemporalNNConv, TemporalGATConv
from swissrivernetwork.benchmark.transformer import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding
)


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
    """
    Transformer model.
    """


    def __init__(
            self, input_size: int, num_embeddings: int, embedding_size: int, num_heads: int, num_layers: int,
            dim_feedforward: int, dropout: float = 0.1,
            d_model: int | None = None, ratio_heads_to_d_model: int | None = 8,
            max_len: int = 500,
            # 'mask_embedding' or 'interpolation' or 'zero' or None:
            missing_value_method: str | None = 'mask_embedding',
            use_current_x: bool = True,
            positional_encoding: str = 'rope'  # 'learnable' or 'sinusoidal' or 'rope' or None
    ):
        """
        Parameters
        ----------
        ratio_heads_to_d_model : int | None
            If d_model is None, then d_model = num_heads * ratio_heads_to_d_model.
            If d_model is given, then ratio_heads_to_d_model is ignored.
        """
        super().__init__()
        self.use_current_x = use_current_x
        self.use_mask_embedding = (missing_value_method == 'mask_embedding')
        self.positional_encoding = positional_encoding

        # Optional station embedding:
        self.embedding = nn.Embedding(num_embeddings, embedding_size) if num_embeddings > 0 else None

        # Project input to d_model:
        self.input_proj = nn.Linear(input_size + (embedding_size if self.embedding else 0), d_model)

        if d_model is not None:
            assert d_model % num_heads == 0, 'd_model must be multiple of num_heads.'
        else:
            assert d_model is None and ratio_heads_to_d_model is not None
            d_model = int(num_heads * ratio_heads_to_d_model)

        # Positional Encoding:
        if positional_encoding == 'rope':
            from transformers import RoFormerModel, RoFormerConfig
            config = RoFormerConfig(
                hidden_size=d_model,
                num_attention_heads=num_heads,
                num_hidden_layers=num_layers,
                intermediate_size=dim_feedforward,
                hidden_dropout_prob=dropout,  # dropout for fully connected layers. Default 0.1
                attention_probs_dropout_prob=dropout,  # dropout for attention probabilities. Default 0.1
                max_position_embeddings=max_len,
                is_decoder=False,  # True for decoder, False for encoder. Default False.
                use_cache=True,  # Whether the model should return the last key/values attentions. Default True.
                rotary_value=False,  # If True, Use RoPE for value as well. Default False.
                pad_token_id=0  # Padding token id. Default 0.
            )
            self.transformer = RoFormerModel(config)
        else:
            if positional_encoding == 'learnable':
                self.pos_embedding = LearnablePositionalEncoding(d_model, max_len=max_len)
            elif positional_encoding == 'sinusoidal':
                self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
            else:
                raise ValueError(f'Unknown positional_encoding: {positional_encoding}.')

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        if self.use_mask_embedding:
            # Learnable embedding for missing values:
            # Caution: it is possible that ``mask_embedding`` is not used but still in the state_dict (e.g., when
            # ``missing_value_method`` is set incorrectly). Make sure to match this with the correct dataset settings
            # (e.g., the ones that generate the corresponding ``time_masks``).
            self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))


    def forward(self, e, x, time_masks=None, pad_masks=None):
        """
        e: [batch, seq_len] (station id)
        x: [batch, seq_len, input_size] (features per time step)
        time_masks: [batch, seq_len] (Masks for missing time stamps (at the middle) along the consecutive time series.
        True means missing value, optional)
        pad_masks: [batch, seq_len] (Padding masks for short sequences. True means padding position, optional)
        """
        if x.isnan().any():
            raise ValueError('Input contains NaN values! QK matrix will be corrupted.')

        # Station embedding:
        if self.embedding:
            emb = self.embedding(e)  # [batch, seq_len, embedding_size]
            x = torch.cat((emb, x), dim=-1)  # fuse embedding

        x = self.input_proj(x)  # [batch, seq_len, d_model]
        seq_len = x.size(1)

        if self.positional_encoding in ['learnable', 'sinusoidal']:
            x = self.pos_embedding(x)  # add positional encoding

        if time_masks is not None and self.use_mask_embedding:
            # Add mask embedding to the input at missing value positions:
            x = x + time_masks.unsqueeze(-1) * self.mask_embedding  # add mask
            # Replace missing values with mask embedding:
            # x = torch.where(time_masks.unsqueeze(-1), self.mask_embedding, x)  # substitute missing values

        # Mask the future positions (causal) - True means to ignore / to mask:
        if self.use_current_x:
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        else:
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=0).bool()

        # # Check mask validity:  comment this out for now because it is too slow. todo: optimize it.
        # if not self.use_mask_embedding:
        #     self.check_mask_validity(causal_mask, time_masks, x.size(0), seq_len)

        if self.positional_encoding == 'rope':
            # HuggingFace RoFormer expects input_ids or embeddings:

            # Construct attention mask for HuggingFace:
            # - For causal mask of shape [B, L, L], it will be expanded to [B, 1, L, L] internally by
            # function ``get_extended_attention_mask`` in ``RoFormerModel.forward()``, and then added to attention
            # scores after attention before softmax in ``RoFormerSelfAttention.forward()``.
            # - For padding mask of shape [B, L], it will be expanded to [B, 1, 1, L] internally by
            # function ``get_extended_attention_mask`` in ``RoFormerModel.forward()``, and then added to attention
            # scores after attention before softmax in ``RoFormerSelfAttention.forward()``.
            # Use boolean mask instead of int / float mask to benefit from potential lazy broadcasting.
            # This does not work for now because RoFormerModel still expands the mask with full size memory.
            hf_causal_mask = (~causal_mask).bool()  # [L, L], 1 = keep, 0 = mask
            hf_mask = hf_causal_mask[None, :, :]  # [1, seq_len, seq_len]
            # Combine with time_masks (missing mask)
            if time_masks is not None and not self.use_mask_embedding:
                # 1 = keep, 0 = mask
                hf_time_mask = (~time_masks).bool().unsqueeze(1)
                hf_mask = hf_mask & hf_time_mask  # [batch, seq_len, seq_len]
            if pad_masks is not None:
                # 1 = keep, 0 = mask
                hf_pad_mask = (~pad_masks).bool().unsqueeze(1)
                hf_mask = hf_mask & hf_pad_mask  # [batch, seq_len, seq_len]

            # [batch, seq_len, d_model]:
            out = self.transformer(inputs_embeds=x, attention_mask=hf_mask).last_hidden_state
        else:
            # For both ``mask`` and ``src_key_padding_mask``, True values are positions that will be ignored.
            # False = keep, True = mask
            # Notice that the values at the masked positions in ``out`` will still be computed, which should be
            # ignored in some subsequent process and the loss calculation:
            src_key_padding_mask = None if self.use_mask_embedding else time_masks
            if pad_masks is not None:
                if src_key_padding_mask is None:
                    src_key_padding_mask = pad_masks
                else:
                    src_key_padding_mask = src_key_padding_mask | pad_masks

            out = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        # [batch, seq_len, 1]  token-wise projection, masked values do not affect others at this step:
        target = self.linear(out)
        return target


    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        if 'pos_embedding' in state_dict and 'pos_embedding.pe' not in state_dict:
            # Convert old positional encoding to new format:
            state_dict['pos_embedding.pe'] = state_dict.pop('pos_embedding')
        if not self.use_mask_embedding and 'mask_embedding' in state_dict:
            # Remove mask embedding if not used:
            # CAUTION: this is a monkey patch. Make sure this is what you want.
            state_dict.pop('mask_embedding')
        super().load_state_dict(state_dict, strict, assign)


    @staticmethod
    def check_mask_validity(mask, src_key_padding_mask, batch_size, seq_len):
        """
        Check that the combination of causal mask and padding mask does not fully mask any position.
        mask: [L, L] or [B*heads, L, L] (True means masked)
        src_key_padding_mask: [B, L] (True means masked)
        batch_size: B
        seq_len: L
        1. If mask is [B*heads, L, L], we only check the first batch.
        2. If src_key_padding_mask is None, we only check the causal mask.
        3. If src_key_padding_mask is given, we check the combination of both masks.
        4. If any position is fully masked, raise ValueError.
        """
        # Convert mask to [L, L] if needed. If mask is [B*heads, L, L], we only check the first batch:
        if mask.dim() == 3:
            mask = mask[0]

        for b in range(batch_size):
            for t in range(seq_len):
                # If this position is already padding, skip the check:
                if src_key_padding_mask is not None and src_key_padding_mask[b, t]:
                    continue

                # mask[t] means the causal mask for position t: [L] (True means masked):
                row_mask = mask[t] | (src_key_padding_mask[b] if src_key_padding_mask is not None else 0)

                if row_mask.all():  # All positions are masked
                    raise ValueError(
                        f"Invalid mask: batch={b}, position={t} is fully masked "
                        f"(causal + padding mask overlap)."
                    )


    def compute_time_steps_since_last_observation(self, time_stamps):
        """
        time_stamps: [batch, seq_len] (time in days)
        return: time_masks: [batch, seq_len] (True means missing value)
        """
        time_since = torch.full_like(obs_mask, -1, dtype=torch.float)  # or zeros
        for b in range(B):
            last = -1e9
            for t in range(L):
                if obs_mask[b, t]:
                    last = 0.
                    time_since[b, t] = 0.
                else:
                    last += 1.
                    time_since[b, t] = last


# # The self defined version for RoPE:
# class TransformerEmbeddingModel(nn.Module):
#     def __init__(
#             self, input_size: int, num_embeddings: int, embedding_size: int, num_heads: int, num_layers: int,
#             dim_feedforward: int, dropout: float = 0.1,
#             d_model: int | None = None, ratio_heads_to_d_model: int = 8,
#             max_len: int = 500,
#             missing_value_method: str = 'mask_embedding',  # 'mask_embedding' or None
#             use_current_x: bool = True,
#             positional_encoding: str = 'rope'  # 'learnable' or 'sinusoidal' or 'rope' or None
#     ):
#         super().__init__()
#         self.use_current_x = use_current_x
#         self.use_mask_embedding = (missing_value_method == 'mask_embedding')
#         self.positional_encoding = positional_encoding
#
#         # Optional station embedding:
#         self.embedding = nn.Embedding(num_embeddings, embedding_size) if num_embeddings > 0 else None
#
#         # Project input to d_model:
#         self.input_proj = nn.Linear(input_size + (embedding_size if self.embedding else 0), d_model)
#
#         # Positional Encoding:
#         if positional_encoding == 'learnable':
#             self.pos_embedding = LearnablePositionalEncoding(d_model, max_len=max_len)  # [1, max_len, d_model]
#         elif positional_encoding == 'sinusoidal':
#             self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_len)  # [1, max_len, d_model]
#         # elif positional_encoding == 'rope':  # test
#         #     self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_len)  # [1, max_len, d_model]
#
#         # Transformer Encoder:
#         if d_model is not None:
#             assert d_model % num_heads == 0, 'd_model must be multiple of num_heads.'
#         else:
#             assert d_model is None and ratio_heads_to_d_model is not None
#             d_model = int(num_heads * ratio_heads_to_d_model)
#         if positional_encoding == 'rope':
#             from swissrivernetwork.benchmark.transformer import (
#                 FlexibleMultiheadAttention, FlexibleTransformerEncoderLayer,
#             )
#             self_attn = FlexibleMultiheadAttention
#             self_attn_kwargs = {
#                 'multi_head_attention_forward': FlexibleMultiheadAttention.multi_head_attention_forward_with_rope
#             }
#             encoder_layer = FlexibleTransformerEncoderLayer(
#                 d_model=d_model, nhead=num_heads, max_len=max_len, dim_feedforward=dim_feedforward,
#                 self_attn=self_attn, self_attn_kwargs=self_attn_kwargs,
#                 dropout=dropout, batch_first=True
#             )
#         else:
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
#             )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Final linear layer to predict:
#         self.linear = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(d_model, 1)
#         )
#
#         if self.use_mask_embedding:
#             # Learnable embedding for missing values:
#             self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
#
#
#     def forward(self, e, x, time_masks=None):
#         """
#         e: [batch, seq_len] (station id)
#         x: [batch, seq_len, input_size] (features per time step)
#         """
#         if x.isnan().any():
#             raise ValueError('Input contains NaN values! QK matrix will be corrupted.')
#
#         if self.embedding:
#             emb = self.embedding(e)  # [batch, seq_len, embedding_size]
#             x = torch.cat((emb, x), dim=-1)  # fuse embedding
#         else:
#             pass  # no embedding
#
#         x = self.input_proj(x)  # [batch, seq_len, d_model]
#         seq_len = x.size(1)
#         if self.positional_encoding in ['learnable', 'sinusoidal']:
#             x = self.pos_embedding(x)  # add positional encoding
#         # if self.positional_encoding in ['rope']:  # test
#         #     x = self.pos_embedding(x)  # add positional encoding
#         if time_masks is not None and self.use_mask_embedding:
#             # Add mask embedding to the input at missing value positions:
#             x = x + time_masks.unsqueeze(-1) * self.mask_embedding  # add mask
#             # Replace missing values with mask embedding:
#             # x = torch.where(time_masks.unsqueeze(-1), self.mask_embedding, x)  # substitute missing values
#
#         # Mask the future positions (causal) - True means to ignore / to mask:
#         if self.use_current_x:
#             mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
#         else:
#             mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=0).bool()
#
#         # Check mask validity:
#         if not self.use_mask_embedding:
#             self.check_mask_validity(mask, time_masks, x.size(0), seq_len)
#
#         # For both ``mask`` and ``src_key_padding_mask``, True values are positions that will be ignored.
#         # Notice that the values at the masked positions in ``out`` will still be computed, which should be
#         # ignored in some subsequent process and the loss calculation:
#         src_key_padding_mask = None if self.use_mask_embedding else time_masks
#         out = self.transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)  # [batch, seq_len, d_model]
#         # [batch, seq_len, 1]  token-wise projection, masked values do not affect others at this step:
#         target = self.linear(out)
#         return target


class SpatioTemporalEmbeddingModel(nn.Module):

    def __init__(
            self, method, input_size, num_embeddings, embedding_size, hidden_size, num_layers, num_convs, num_heads,
            **kwargs
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
        assert self.num_convs > 0, 'num_convs must be positive.'

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

        elif self.method == 'GIN':
            nn_gin = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            self.gconvs = nn.ModuleList([gnn.GINConv(nn_gin) for _ in range(num_convs)])

        elif self.method == 'GAT':
            # assert False, 'GAT is not supported'
            convs = []
            for i in range(num_convs):
                concat = (i == (num_convs - 1))  # concat last layer
                convs.append(
                    TemporalGATConv(
                        hidden_size, hidden_size, heads=num_heads, concat=concat,
                        add_self_loops=False  # self-loops are already added
                    )
                )
            self.gconvs = nn.ModuleList(convs)
            # self.linear = nn.Sequential(nn.ReLU(),nn.Linear(hidden_size*num_heads, 1)) # fix linear layer
            self.linear = nn.Linear(hidden_size * num_heads, 1)  # fix linear layer

        elif self.method == 'GraphSAGE':
            self.gconvs = nn.ModuleList([gnn.SAGEConv(hidden_size, hidden_size) for _ in range(num_convs)])

        elif self.method == 'MPNN':
            self.edge_hidden_size = kwargs.get('edge_hidden_size')
            edge_network = nn.Sequential(
                nn.Linear(1, self.edge_hidden_size),
                nn.ReLU(),
                nn.Linear(self.edge_hidden_size, hidden_size * hidden_size)
            )
            self.gconvs = nn.ModuleList(
                [TemporalNNConv(hidden_size, hidden_size, edge_network) for _ in range(num_convs)]
            )

        else:
            raise ValueError(f'Unknown method: {self.method}.')


    def apply_temporal_model(self, x):
        # input: batch x nodes x sequence x feature
        # output: batch x nodes x sequence x hidden_size
        hs = []
        for i in range(self.stations):
            x_node = x[:, i, :, :]
            e = torch.full((x_node.shape[0], x_node.shape[1]), i, dtype=torch.long).to(x.device)
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
        # TODO: what about selfloops?  why?

        # print('[DEBUG]: ', hs.shape, hs.dtype)
        # print('[DEBUG]: ', edge_index.shape, edge_index.dtype)

        extra_inputs = {}
        if self.method == 'MPNN':
            # For MPNN, we need edge attributes, e.g., edge lengths.
            # edge_attr should be of shape [num_edges, num_edge_features] (Static for all samples and time steps).
            # Here we use a dummy edge_attr of ones
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)
            extra_inputs['edge_attr'] = edge_attr

        for g in range(0, self.num_convs):
            hs = F.relu(hs)
            hs = self.gconvs[g](hs, edge_index, **extra_inputs)  # GAT

        # Restore dimensions:
        hs = torch.transpose(hs, 1, 2)  # [B, n_stations, T, hidden_size]

        # Predict water temperatures
        target = self.linear(hs)

        return target
