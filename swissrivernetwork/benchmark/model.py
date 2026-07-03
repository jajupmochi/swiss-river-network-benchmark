"""Neural network models for the Swiss River Network Benchmark.

Defines the three model families used across the benchmark and their
extrapolation (forecasting) variants:

* LSTM baselines — :class:`LstmModel`, :class:`LstmEmbeddingModel` (with a
  station embedding), plus ``Extrapo*`` variants for future-step prediction.
* Transformer — :class:`TransformerEmbeddingModel` (RoPE / learnable /
  sinusoidal positional encoding, optional mask embedding for missing days)
  and the embedding-free :class:`TransformerModel`.
* Spatio-temporal GNN — :class:`SpatioTemporalEmbeddingModel`, which runs a
  per-node temporal model (LSTM or Transformer) then message passing over the
  river graph (GCN / GIN / GAT / GraphSAGE / MPNN).

The ``ExtrapoLstm*`` / ``ExtrapoLstmEmbedding*`` factory functions dispatch on
``extrapo_mode`` (``"limo"`` = last-input-multiple-output, or
``"future_embedding"``).
"""

from typing import Any, Mapping, override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn

from swissrivernetwork.benchmark.nn import TemporalGATConv, TemporalNNConv
from swissrivernetwork.benchmark.transformer import LearnablePositionalEncoding, SinusoidalPositionalEncoding

# %% LSTM Models:


class LstmModel(nn.Module):
    """Plain per-step LSTM regressor: maps ``[B, seq, input_size]`` to a scalar per step.

    Constructor args are the LSTM ``input_size``, ``hidden_size``, and ``num_layers``;
    a ReLU + linear head projects each hidden state to one output.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        """Run the LSTM over ``x`` ``[B, seq, input_size]`` and return ``[B, seq, 1]``."""
        out, hidden = self.lstm(x)  # x in [batch x sequence x features]
        target = self.linear(out)  # expect [batch x sequence x features_out(1)]
        return target


def ExtrapoLstmModel(*args, extrapo_mode: str | None = None, **kwargs):
    """
    Factory function to create LSTM model for timewise extrapolation.

    Args:
        extrapo_mode: str | None
            Extrapolation mode. Options are:
            - 'limo': Last Input Multiple Output, use the last input to predict multiple future steps directly.
            - 'future_embedding': Use learnable future step embeddings to fill in future steps. Then use standard LSTM
                to predict all steps together.
            - 'recursive': Recursive prediction, predict one step at a time and feed it back as input for the next step.
                (not implemented yet)
            - None: Default setting (LIMO).

    """
    if extrapo_mode is None or extrapo_mode == "limo":
        return ExtrapoLstmModelLIMO(*args, **kwargs)
    elif extrapo_mode == "future_embedding":
        return ExtrapoLstmModelFEmbed(*args, **kwargs)
    else:
        raise ValueError(f"Unknown extrapo_mode: {extrapo_mode}.")


class ExtrapoLstmModelLIMO(nn.Module):
    """
    LSTM model for timewise extrapolation with LIMO (Last Input Multiple Output) strategy. The output of the last time
    step is used to predict multiple future steps directly via a linear layer.
    """

    def __init__(self, input_size, hidden_size, num_layers, future_steps: int = 1, return_hidden: bool = True):
        super().__init__()
        self.future_steps = future_steps
        self.return_hidden = return_hidden

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, future_steps))

    def forward(self, x):
        """Encode the history ``x[:, :-future_steps]`` and predict all future steps from the
        last hidden state. Returns ``[B, future_steps, 1]`` (and the full LSTM output when
        ``return_hidden``)."""
        x = x[:, : -self.future_steps]
        out, hidden = self.lstm(x)
        target = self.linear(out[:, -1, :])  # only use the last time step
        if self.return_hidden:
            return target.unsqueeze(-1), out  # [B, future_steps, 1], [B, seq_len, hidden_size]
        return target.unsqueeze(-1)  # [B, future_steps, 1]


class ExtrapoLstmModelFEmbed(nn.Module):
    """Extrapolation LSTM using learnable future-step embeddings (no station embedding).

    The history is projected to ``d_future_emb`` and concatenated with ``future_steps``
    learnable embedding vectors, then run jointly through the LSTM so future steps are
    predicted in one pass.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        future_steps: int = 1,
        d_future_emb: int = 32,  # dimension of future step embedding # todo: this can be tuned
        return_all_steps: bool = False,  # whether to return all steps or only future steps
    ):
        """Build the input projection, the ``(1, future_steps, d_future_emb)`` learnable
        future embedding, the LSTM, and the linear head. When ``return_all_steps`` is False,
        installs a postprocessor that keeps only the last ``future_steps`` outputs."""
        super().__init__()
        self.future_steps = future_steps
        self.d_future_emb = d_future_emb

        # Project input to d_future_emb to keep the dimension consistent:
        self.input_proj = nn.Linear(input_size, d_future_emb) if input_size != d_future_emb else nn.Identity()
        # Learnable embedding for future steps:
        self.future_step_embedding = nn.Parameter(torch.zeros(1, self.future_steps, d_future_emb))
        self.lstm = nn.LSTM(input_size=d_future_emb, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))

        self.target_postprocessor = None
        if not return_all_steps:
            self.target_postprocessor = lambda target: target[:, -self.future_steps :, :]  # only return future steps

    def forward(self, x):
        """Project the history, append the future-step embeddings, run the LSTM jointly, and
        return per-step predictions (only the future steps unless ``return_all_steps``)."""
        x_history = x[:, : -self.future_steps, :]  # [batch, seq_len - future_steps, input_size]
        # Project input to d_future_emb, since normally the input_size is small (e.g., 1):
        x_history = self.input_proj(x_history)  # [batch, seq_len - future_steps, d_future_emb]
        x_future = self.future_step_embedding  # [1, future_steps, d_future_emb]
        x_future = x_future.expand(x.size(0), -1, -1)  # [batch, future_steps, d_future_emb]
        x = torch.cat((x_history, x_future), dim=1)  # [batch, seq_len, d_future_emb]
        out, hidden = self.lstm(x)  # x in [batch x sequence x features]
        target = self.linear(out)  # expect [batch x sequence x features_out(1)]
        if self.target_postprocessor is not None:
            target = self.target_postprocessor(target)
        return target


# %% LSTM with Embedding Models:


class LstmEmbeddingModel(nn.Module):
    """Per-step LSTM regressor with a learnable station embedding concatenated to inputs.

    ``num_embeddings`` stations map to ``embedding_size``-dim vectors that are fused with
    the per-step features before the LSTM; a ReLU + linear head yields one output per step.
    """

    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(
            input_size=input_size + embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, e, x):
        """
        Arg:
            e: [batch x sequence] (station id)
            x: [batch x sequence x features] (features per time step, e.g., air temperature)
        """
        emb = self.embedding(e)  # e in [batch x sequence x station_id]
        x = torch.cat((emb, x), 2)  # x in [batch x sequence x features]
        out, hidden = self.lstm(x)
        target = self.linear(out)
        return target


def ExtrapoLstmEmbeddingModel(*args, extrapo_mode: str | None = None, **kwargs):
    """
    Factory function to create LSTM model with station embeddings for timewise extrapolation.

    Args:
        extrapo_mode: str | None
            Extrapolation mode. Options are:
            - 'limo': Last Input Multiple Output, use the last input to predict multiple future steps directly.
            - 'future_embedding': Use learnable future step embeddings to fill in future steps. Then use standard LSTM
                to predict all steps together.
            - 'recursive': Recursive prediction, predict one step at a time and feed it back as input for the next step.
                (not implemented yet)
            - None: Default setting (LIMO).

    """
    if extrapo_mode is None or extrapo_mode == "limo":
        return ExtrapoLstmEmbeddingModelLIMO(*args, **kwargs)
    elif extrapo_mode == "future_embedding":
        return ExtrapoLstmEmbeddingModelFEmbed(*args, **kwargs)
    else:
        raise ValueError(f"Unknown extrapo_mode: {extrapo_mode}.")


class ExtrapoLstmEmbeddingModelLIMO(nn.Module):
    """
    LSTM model with extrapolation for missing values.
    """

    def __init__(self, input_size, num_embeddings, embedding_size, hidden_size, num_layers, future_steps: int = 1):
        super().__init__()
        self.future_steps = future_steps
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(
            input_size=input_size + embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, future_steps))

    def forward(self, e, x):
        """
        The input e and x should be the full sequence, including historical observed values and "future" values to be
        predicted. This is a bit anti-intuitive, but is aligned with the Transformer model design, which uses causal
        masking.
        """
        e, x = e[:, : -self.future_steps], x[:, : -self.future_steps]
        emb = self.embedding(e)  # e in [batch x sequence x station_id]
        x = torch.cat((emb, x), 2)  # x in [batch x sequence x features]
        out, hidden = self.lstm(x)
        target = self.linear(out[:, -1, :])  # only use the last time step
        return target.unsqueeze(-1)  # [B, future_steps, 1]

        # x_t = x[:, -1, :]  # [B, D+emb]
        # preds = []
        #
        # for _ in range(future_steps):
        #     # Single step recursive forward:
        #     out, (h, c) = self.lstm(x_t.unsqueeze(1), hidden)
        #     y_pred = self.linear(out[:, -1, :])  # [B, 1]
        #
        #     preds.append(y_pred.unsqueeze(1))  # [B,1,1]
        #
        #     # Construct next input, using predicted values:
        #     # station embedding using last time step (same station id along the sequence):
        #     emb_next = self.embedding(e[:, -1])  # [B, emb]
        #     #
        #     x_t = torch.cat([y_pred, emb_next], dim=-1)  # [B, D+emb]
        #
        # preds = torch.cat(preds, dim=1)  # [B, steps, 1]
        # return preds

        target = self.linear(out)
        return target


class ExtrapoLstmEmbeddingModelFEmbed(nn.Module):
    """Extrapolation LSTM with both a station embedding and learnable future-step embeddings.

    Like :class:`ExtrapoLstmModelFEmbed`, but the station embedding is concatenated to every
    step (history and future) before the LSTM.
    """

    def __init__(
        self,
        input_size: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        future_steps: int = 1,
        d_future_emb: int = 32,  # dimension of future step embedding # todo: this can be tuned
        return_all_steps: bool = False,  # whether to return all steps or only future steps
    ):
        """Build the station embedding, input projection, learnable future-step embedding,
        LSTM, and linear head; install the future-only postprocessor unless
        ``return_all_steps``."""
        super().__init__()
        self.future_steps = future_steps
        self.d_future_emb = d_future_emb

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        # Project input to d_future_emb to keep the dimension consistent:
        self.input_proj = nn.Linear(input_size, d_future_emb) if input_size != d_future_emb else nn.Identity()
        # Learnable embedding for future steps:
        self.future_step_embedding = nn.Parameter(torch.zeros(1, self.future_steps, d_future_emb))
        self.lstm = nn.LSTM(
            input_size=d_future_emb + embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))

        self.target_postprocessor = None
        if not return_all_steps:
            self.target_postprocessor = lambda target: target[:, -self.future_steps :, :]  # only return future steps

    def forward(self, e, x):
        """Project the history, append future-step embeddings, fuse the station embedding on
        every step, run the LSTM, and return per-step predictions (future steps only unless
        ``return_all_steps``)."""
        x_history = x[:, : -self.future_steps, :]  # [batch, seq_len - future_steps, input_size]
        x_future = self.future_step_embedding  # [1, future_steps, d_future_emb]
        x_future = x_future.expand(x.size(0), -1, -1)  # [batch, future_steps, d_future_emb]
        # Project input to d_future_emb, since normally the input_size is small (e.g., 1):
        x_history = self.input_proj(x_history)  # [batch, seq_len - future_steps, d_future_emb]
        x = torch.cat((x_history, x_future), dim=1)  # [batch, seq_len, d_future_emb]

        # Keep the original station embedding for both historical and future steps:
        emb = self.embedding(e)  # e in [batch x sequence x station_id]
        x = torch.cat((emb, x), 2)  # x in [batch x sequence x d_future_emb + station_emb]
        out, hidden = self.lstm(x)
        target = self.linear(out)  # expect [batch x sequence x features_out(1)]
        if self.target_postprocessor is not None:
            target = self.target_postprocessor(target)
        return target


# %% Transformer Models:


class TransformerEmbeddingModel(nn.Module):
    """
    Transformer model.

    Args:
        input_size: int
            Number of input features per time step.
        num_embeddings: int
            Number of station embeddings.
        embedding_size: int
            Size of the station embedding vector.
        num_heads: int
            Number of attention heads for the Transformer encoder.
        num_layers: int
            Number of Transformer encoder layers.
        dim_feedforward: int
            Dimension of the feedforward network in the Transformer encoder.
        dropout: float
            Dropout rate.
        d_model: int | None
            Dimension of the model (hidden size). If None, it will be set to num_heads * ratio_heads_to_d_model.
        ratio_heads_to_d_model: int | None
            Ratio to determine d_model if d_model is None, namely d_model = num_heads * ratio_heads_to_d_model.
        max_len: int
            Maximum length of the input sequences.
        missing_value_method: str | None
            Method to handle missing values in the sequences. Options are 'mask_embedding', 'interpolation', 'zero', or
            None.
        use_current_x: bool
            Whether to use current input values for prediction. If False, the model predicts future steps based on
            historical data. Notice that the input x should contain both historical and future steps.
        positional_encoding: str
            Type of positional encoding to use. Options are 'learnable', 'sinusoidal', 'rope', or None.
        future_steps: int
            Number of future steps to predict. Only used if use_current_x is False. The input x will be split into
            historical and future parts accordingly, i.e., x[:, :-future_steps] and x[:, -future_steps:].
        d_future_emb: int
            Dimension of the learnable future step embedding. Only used if use_current_x is False.
        return_all_steps: bool
            Whether to return predictions for all time steps or only the future steps when use_current_x is False.
            When use_current_x is True, this parameter is ignored and all steps are returned. Default is False.
    """

    def __init__(
        self,
        input_size: int,
        num_embeddings: int,
        embedding_size: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        d_model: int | None = None,
        ratio_heads_to_d_model: int | None = 8,
        max_len: int = 500,
        # 'mask_embedding' or 'interpolation' or 'zero' or None:
        missing_value_method: str | None = "mask_embedding",
        use_current_x: bool = True,
        positional_encoding: str = "rope",  # 'learnable' or 'sinusoidal' or 'rope' or None
        future_steps: int = 1,  # for extrapolation model. Only works if `use_current_x` is False.
        d_future_emb: int = 32,  # dimension of future step embedding # todo: this can be tuned
        return_all_steps: bool = False,  # whether to return all steps when use_current_x is False
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
        self.use_mask_embedding = missing_value_method == "mask_embedding"
        self.positional_encoding = positional_encoding
        self.future_steps = future_steps

        # Optional station embedding:
        self.embedding = nn.Embedding(num_embeddings, embedding_size) if num_embeddings > 0 else None

        # Project input to d_model:
        self.input_proj = nn.Linear(input_size + (embedding_size if self.embedding else 0), d_model)

        if d_model is not None:
            assert d_model % num_heads == 0, "d_model must be multiple of num_heads."
        else:
            assert d_model is None and ratio_heads_to_d_model is not None
            d_model = int(num_heads * ratio_heads_to_d_model)

        # Positional Encoding:
        if positional_encoding == "rope":
            from transformers import RoFormerConfig, RoFormerModel

            config = RoFormerConfig(
                # todo: remove the entire word_embeddings from the model, in case that d_model is large.
                vocab_size=1,  # Avoid building "big" word_embeddings. Default 50000. Can not set to 0, unfortunately.
                hidden_size=d_model,  # embedding_size = hidden_size if embedding_size is None else embedding_size
                num_attention_heads=num_heads,
                num_hidden_layers=num_layers,
                intermediate_size=dim_feedforward,
                hidden_dropout_prob=dropout,  # dropout for fully connected layers. Default 0.1
                attention_probs_dropout_prob=dropout,  # dropout for attention probabilities. Default 0.1
                max_position_embeddings=max_len,
                is_decoder=False,  # True for decoder, False for encoder. Default False.
                use_cache=True,  # Whether the model should return the last key/values attentions. Default True.
                rotary_value=False,  # If True, Use RoPE for value as well. Default False.
                pad_token_id=0,  # Padding token id. Default 0.
            )
            self.transformer = RoFormerModel(config)
        else:
            if positional_encoding == "learnable":
                self.pos_embedding = LearnablePositionalEncoding(d_model, max_len=max_len)
            elif positional_encoding == "sinusoidal":
                self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
            else:
                raise ValueError(f"Unknown positional_encoding: {positional_encoding}.")

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer
        # dim_out_features = 1 if use_current_x else future_steps
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(d_model, 1))

        if self.use_mask_embedding:
            # Learnable embedding for missing values:
            # Caution: it is possible that ``mask_embedding`` is not used but still in the state_dict (e.g., when
            # ``missing_value_method`` is set incorrectly). Make sure to match this with the correct dataset settings
            # (e.g., the ones that generate the corresponding ``time_masks``).
            self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        self.target_postprocessor = None
        if not self.use_current_x:
            # Learnable embedding for future steps:
            self.future_step_embedding = nn.Parameter(torch.zeros(1, self.future_steps, d_future_emb))
            if self.embedding:
                self.future_proj = nn.Linear(embedding_size + d_future_emb, d_model)
            else:
                if d_model == d_future_emb:
                    self.future_proj = nn.Identity()
                else:
                    self.future_proj = nn.Linear(d_future_emb, d_model)
            if not return_all_steps:
                self.target_postprocessor = lambda target: target[
                    :, -self.future_steps :, :
                ]  # only return future steps

    def forward(self, e, x, time_masks=None, pad_masks=None):
        """
        e: [batch, seq_len] (station id)
        x: [batch, seq_len, input_size] (features per time step)
        time_masks: [batch, seq_len] (Masks for missing time stamps (at the middle) along the consecutive time series.
        True means missing value, optional)
        pad_masks: [batch, seq_len] (Padding masks for short sequences. True means padding position, optional)
        """
        if x.isnan().any():
            raise ValueError("Input contains NaN values! QK matrix will be corrupted.")

        if self.use_current_x:
            # Current value based prediction:
            # Station embedding:
            if self.embedding:
                emb = self.embedding(e)  # [batch, seq_len, embedding_size]
                x = torch.cat((emb, x), dim=-1)  # fuse embedding

            x = self.input_proj(x)  # [batch, seq_len, d_model]
            seq_len = x.size(1)
        else:
            # Separate historical observed values and future values to be predicted:
            x_history = x[:, : -self.future_steps, :]  # [batch, seq_len - future_steps, input_size]
            # Add future step embeddings if needed:
            x_future = self.future_step_embedding  # [1, future_steps, d_future_emb]
            x_future = x_future.expand(x.size(0), -1, -1)  # [batch, future_steps, d_future_emb]

            # Station embedding:
            if self.embedding:
                emb = self.embedding(e)  # [batch, seq_len, embedding_size]
                x_history = torch.cat((emb[:, : -self.future_steps, :], x_history), dim=-1)  # fuse embedding
                # [batch, future_steps, d_emb + d_future_emb]:
                x_future = torch.cat((emb[:, -self.future_steps :, :], x_future), dim=-1)

            x_history = self.input_proj(x_history)  # [batch, seq_len - future_steps, d_model]
            x_future = self.future_proj(x_future)  # [batch, future_steps, d_model]

            x = torch.cat((x_history, x_future), dim=1)  # [batch, seq_len, d_model]
            seq_len = x.size(1)

        if self.positional_encoding in ["learnable", "sinusoidal"]:
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
            # Here we try to use full attention among all steps. Notice that learnable future step embeddings are used.
            causal_mask = torch.zeros((seq_len, seq_len), device=x.device).bool()
            # Here we do not allow all future steps to attend to each other (only the ones before the current step):
            # causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()

        # # Check mask validity:  comment this out for now because it is too slow. todo: optimize it.
        # if not self.use_mask_embedding:
        #     self.check_mask_validity(causal_mask, time_masks, x.size(0), seq_len)

        if self.positional_encoding == "rope":
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
        if self.target_postprocessor is not None:
            target = self.target_postprocessor(target)
        return target

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load a checkpoint, patching legacy keys before delegating to ``nn.Module``.

        Renames an old ``pos_embedding`` tensor to ``pos_embedding.pe`` and drops a stale
        ``mask_embedding`` entry when this instance does not use one, so checkpoints saved
        under earlier code / settings still load.
        """
        if "pos_embedding" in state_dict and "pos_embedding.pe" not in state_dict:
            # Convert old positional encoding to new format:
            state_dict["pos_embedding.pe"] = state_dict.pop("pos_embedding")
        if not self.use_mask_embedding and "mask_embedding" in state_dict:
            # Remove mask embedding if not used:
            # CAUTION: this is a monkey patch. Make sure this is what you want.
            state_dict.pop("mask_embedding")
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
                        f"Invalid mask: batch={b}, position={t} is fully masked (causal + padding mask overlap)."
                    )

    # def compute_time_steps_since_last_observation(self, time_stamps):
    #     """
    #     time_stamps: [batch, seq_len] (time in days)
    #     return: time_masks: [batch, seq_len] (True means missing value)
    #     """
    #     time_since = torch.full_like(obs_mask, -1, dtype=torch.float)  # or zeros
    #     for b in range(B):
    #         last = -1e9
    #         for t in range(L):
    #             if obs_mask[b, t]:
    #                 last = 0.
    #                 time_since[b, t] = 0.
    #             else:
    #                 last += 1.
    #                 time_since[b, t] = last


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


class TransformerModel(TransformerEmbeddingModel):
    """Embedding-free Transformer: same as :class:`TransformerEmbeddingModel` but takes no
    station id (call with ``num_embeddings=0``)."""

    @override
    def forward(self, x, time_masks=None, pad_masks=None):
        """Forward without a station embedding (passes ``e=None`` to the parent)."""
        return super().forward(None, x, time_masks=None, pad_masks=None)


class SpatioTemporalEmbeddingModel(nn.Module):
    """Spatio-temporal GNN: a per-station temporal encoder followed by graph message passing.

    Each of the ``num_embeddings`` stations (graph nodes) is first encoded independently in
    time by ``temporal_func`` (``"lstm_embedding"`` or ``"transformer_embedding"``, with its
    output head replaced by identity so it emits hidden states). The resulting per-node,
    per-timestep hidden states are then propagated over the (undirected, self-looped) river
    graph by ``num_convs`` layers of ``method`` (``"GCN"``, ``"GIN"``, ``"GAT"``,
    ``"GraphSAGE"``, or ``"MPNN"``) before a final linear head predicts water temperature.
    Supports both current-step regression and forecasting (``use_current_x`` /
    ``future_steps`` / ``extrapo_mode`` via ``kwargs``).
    """

    def __init__(
        self,
        method,
        input_size,
        num_embeddings,
        embedding_size,
        hidden_size,
        num_layers,
        num_convs,
        num_heads,
        temporal_func: str = "lstm_embedding",  # 'lstm_embedding' or 'transformer_embedding'
        **kwargs,
    ):
        """Construct the temporal encoder and the ``num_convs`` graph-conv layers for
        ``method``, wiring output dimensions to the forecasting mode. See the class
        docstring for the roles of ``method``, ``temporal_func``, and the ``kwargs``
        (``use_current_x``, ``future_steps``, ``extrapo_mode``, ``use_station_embedding``,
        and method-specific keys such as ``edge_hidden_size`` for MPNN)."""
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
        self.temporal_func = temporal_func
        self.use_current_x: bool = kwargs["use_current_x"]
        self.future_steps = kwargs.get("future_steps", 1)
        # whether to return all steps when use_current_x is False:
        self.return_all_steps = kwargs.get("return_all_steps", False)
        self.use_station_embedding = kwargs.get("use_station_embedding", True)
        self.kwargs = kwargs

        self.input_emb_for_temporal = not (temporal_func == "lstm_embedding" and not self.use_station_embedding)

        # Validate input
        assert self.num_convs > 0, "num_convs must be positive."

        # Input processor:
        if (
            not self.use_current_x
            and temporal_func == "lstm_embedding"
            and kwargs.get("extrapo_mode") in [None, "limo"]
        ):
            self.input_preprocessor = lambda x: x[..., : -self.future_steps, :]
        else:
            self.input_preprocessor = None

        # Temporal Module: based on an LSTMEmbeddingModel per Node
        if temporal_func == "lstm_embedding":
            if self.use_current_x:  # non-extrapolation scenario:
                if self.use_station_embedding:
                    self.temporal = LstmEmbeddingModel(
                        input_size, num_embeddings, embedding_size, hidden_size, num_layers
                    )
                else:
                    self.temporal = LstmModel(input_size, hidden_size, num_layers)
            elif kwargs.get("extrapo_mode") in [None, "limo"]:
                # self.input_preprocessor will remove the future steps from input x, so that the LSTM only sees
                # historical data without information leakage:
                if not self.use_station_embedding:
                    raise NotImplementedError("LstmModel does not support extrapolation yet.")
                self.temporal = LstmEmbeddingModel(input_size, num_embeddings, embedding_size, hidden_size, num_layers)
            elif kwargs.get("extrapo_mode") == "future_embedding":
                if not self.use_station_embedding:
                    raise NotImplementedError("ExtrapoLstmEmbeddingModelFEmbed does not support no embedding yet.")
                self.temporal = ExtrapoLstmEmbeddingModelFEmbed(
                    input_size,
                    num_embeddings,
                    embedding_size,
                    hidden_size,
                    num_layers,
                    future_steps=self.future_steps,
                    d_future_emb=kwargs.get("d_future_emb", 32),
                    return_all_steps=True,
                )
            else:
                raise ValueError(f"Unknown extrapo_mode: {kwargs.get('extrapo_mode')}.")
        elif temporal_func == "transformer_embedding":
            self.temporal = TransformerEmbeddingModel(
                input_size=input_size,
                num_embeddings=num_embeddings if kwargs.get("use_station_embedding", True) else 0,
                embedding_size=embedding_size,
                num_heads=kwargs["num_t_heads"],
                num_layers=num_layers,
                dim_feedforward=kwargs["dim_feedforward"],
                dropout=kwargs["dropout"],
                d_model=hidden_size,
                max_len=kwargs["max_len"],
                missing_value_method=kwargs["missing_value_method"],
                use_current_x=self.use_current_x,
                positional_encoding=kwargs.get("positional_encoding", "rope"),
                future_steps=self.future_steps,
            )
            if not self.use_current_x:
                self.temporal.target_postprocessor = None  # This is equivalent to returning all steps
        else:
            raise ValueError(f"Unknown temporal_func: {temporal_func}.")
        self.temporal.linear = nn.Identity()  # remove linear layer

        # predefine linear layer
        # self.linear = nn.Sequential(nn.ReLU(),nn.Linear(hidden_size, 1))
        if not self.use_current_x and (
            temporal_func == "lstm_embedding" and kwargs.get("extrapo_mode") in [None, "limo"]
        ):
            dim_out_features = self.future_steps
        else:
            dim_out_features = 1
        self.linear = nn.Linear(hidden_size, dim_out_features)

        if self.method == "GCN":
            self.gconvs = nn.ModuleList(
                [gnn.GCNConv(hidden_size, hidden_size, normalize=True, add_self_loops=True) for _ in range(num_convs)]
            )

        elif self.method == "GIN":
            nn_gin = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            self.gconvs = nn.ModuleList([gnn.GINConv(nn_gin) for _ in range(num_convs)])

        elif self.method == "GAT":
            # assert False, 'GAT is not supported'
            convs = []
            for i in range(num_convs):
                concat = i == (num_convs - 1)  # concat last layer
                convs.append(
                    TemporalGATConv(
                        hidden_size,
                        hidden_size,
                        heads=num_heads,
                        concat=concat,
                        add_self_loops=False,  # self-loops are already added
                    )
                )
            self.gconvs = nn.ModuleList(convs)
            # self.linear = nn.Sequential(nn.ReLU(),nn.Linear(hidden_size*num_heads, 1)) # fix linear layer
            self.linear = nn.Linear(hidden_size * num_heads, dim_out_features)  # fix linear layer

        elif self.method == "GraphSAGE":
            self.gconvs = nn.ModuleList([gnn.SAGEConv(hidden_size, hidden_size) for _ in range(num_convs)])

        elif self.method == "MPNN":
            self.edge_hidden_size = kwargs.get("edge_hidden_size")
            edge_network = nn.Sequential(
                nn.Linear(1, self.edge_hidden_size),
                nn.ReLU(),
                nn.Linear(self.edge_hidden_size, hidden_size * hidden_size),
            )
            self.gconvs = nn.ModuleList(
                [TemporalNNConv(hidden_size, hidden_size, edge_network) for _ in range(num_convs)]
            )

        else:
            raise ValueError(f"Unknown method: {self.method}.")

    def apply_temporal_model(self, x):
        """Run the shared temporal encoder on each station independently and stack the results.

        Input ``[B, nodes, seq, feature]`` -> output ``[B, nodes, seq, hidden_size]``; a
        constant per-node station id is fed as the embedding index when the encoder expects
        one.
        """
        # input: batch x nodes x sequence x feature
        # output: batch x nodes x sequence x hidden_size
        hs = []
        for i in range(self.stations):
            x_node = x[:, i, :, :]
            if self.input_emb_for_temporal:
                e = torch.full((x_node.shape[0], x_node.shape[1]), i, dtype=torch.long).to(x.device)
                out_node = self.temporal(e, x_node)
            else:
                out_node = self.temporal(x_node)
            hs.append(out_node)
        return torch.stack(hs, dim=1)  # [batch x node x sequence x latent]

    def postprocess_target(self, target):
        """Apply the final linear head to the post-GNN hidden states and shape the output.

        For current-step regression, projects every step. For forecasting, selects the last
        step (LIMO) or the future steps (future-embedding / transformer) as dictated by
        ``temporal_func`` / ``extrapo_mode`` / ``return_all_steps`` before/after projection.
        """
        if self.use_current_x:
            target = self.linear(target)
        else:
            if self.temporal_func == "lstm_embedding":
                if self.kwargs.get("extrapo_mode") in [None, "limo"]:
                    target = target[..., -1, :]  # only use the last time step
                    target = self.linear(target)
                    target = target.unsqueeze(-1)
                elif self.kwargs.get("extrapo_mode") == "future_embedding":
                    target = self.linear(target)
                    if not self.return_all_steps:
                        target = target[..., -self.future_steps :, :]  # only use the future steps
                else:
                    raise ValueError(f"Unknown extrapo_mode: {self.kwargs.get('extrapo_mode')}.")
            elif self.temporal_func == "transformer_embedding":
                target = self.linear(target)
                if not self.return_all_steps:
                    target = target[..., -self.future_steps :, :]  # only use the future steps
            else:
                raise ValueError(f"Unknown temporal_func: {self.temporal_func}.")
        return target

    def forward(self, x, edge_index):
        """
        x are features in [batch x nodes x window x parameter (at)]
        """
        if self.input_preprocessor is not None:
            x = self.input_preprocessor(x)

        # apply temporal models:
        hs = self.apply_temporal_model(x)

        # bring time before node (Apply GNN at each timestep):
        hs = torch.transpose(hs, 1, 2)

        # Use undirected edges
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
        # TODO: what about selfloops?  why?

        # print('[DEBUG]: ', hs.shape, hs.dtype)
        # print('[DEBUG]: ', edge_index.shape, edge_index.dtype)

        extra_inputs = {}
        if self.method == "MPNN":
            # For MPNN, we need edge attributes, e.g., edge lengths.
            # edge_attr should be of shape [num_edges, num_edge_features] (Static for all samples and time steps).
            # Here we use a dummy edge_attr of ones
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)
            extra_inputs["edge_attr"] = edge_attr

        for g in range(0, self.num_convs):
            hs = F.relu(hs)
            hs = self.gconvs[g](hs, edge_index, **extra_inputs)  # GAT

        # Restore dimensions:
        hs = torch.transpose(hs, 1, 2)  # [B, n_stations, T, hidden_size]

        # Predict water temperatures:
        target = self.postprocess_target(hs)

        return target
