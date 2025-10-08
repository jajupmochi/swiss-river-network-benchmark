"""
positional_embedding



@Author: linlin
@Date: Sep 22 2025
"""

import math
from typing import Callable, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, dim))
        self.max_len = max_len


    def forward(self, x):
        """
        x: shape (batch, seq_len, dim)
        return: same shape with positional encoding added
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f'Sequence length {seq_len} exceeds maximum length {self.max_len}.')
        pos_encoding = self.pe[:seq_len, :]  # shape (seq_len, dim)
        pos_encoding = pos_encoding.unsqueeze(0)  # shape (1, seq_len, dim)
        return x + pos_encoding


# ---- sinusoidal encoding (Vaswani 2017) ----
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, dim)
        self.register_buffer("pe", pe, persistent=False)


    def forward(self, x):
        """
        x: shape (batch, seq_len, dim)
        return: same shape with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


# %%
# Unfortunately, the implementation for the RoPE in this section is not correct. Valid RMSE goes up rather than down.
# We use Huggingface's implementation instead in the end.
# See [class ``benchmark.model.TransformerEmbeddingModel``](model.py) for details.

# ---- rotary embedding (Su et al. 2021) ----
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        dim: head_dim (must be even)
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)


    def forward(self, q, k):
        """
        q, k: (batch, seq_len, n_heads, head_dim)
        return: rotated q, k with same shape
        """
        cos = self.cos_cached[:, :q.size(1), None, :]  # (1, seq_len, 1, head_dim)
        sin = self.sin_cached[:, :q.size(1), None, :]
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length.

    Copied from [``transformers.models.roformer.modeling_roformer.RoFormerSinusoidalPositionalEmbedding``](
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L49).
    """


    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self._init_weight()  # added


    def _init_weight(self):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = self.weight.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out = torch.empty(n_pos, dim, dtype=self.weight.dtype, requires_grad=False)
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        self.weight = nn.Parameter(out, requires_grad=False)


    @torch.no_grad()
    def forward(
            self, input_ids_shape: torch.Size, past_key_values_length: int = 0,
            position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        if position_ids is None:
            bsz, seq_len = input_ids_shape[:2]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
            )
        return super().forward(position_ids)


class FlexibleTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Same as ``nn.TransformerEncoderLayer`` but allowing using self-defined attention module. Check
    [``nn.MultiheadAttention``](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) for
    the required interface of attention module.
    """


    def __init__(
            self,
            d_model: int,
            nhead: int,
            max_len: int = 5000,
            dim_feedforward: int = 2048,
            self_attn: nn.Module | None = None,
            self_attn_kwargs: dict = {},
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(max_len, d_model // nhead)
        if self_attn is not None:
            factory_kwargs = {"device": device, "dtype": dtype}
            self_attn_kwargs['attention_forward_kwargs'] = {
                'embed_positions': self.embed_positions, 'past_key_values_length': 0
            }
            self.self_attn = self_attn(
                d_model,
                nhead,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                **factory_kwargs,
                **self_attn_kwargs,
            )


class FlexibleMultiheadAttention(nn.MultiheadAttention):

    def __init__(
            self,
            embed_dim,
            num_heads,
            multi_head_attention_forward: Optional[Callable] = None,
            attention_forward_kwargs: dict = {},
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )
        if multi_head_attention_forward is None:
            self.multi_head_attention_forward = nn.functional.multi_head_attention_forward
        else:
            self.multi_head_attention_forward = multi_head_attention_forward
        self.attention_forward_kwargs = attention_forward_kwargs


    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Copied from [``nn.MultiheadAttention``](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html),
        but using ``self.multi_head_attention_forward`` instead of ``nn.functional.multi_head_attention_forward``.
        """
        why_not_fast_path = ""
        if (
                (attn_mask is not None and torch.is_floating_point(attn_mask))
                or (key_padding_mask is not None)
                and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
                key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = (
                "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
            )
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                    _arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
                "MultiheadAttention does not support NestedTensor outside of its fast path. "
                + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(  # Replaced
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                **self.attention_forward_kwargs,
            )
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(  # Replaced
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                **self.attention_forward_kwargs,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


    # # A monkey patch implementation of forward function, which replaces the call to
    # # nn.functional.multi_head_attention_forward with self.multi_head_attention_forward:
    # def forward(
    #         self,
    #         query: Tensor,
    #         key: Tensor,
    #         value: Tensor,
    #         key_padding_mask: Optional[Tensor] = None,
    #         need_weights: bool = True,
    #         attn_mask: Optional[Tensor] = None,
    #         average_attn_weights: bool = True,
    #         is_causal: bool = False,
    # ) -> tuple[Tensor, Optional[Tensor]]:
    #     # Replace the call to nn.functional.multi_head_attention_forward:
    #     old_attn_forward = nn.functional.multi_head_attention_forward
    #     F.multi_head_attention_forward = self.multi_head_attention_forward
    #     try:
    #         return super().forward(
    #             query,
    #             key,
    #             value,
    #             key_padding_mask=key_padding_mask,
    #             need_weights=need_weights,
    #             attn_mask=attn_mask,
    #             average_attn_weights=average_attn_weights,
    #             is_causal=is_causal,
    #         )
    #     finally:
    #         F.multi_head_attention_forward = old_attn_forward

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        """
        Copied from [``transformers.models.roformer.modeling_roformer.RoFormerModel.apply_rotary_position_embeddings``](
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L234).

        hidden_states [256, 90, 32]

        sinusoidal_pos [1, 1, 90, 8]
        query_layer key_layer [256, 4, 90 ,8]
        sin cos [1, 1, 90, 4]
        sin_pos cos_pos [1, 1, 90, 8]
        rotate_half_query_layer rotate_half_key_layer [256, 4, 90, 8]
        query_layer key_layer [256, 4, 90, 8]
        """
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


    @staticmethod
    def multi_head_attention_forward_with_rope(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            embed_dim_to_check: int,
            num_heads: int,
            in_proj_weight: Optional[Tensor],
            in_proj_bias: Optional[Tensor],
            bias_k: Optional[Tensor],
            bias_v: Optional[Tensor],
            add_zero_attn: bool,
            dropout_p: float,
            out_proj_weight: Tensor,
            out_proj_bias: Optional[Tensor],
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
            **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Almost identical to torch.nn.functional.multi_head_attention_forward,
        but applies RoPE to Q/K after in-projection.
        """
        tens_ops = (
            query,
            key,
            value,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            out_proj_weight,
            out_proj_bias,
        )
        if has_torch_function(tens_ops):
            return handle_torch_function(
                FlexibleMultiheadAttention.multi_head_attention_forward_with_rope,
                tens_ops,
                query,
                key,
                value,
                embed_dim_to_check,
                num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                add_zero_attn,
                dropout_p,
                out_proj_weight,
                out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                is_causal=is_causal,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                static_k=static_k,
                static_v=static_v,
                average_attn_weights=average_attn_weights,
            )

        is_batched = F._mha_shape_check(
            query, key, value, key_padding_mask, attn_mask, num_heads
        )

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        assert embed_dim == embed_dim_to_check, (
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        )
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, (
            f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        )
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], (
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )
        else:
            assert key.shape == value.shape, (
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            assert in_proj_weight is not None, (
                "use_separate_proj_weight is False but in_proj_weight is None"
            )
            q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, (
                "use_separate_proj_weight is True but q_proj_weight is None"
            )
            assert k_proj_weight is not None, (
                "use_separate_proj_weight is True but k_proj_weight is None"
            )
            assert v_proj_weight is not None, (
                "use_separate_proj_weight is True but v_proj_weight is None"
            )
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = F._in_projection(
                query,
                key,
                value,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                b_q,
                b_k,
                b_v,
            )

        # prep attention mask

        # RoPE
        # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
        embed_positions = kwargs['embed_positions']
        hidden_states_shape = (q.shape[1], q.shape[0])
        # Original code in Huggingface:
        # sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1], past_key_values_length)[None, None, :, :]
        sinusoidal_pos = embed_positions(hidden_states_shape, kwargs.get('past_key_values_length', 0))[None, None, :, :]
        q = q.permute(1, 0, 2).contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        k = k.permute(1, 0, 2).contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        q, k = FlexibleMultiheadAttention.apply_rotary_position_embeddings(sinusoidal_pos, q, k)

        # reshape q, k, v for multihead attention and make them batch first
        # # Original MHA code: # test
        # q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        q = q.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        if static_k is None:
            # # Original MHA code: # test
            # k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(bsz * num_heads, src_len, head_dim)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, (
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            )
            assert static_k.size(2) == head_dim, (
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            )
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, (
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            )
            assert static_v.size(2) == head_dim, (
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            )
            v = static_v

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                F._check_key_padding_mask(key_padding_mask, src_len, bsz)

            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        if need_weights:
            _B, _Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (is_causal and attn_mask is None), (
                "FIXME: is_causal not implemented for need_weights"
            )

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            )
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal
            )
            attn_output = (
                attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
            )

            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None


def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply RoPE (rotary positional encoding) to Q/K.
    q, k: (batch*num_heads, seq_len, head_dim)
    """

    head_dim = q.size(-1)
    assert head_dim % 2 == 0
    device, dtype = q.device, q.dtype
    pos_seq = torch.arange(q.size(1), device=device, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()

    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q, k
# %%
