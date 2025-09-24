"""
positional_embedding



@Author: linlin
@Date: Sep 22 2025
"""

import math

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.nn.Parameter(torch.zeros(max_len, dim))
        self.max_len = max_len


    def forward(self, x):
        """
        x: shape (batch, seq_len, dim)
        return: same shape with positional encoding added
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        pos_encoding = self.pos_embedding[:seq_len, :]  # shape (seq_len, dim)
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
