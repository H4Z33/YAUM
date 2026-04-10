"""Minimal causal transformer char model matching the RNN interface.

The trainer is agnostic to architecture: it only needs a module with
``vocab_size`` plus ``forward(embedded, hidden)`` and ``init_hidden``.
This transformer ignores the hidden argument and returns ``None`` as
its carry, so it slots into the same force-field pipeline as
``RNNCharModel`` without any trainer changes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerCharModel(nn.Module):
    """Pre-norm encoder-only transformer with a causal mask.

    The embedding matrix lives outside this module (the trainer treats
    it as the dynamical variable ``E``). Sinusoidal positional codes
    are added on every forward pass; they are stored as a buffer so
    they move with the model and get saved in the state_dict.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        max_context: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by n_heads={n_heads}"
            )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_context = max_context

        self.register_buffer(
            "pos_encoding", self._build_sinusoidal(max_context, embedding_dim)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first=True; passing
        # False here silences PyTorch's warning without changing behaviour.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, embedded: torch.Tensor, hidden=None):
        seq_len = embedded.shape[1]
        if seq_len > self.max_context:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_context={self.max_context}"
            )
        x = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.encoder(x, mask=mask, is_causal=True)
        logits = self.fc(self.norm(out))
        return logits, None

    def init_hidden(self, batch_size: int):
        # Transformers have no recurrent state. Returning None keeps the
        # trainer / force-field pipeline model-agnostic.
        return None
