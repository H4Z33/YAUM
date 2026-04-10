"""Specs for the transformer char model and the model factory."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from yaum.models import make_model
from yaum.models.rnn import RNNCharModel
from yaum.models.transformer import TransformerCharModel


def test_forward_shape_matches_rnn_contract():
    torch.manual_seed(0)
    model = TransformerCharModel(
        vocab_size=16, embedding_dim=8, hidden_dim=32,
        n_layers=2, n_heads=2, max_context=32,
    )
    embedded = torch.randn(3, 7, 8)
    logits, hidden = model(embedded, model.init_hidden(3))
    assert logits.shape == (3, 7, 16)
    assert hidden is None


def test_embedding_dim_must_divide_n_heads():
    with pytest.raises(ValueError):
        TransformerCharModel(vocab_size=10, embedding_dim=5, hidden_dim=8, n_heads=2)


def test_sequence_longer_than_max_context_raises():
    model = TransformerCharModel(
        vocab_size=5, embedding_dim=4, hidden_dim=8, n_heads=2, max_context=3
    )
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, 4), None)


def test_causal_mask_prevents_future_leakage():
    """Changing future tokens must not change earlier logits."""
    torch.manual_seed(0)
    model = TransformerCharModel(
        vocab_size=8, embedding_dim=8, hidden_dim=16,
        n_layers=1, n_heads=2, max_context=16,
    ).eval()
    x = torch.randn(1, 6, 8)
    y = x.clone()
    y[0, 4:, :] = torch.randn(2, 8)  # mutate tokens 4 and 5
    with torch.no_grad():
        logits_x, _ = model(x, None)
        logits_y, _ = model(y, None)
    # Positions 0..3 must be identical.
    assert torch.allclose(logits_x[0, :4], logits_y[0, :4], atol=1e-6)
    # Position 4 must differ because we changed token 4.
    assert not torch.allclose(logits_x[0, 4], logits_y[0, 4])


def test_backward_pass_populates_gradients():
    torch.manual_seed(0)
    model = TransformerCharModel(
        vocab_size=10, embedding_dim=8, hidden_dim=16, n_heads=2, max_context=8
    )
    criterion = nn.CrossEntropyLoss()
    embedded = torch.randn(2, 5, 8, requires_grad=True)
    logits, _ = model(embedded, None)
    targets = torch.randint(0, 10, (2, 5))
    loss = criterion(logits.reshape(-1, 10), targets.reshape(-1))
    loss.backward()
    assert embedded.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_make_model_defaults_to_rnn():
    config = {"embedding_dim": 8, "hidden_dim": 16, "n_layers": 1}
    model = make_model(vocab_size=10, config=config)
    assert isinstance(model, RNNCharModel)


def test_make_model_builds_transformer_with_defaults():
    config = {
        "model_type": "transformer",
        "embedding_dim": 8,
        "hidden_dim": 16,
        "n_layers": 2,
        "n_heads": 2,
        "context_window": 12,
    }
    model = make_model(vocab_size=10, config=config)
    assert isinstance(model, TransformerCharModel)
    assert model.max_context == 12
    assert model.n_heads == 2


def test_make_model_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_model(vocab_size=10, config={
            "model_type": "quantum", "embedding_dim": 4,
            "hidden_dim": 4, "n_layers": 1,
        })
