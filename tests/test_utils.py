"""Specs for batch sampling."""
from __future__ import annotations

import pytest
import torch

from yaum.core.utils import get_batch


def test_batch_shapes_and_next_token_relation():
    torch.manual_seed(0)
    data = torch.arange(1000)
    x, y = get_batch(data, context_window=8, batch_size=4, target_device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is always the next token after x.
    assert torch.equal(y, x + 1)


def test_batch_accepts_python_list():
    data = list(range(200))
    x, y = get_batch(data, context_window=5, batch_size=3, target_device=torch.device("cpu"))
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert torch.equal(y, x + 1)


def test_batch_rejects_undersized_data():
    data = torch.arange(5)
    with pytest.raises(ValueError):
        get_batch(data, context_window=8, batch_size=2, target_device=torch.device("cpu"))


def test_batch_never_overflows_data_range():
    torch.manual_seed(1)
    data = torch.arange(50)
    for _ in range(64):
        x, y = get_batch(data, context_window=10, batch_size=8, target_device=torch.device("cpu"))
        assert int(x.min()) >= 0
        assert int(y.max()) < len(data)
