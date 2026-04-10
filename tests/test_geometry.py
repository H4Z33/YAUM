"""Specs for the Fisher-information mass estimator."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from yaum.core.geometry import (
    FisherMassConfig,
    FisherMassEstimator,
    fisher_diagonal_sample,
)
from yaum.models.rnn import RNNCharModel


# ---- config validation ---------------------------------------------------

def test_config_rejects_bad_beta():
    with pytest.raises(ValueError):
        FisherMassConfig(beta=1.0)
    with pytest.raises(ValueError):
        FisherMassConfig(beta=-0.1)


def test_config_rejects_non_positive_eps():
    with pytest.raises(ValueError):
        FisherMassConfig(eps=0.0)


def test_config_rejects_non_positive_refresh():
    with pytest.raises(ValueError):
        FisherMassConfig(refresh_every=0)


# ---- estimator bookkeeping ----------------------------------------------

def test_estimator_is_not_usable_before_initialisation():
    est = FisherMassEstimator(shape=(4, 3), config=FisherMassConfig())
    with pytest.raises(RuntimeError):
        est.current()


def test_estimator_initialises_to_eps_floor():
    cfg = FisherMassConfig(eps=0.5)
    est = FisherMassEstimator(shape=(5, 2), config=cfg)
    est.initialise(device="cpu", dtype=torch.float32)
    m = est.current()
    assert m.shape == (5, 2)
    assert torch.all(m == 0.5)


def test_estimator_update_blends_sample_via_beta():
    cfg = FisherMassConfig(beta=0.5, eps=1e-6)
    est = FisherMassEstimator(shape=(2, 2), config=cfg)
    est.initialise("cpu", torch.float32)
    prior = est.current().clone()
    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new = est.update(sample)
    expected = 0.5 * prior + 0.5 * sample
    assert torch.allclose(new, expected)
    assert est.update_count == 1


def test_estimator_enforces_floor_after_update():
    cfg = FisherMassConfig(beta=0.0, eps=0.25)
    est = FisherMassEstimator(shape=(2,), config=cfg)
    est.initialise("cpu", torch.float32)
    est.update(torch.tensor([0.1, 2.0]))
    m = est.current()
    assert m[0] == 0.25  # floored
    assert m[1] == 2.0   # left alone


def test_estimator_rejects_sample_shape_mismatch():
    est = FisherMassEstimator(shape=(3, 2), config=FisherMassConfig())
    est.initialise("cpu", torch.float32)
    with pytest.raises(ValueError):
        est.update(torch.zeros(3, 3))


def test_estimator_stores_on_sample_device_when_first_initialised_via_update():
    cfg = FisherMassConfig(beta=0.0, eps=1e-8)
    est = FisherMassEstimator(shape=(2, 2), config=cfg)
    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    est.update(sample)  # implicit initialise
    assert est.current().device == sample.device


# ---- diagonal Fisher sample from an RNN --------------------------------

def test_fisher_sample_has_embedding_shape():
    torch.manual_seed(0)
    vocab = 6
    d = 4
    model = RNNCharModel(vocab, d, 8, 1)
    E = torch.randn(vocab, d)
    x = torch.randint(0, vocab, (2, 5))
    y = torch.randint(0, vocab, (2, 5))
    sample = fisher_diagonal_sample(model, nn.CrossEntropyLoss(), E, x, y)
    assert sample.shape == (vocab, d)
    assert torch.all(sample >= 0.0)


def test_fisher_sample_zero_for_unseen_tokens():
    torch.manual_seed(0)
    vocab = 6
    d = 4
    model = RNNCharModel(vocab, d, 8, 1)
    E = torch.randn(vocab, d)
    # Use only tokens 0..2 in the batch — rows 3..5 should stay at zero grad.
    x = torch.randint(0, 3, (2, 5))
    y = torch.randint(0, 3, (2, 5))
    sample = fisher_diagonal_sample(model, nn.CrossEntropyLoss(), E, x, y)
    assert torch.all(sample[3:] == 0.0)
    assert torch.any(sample[:3] > 0.0)
