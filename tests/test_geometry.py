"""Specs for the Fisher-information mass estimator."""
from __future__ import annotations

import pytest
import torch
from torch import nn

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


def test_estimator_mass_is_rms_of_force():
    cfg = FisherMassConfig(beta=0.5, eps=1e-8)
    est = FisherMassEstimator(shape=(2, 2), config=cfg)
    est.initialise("cpu", torch.float32)
    first = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
    m1 = est.update(first)
    # First observation initialises v directly (no blend with zero).
    assert torch.allclose(m1, first.sqrt(), atol=1e-6)
    second = torch.tensor([[3.0, 4.0], [9.0, 0.5]])
    m2 = est.update(second)
    assert torch.allclose(m2, (0.5 * first + 0.5 * second).sqrt(), atol=1e-6)
    assert est.update_count == 2


def test_estimator_is_scale_invariant_in_acceleration():
    """F / m should not depend on the raw gradient scale."""
    cfg = FisherMassConfig(beta=0.0, eps=1e-12)
    g = torch.tensor([[0.3, -0.7], [1.2, 0.05]])
    accel = []
    for scale in (1e-4, 1.0):
        est = FisherMassEstimator(shape=(2, 2), config=cfg)
        est.initialise("cpu", torch.float32)
        m = est.update((g * scale) ** 2)
        accel.append((g * scale) / m)
    assert torch.allclose(accel[0], accel[1], atol=1e-5)


def test_estimator_leaves_unseen_rows_untouched():
    cfg = FisherMassConfig(beta=0.5, eps=1e-8)
    est = FisherMassEstimator(shape=(2, 2), config=cfg)
    est.initialise("cpu", torch.float32)
    est.update(torch.tensor([[4.0, 4.0], [4.0, 4.0]]))
    before = est.current().clone()
    m = est.update(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert torch.allclose(m[1], before[1])  # row 1 not in batch: mass frozen
    assert torch.all(m[0] < before[0])


def test_estimator_load_mass_round_trips():
    cfg = FisherMassConfig(beta=0.9, eps=1e-6)
    est = FisherMassEstimator(shape=(2, 3), config=cfg)
    est.initialise("cpu", torch.float32)
    est.update(torch.rand(2, 3) + 0.1)
    saved = est.current().clone()
    fresh = FisherMassEstimator(shape=(2, 3), config=cfg)
    fresh.load_mass(saved)
    assert torch.allclose(fresh.current(), saved, atol=1e-6)
    assert fresh.update_count >= 1


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
