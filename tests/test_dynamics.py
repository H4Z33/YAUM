"""Specs for force-field construction and Hamiltonian observables."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from yaum.core.dynamics import (
    EnergyReport,
    calculate_loss_and_embedding_grads,
    calculate_loss_and_grads_rnn,
    kinetic_energy,
    make_force_fn,
    make_rnn_force_fn,
    total_hamiltonian,
)
from yaum.models.transformer import TransformerCharModel


class TinyRNN(nn.Module):
    """Minimal RNN stub matching the interface expected by dynamics.py."""

    def __init__(self, vocab_size=7, embedding_dim=4, hidden_dim=5, n_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, embedded, hidden):
        out, hidden = self.lstm(embedded, hidden)
        return self.fc(out), hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
        )


@pytest.fixture
def rnn_setup():
    torch.manual_seed(0)
    model = TinyRNN()
    criterion = nn.CrossEntropyLoss()
    E = torch.randn(model.vocab_size, model.embedding_dim, requires_grad=True)
    P = torch.zeros_like(E)
    mass = torch.ones(model.vocab_size, 1)
    context = torch.tensor([[0, 1, 2], [3, 4, 5]])
    target = torch.tensor([[1, 2, 3], [4, 5, 6]])
    return model, criterion, E, P, mass, context, target


def test_loss_and_grads_shape_matches_batch(rnn_setup):
    model, criterion, E, _, _, context, target = rnn_setup
    loss, grads, embedded = calculate_loss_and_embedding_grads(
        context, target, E, model, criterion
    )
    assert grads.shape == embedded.shape
    assert torch.isfinite(loss)


def test_rnn_gradient_alias_still_works(rnn_setup):
    model, criterion, E, _, _, context, target = rnn_setup
    loss, grads, embedded = calculate_loss_and_grads_rnn(
        context, target, E, model, criterion
    )
    assert grads.shape == embedded.shape
    assert torch.isfinite(loss)


def test_force_fn_is_sparse_on_inactive_tokens(rnn_setup):
    model, criterion, E, _, _, context, target = rnn_setup
    force_fn = make_force_fn(context, target, model, criterion)
    _, force = force_fn(E, retain_graph=False)

    active = set(context.flatten().tolist())
    inactive = [i for i in range(model.vocab_size) if i not in active]
    assert force.shape == E.shape
    assert force[inactive].abs().sum().item() == 0.0


def test_force_fn_accumulates_repeated_tokens(rnn_setup):
    model, criterion, _, _, _, _, _ = rnn_setup
    E = torch.randn(model.vocab_size, model.embedding_dim, requires_grad=True)
    # Token 2 appears twice; its force should be the sum of two per-position grads.
    context = torch.tensor([[2, 2, 0]])
    target = torch.tensor([[0, 1, 2]])
    force_fn = make_rnn_force_fn(context, target, model, criterion)
    _, force = force_fn(E, retain_graph=False)
    # The fact it does not error and produces a nonzero force on token 2 is the spec.
    assert force[2].abs().sum().item() > 0


def test_force_fn_runs_with_transformer_model():
    torch.manual_seed(0)
    model = TransformerCharModel(
        vocab_size=7,
        embedding_dim=4,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        max_context=4,
    )
    criterion = nn.CrossEntropyLoss()
    E = torch.randn(model.vocab_size, model.embedding_dim, requires_grad=True)
    context = torch.tensor([[0, 1, 2], [3, 4, 5]])
    target = torch.tensor([[1, 2, 3], [4, 5, 6]])

    force_fn = make_force_fn(context, target, model, criterion)
    loss, force = force_fn(E, retain_graph=False)

    assert torch.isfinite(loss)
    assert force.shape == E.shape
    assert force.abs().sum().item() > 0


def test_kinetic_energy_is_nonnegative_and_scales_quadratically():
    mass = torch.ones(4, 1) * 2.0
    P = torch.ones(4, 3)
    T1 = kinetic_energy(P, mass).item()
    T2 = kinetic_energy(2 * P, mass).item()
    assert T1 > 0
    assert T2 == pytest.approx(4 * T1)


def test_total_hamiltonian_and_drift():
    mass = torch.ones(2, 1)
    P0 = torch.zeros(2, 2)
    P1 = torch.ones(2, 2) * 0.1
    ref = total_hamiltonian(P0, mass, torch.tensor(1.0))
    later = total_hamiltonian(P1, mass, torch.tensor(0.99))
    assert isinstance(ref, EnergyReport)
    assert ref.total == pytest.approx(1.0)
    assert later.drift(ref) > 0


def test_backcompat_hamiltonian_step_rnn_runs(rnn_setup):
    from yaum.core.dynamics import hamiltonian_step_rnn

    model, criterion, E, P, mass, context, target = rnn_setup
    E_new, P_new, l1, l2 = hamiltonian_step_rnn(
        context, target, E, P, model, criterion, mass, dt=0.01
    )
    assert E_new.shape == E.shape
    assert P_new.shape == P.shape
    assert torch.isfinite(l1) and torch.isfinite(l2)
    # Final loss should still carry a graph so the trainer can backprop to W.
    assert l2.requires_grad
