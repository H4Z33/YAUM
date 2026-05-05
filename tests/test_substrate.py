"""Specs for the explicit Hamiltonian substrate wrapper."""
from __future__ import annotations

import torch

from yaum.core.integrators import LeapfrogIntegrator
from yaum.core.substrate import HamiltonianSubstrate


def harmonic_force(k: float):
    def force_fn(x, *, retain_graph):
        x_leaf = x if x.requires_grad else x.detach().requires_grad_(True)
        loss = 0.5 * k * (x_leaf * x_leaf).sum()
        (grad,) = torch.autograd.grad(
            loss, x_leaf, create_graph=retain_graph, retain_graph=retain_graph
        )
        return loss, -grad

    return force_fn


def test_random_substrate_initializes_phase_state():
    torch.manual_seed(0)
    substrate = HamiltonianSubstrate.random(
        vocab_size=5,
        embedding_dim=3,
        mass=torch.ones(5, 1),
        integrator=LeapfrogIntegrator(),
        dt=0.01,
        device=torch.device("cpu"),
    )
    assert substrate.E.shape == (5, 3)
    assert substrate.P.shape == (5, 3)
    assert substrate.E.requires_grad


def test_substrate_propose_is_non_mutating_until_commit():
    substrate = HamiltonianSubstrate(
        E=torch.tensor([[1.0]], requires_grad=True),
        P=torch.tensor([[0.0]]),
        mass=torch.ones(1, 1),
        integrator=LeapfrogIntegrator(),
        dt=0.05,
    )
    old_E = substrate.E.detach().clone()
    old_P = substrate.P.detach().clone()

    step = substrate.propose(harmonic_force(1.0), retain_final=False)

    assert torch.allclose(substrate.E.detach(), old_E)
    assert torch.allclose(substrate.P.detach(), old_P)
    substrate.commit(step)
    assert not torch.allclose(substrate.E.detach(), old_E)
    assert substrate.E.requires_grad
