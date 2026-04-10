"""Specs for time-reversibility measurement."""
from __future__ import annotations

import pytest
import torch

from yaum.core.diagnostics import ReversibilityResidual, measure_reversibility
from yaum.core.integrators import LeapfrogIntegrator, YoshidaIntegrator


def harmonic_force(k: float):
    def force_fn(x, *, retain_graph):
        x_leaf = x if x.requires_grad else x.detach().requires_grad_(True)
        loss = 0.5 * k * (x_leaf * x_leaf).sum()
        (grad,) = torch.autograd.grad(
            loss, x_leaf, create_graph=retain_graph, retain_graph=retain_graph
        )
        return loss, -grad
    return force_fn


@pytest.mark.parametrize(
    "integrator_cls", [LeapfrogIntegrator, YoshidaIntegrator]
)
def test_harmonic_oscillator_residual_is_machine_precision(integrator_cls):
    torch.manual_seed(0)
    E0 = torch.tensor([[1.0], [-0.4]], dtype=torch.float64)
    P0 = torch.tensor([[0.0], [0.3]], dtype=torch.float64)
    mass = torch.ones(2, 1, dtype=torch.float64)

    residual = measure_reversibility(
        integrator_cls(),
        E0,
        P0,
        mass,
        harmonic_force(k=1.0),
        dt=0.05,
        n_steps=50,
    )
    assert isinstance(residual, ReversibilityResidual)
    assert residual.position < 1e-10
    assert residual.momentum < 1e-10
    assert residual.total < 2e-10


def test_reversibility_rejects_zero_step_count():
    with pytest.raises(ValueError):
        measure_reversibility(
            LeapfrogIntegrator(),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            harmonic_force(1.0),
            dt=0.01,
            n_steps=0,
        )


def test_residual_grows_under_state_perturbation():
    """Perturbing the intermediate state should break reversibility.

    This encodes the diagnostic's value proposition: the residual is not
    zero by construction — it detects information loss. We build a
    deliberately dissipative "force field" that damps momentum and show
    the residual becomes measurable.
    """
    torch.manual_seed(0)
    mass = torch.ones(2, 1, dtype=torch.float64)
    E0 = torch.tensor([[0.5], [-0.2]], dtype=torch.float64)
    P0 = torch.tensor([[0.1], [0.0]], dtype=torch.float64)

    # A time-reversible force gives a tiny residual...
    good = measure_reversibility(
        LeapfrogIntegrator(), E0, P0, mass,
        harmonic_force(1.0), dt=0.01, n_steps=20,
    )

    # ...but a dissipative, time-dependent field does not.
    call_count = {"n": 0}

    def dissipative(x, *, retain_graph):
        call_count["n"] += 1
        x_leaf = x if x.requires_grad else x.detach().requires_grad_(True)
        # Add a tiny step-dependent bias that breaks time symmetry.
        bias = 1e-3 * call_count["n"]
        loss = 0.5 * (x_leaf * x_leaf).sum() + bias * x_leaf.sum()
        (grad,) = torch.autograd.grad(
            loss, x_leaf, create_graph=retain_graph, retain_graph=retain_graph
        )
        return loss, -grad

    bad = measure_reversibility(
        LeapfrogIntegrator(), E0, P0, mass,
        dissipative, dt=0.01, n_steps=20,
    )

    assert bad.total > good.total * 100
