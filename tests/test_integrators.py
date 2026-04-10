"""Specification tests for symplectic integrators.

The harmonic oscillator is the universal testbed: its Hamiltonian
H = p^2 / 2m + k x^2 / 2 is quadratic, exactly solvable, and any
symplectic integrator should conserve H up to bounded oscillation.
"""
from __future__ import annotations

import math

import pytest
import torch

from yaum.core.integrators import (
    INTEGRATORS,
    LangevinIntegrator,
    LeapfrogIntegrator,
    MultiScaleIntegrator,
    YoshidaIntegrator,
    make_integrator,
)


def harmonic_force(k: float):
    """Force for V = k x^2 / 2. Returns (loss, grad) shaped like x."""

    def force_fn(x, *, retain_graph):
        x_leaf = x if x.requires_grad else x.detach().requires_grad_(True)
        loss = 0.5 * k * (x_leaf * x_leaf).sum()
        (grad,) = torch.autograd.grad(
            loss, x_leaf, create_graph=retain_graph, retain_graph=retain_graph
        )
        return loss, -grad

    return force_fn


def integrate(integrator, x0, p0, mass, force_fn, dt, n_steps):
    x = x0.clone().requires_grad_(True)
    p = p0.clone()
    for _ in range(n_steps):
        step = integrator.step(x, p, dt, force_fn, mass, retain_final=False)
        x = step.E.requires_grad_(True)
        p = step.P
    return x.detach(), p


def hamiltonian(x, p, mass, k):
    return (0.5 * (p * p / mass).sum() + 0.5 * k * (x * x).sum()).item()


@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_registry_roundtrip(name):
    integrator = make_integrator(name)
    assert integrator.name == name
    assert integrator.order >= 2


def test_unknown_integrator_raises():
    with pytest.raises(ValueError):
        make_integrator("does-not-exist")


@pytest.mark.parametrize(
    "integrator_cls, max_drift",
    [(LeapfrogIntegrator, 5e-3), (YoshidaIntegrator, 1e-6)],
)
def test_energy_conserved_on_harmonic_oscillator(integrator_cls, max_drift):
    torch.manual_seed(0)
    k = 1.0
    mass = torch.ones(1, 1)
    x0 = torch.tensor([[1.0]])
    p0 = torch.tensor([[0.0]])
    dt = 0.05
    n_steps = 200

    force_fn = harmonic_force(k)
    H0 = hamiltonian(x0, p0, mass, k)
    x, p = integrate(integrator_cls(), x0, p0, mass, force_fn, dt, n_steps)
    H1 = hamiltonian(x, p, mass, k)

    assert abs(H1 - H0) / abs(H0) < max_drift


def test_leapfrog_is_time_reversible():
    """Run forward then backward: must return (approximately) to the start."""
    torch.manual_seed(0)
    mass = torch.ones(3, 2)
    x0 = torch.randn(3, 2)
    p0 = torch.randn(3, 2)
    dt = 0.03
    n_steps = 50

    force_fn = harmonic_force(k=0.8)
    leap = LeapfrogIntegrator()
    x, p = integrate(leap, x0, p0, mass, force_fn, dt, n_steps)
    x_back, p_back = integrate(leap, x, -p, mass, force_fn, dt, n_steps)

    assert torch.allclose(x_back, x0, atol=1e-5)
    assert torch.allclose(-p_back, p0, atol=1e-5)


def test_yoshida_is_higher_order_than_leapfrog():
    """Global error of Yoshida should decay faster than leapfrog as dt -> 0."""
    torch.manual_seed(0)
    k = 1.0
    mass = torch.ones(1, 1)
    x0 = torch.tensor([[1.0]])
    p0 = torch.tensor([[0.0]])
    T = 2.0

    force_fn = harmonic_force(k)

    def exact(t):
        return torch.tensor([[math.cos(t)]]), torch.tensor([[-math.sin(t)]])

    def err(integrator, dt):
        n = int(round(T / dt))
        x, p = integrate(integrator, x0, p0, mass, force_fn, dt, n)
        xr, pr = exact(n * dt)
        return ((x - xr).abs().max() + (p - pr).abs().max()).item()

    leap = LeapfrogIntegrator()
    yosh = YoshidaIntegrator()

    e_leap = err(leap, 0.05)
    e_yosh = err(yosh, 0.05)

    assert e_yosh < e_leap * 1e-2


def test_langevin_zero_friction_matches_core():
    torch.manual_seed(0)
    mass = torch.ones(1, 1)
    x = torch.tensor([[1.0]]).requires_grad_(True)
    p = torch.tensor([[0.0]])
    force_fn = harmonic_force(1.0)

    leap = LeapfrogIntegrator().step(x, p, 0.02, force_fn, mass, retain_final=False)
    thermo = LangevinIntegrator(friction=0.0, temperature=5.0).step(
        x, p, 0.02, force_fn, mass, retain_final=False
    )
    assert torch.allclose(thermo.E, leap.E)
    assert torch.allclose(thermo.P, leap.P)


def test_langevin_zero_temperature_damps_momentum():
    torch.manual_seed(0)
    mass = torch.ones(1, 1)
    x = torch.tensor([[0.0]]).requires_grad_(True)
    p = torch.tensor([[2.0]])
    integrator = LangevinIntegrator(friction=1.0, temperature=0.0)
    force_fn = harmonic_force(1.0)

    for _ in range(500):
        step = integrator.step(x, p, 0.05, force_fn, mass, retain_final=False)
        x = step.E.requires_grad_(True)
        p = step.P
    assert abs(float(p)) < 1e-3


def test_langevin_equipartition_reaches_target_temperature():
    """Mean kinetic energy should approach kT / 2 per degree of freedom."""
    torch.manual_seed(42)
    kT = 2.5
    mass = torch.ones(1, 1)
    integrator = LangevinIntegrator(friction=1.0, temperature=kT, core="leapfrog")
    force_fn = harmonic_force(1.0)

    x = torch.tensor([[0.0]]).requires_grad_(True)
    p = torch.tensor([[0.0]])

    p_sq_samples = []
    dt = 0.05
    burn_in = 2000
    collect = 8000
    for k in range(burn_in + collect):
        step = integrator.step(x, p, dt, force_fn, mass, retain_final=False)
        x = step.E.requires_grad_(True)
        p = step.P
        if k >= burn_in:
            p_sq_samples.append(float(p) ** 2)

    mean_p_sq = sum(p_sq_samples) / len(p_sq_samples)
    # <p^2 / m> = kT for a single d.o.f. with unit mass.
    assert abs(mean_p_sq - kT) / kT < 0.1


def test_make_integrator_forwards_kwargs_to_langevin():
    integrator = make_integrator(
        "langevin",
        friction=0.7,
        temperature=1.3,
        core="yoshida4",
    )
    assert isinstance(integrator, LangevinIntegrator)
    assert integrator.gamma == 0.7
    assert integrator.kT == 1.3
    assert integrator.core.name == "yoshida4"


def test_make_integrator_drops_unknown_kwargs():
    integrator = make_integrator("leapfrog", friction=99.0, something_random=123)
    assert isinstance(integrator, LeapfrogIntegrator)


def test_langevin_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        LangevinIntegrator(friction=-1.0)
    with pytest.raises(ValueError):
        LangevinIntegrator(temperature=-0.1)
    with pytest.raises(ValueError):
        LangevinIntegrator(core="nope")


def test_multiscale_with_zero_slow_matches_inner_at_subdt():
    """With slow_lambda=0 a multiscale outer step of dt with n_inner=4
    must equal running the inner integrator 4 times at dt/4."""
    torch.manual_seed(0)
    mass = torch.ones(1, 1)
    force_fn = harmonic_force(1.0)

    # Reference: raw leapfrog 4x at dt/4.
    x_ref = torch.tensor([[1.0]]).requires_grad_(True)
    p_ref = torch.tensor([[0.0]])
    leap = LeapfrogIntegrator()
    for _ in range(4):
        s = leap.step(x_ref, p_ref, 0.01, force_fn, mass, retain_final=False)
        x_ref = s.E.requires_grad_(True)
        p_ref = s.P

    # Multi-scale: one outer step of dt=0.04 with n_inner=4.
    ms = MultiScaleIntegrator(inner="leapfrog", n_inner=4, slow_lambda=0.0)
    x = torch.tensor([[1.0]]).requires_grad_(True)
    p = torch.tensor([[0.0]])
    s = ms.step(x, p, 0.04, force_fn, mass, retain_final=False)

    assert torch.allclose(s.E, x_ref.detach(), atol=1e-10)
    assert torch.allclose(s.P, p_ref, atol=1e-10)


def test_multiscale_conserves_total_hamiltonian_with_slow_term():
    """V = V_fast + V_slow = 0.5(k_fast + slow_lambda) x^2; H must be bounded."""
    torch.manual_seed(0)
    mass = torch.ones(1, 1)
    k_fast = 1.0
    slow_lambda = 0.3
    force_fn = harmonic_force(k_fast)

    ms = MultiScaleIntegrator(inner="leapfrog", n_inner=4, slow_lambda=slow_lambda)

    def H(x, p):
        return 0.5 * float((p * p).detach()) + 0.5 * (k_fast + slow_lambda) * float(
            (x * x).detach()
        )

    x = torch.tensor([[1.0]]).requires_grad_(True)
    p = torch.tensor([[0.0]])
    H0 = H(x, p)
    for _ in range(300):
        s = ms.step(x, p, 0.02, force_fn, mass, retain_final=False)
        x = s.E.requires_grad_(True)
        p = s.P
    H1 = H(x, p)
    assert abs(H1 - H0) / abs(H0) < 5e-3


def test_multiscale_force_eval_count_scales_with_n_inner():
    mass = torch.ones(1, 1)
    x = torch.tensor([[0.5]]).requires_grad_(True)
    p = torch.tensor([[0.1]])
    force_fn = harmonic_force(1.0)

    ms = MultiScaleIntegrator(inner="leapfrog", n_inner=5, slow_lambda=0.0)
    s = ms.step(x, p, 0.05, force_fn, mass, retain_final=False)
    # leapfrog does 2 force evals per sub-step
    assert s.n_force_evals == 5 * 2


def test_multiscale_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        MultiScaleIntegrator(n_inner=0)
    with pytest.raises(ValueError):
        MultiScaleIntegrator(slow_lambda=-1.0)
    with pytest.raises(ValueError):
        MultiScaleIntegrator(inner="nope")


def test_make_integrator_forwards_kwargs_to_multiscale():
    integrator = make_integrator(
        "multiscale",
        inner="yoshida4",
        n_inner=3,
        slow_lambda=0.05,
    )
    assert isinstance(integrator, MultiScaleIntegrator)
    assert integrator.inner.name == "yoshida4"
    assert integrator.n_inner == 3
    assert integrator.slow_lambda == 0.05


def test_phase_step_carries_loss_and_eval_count():
    torch.manual_seed(0)
    mass = torch.ones(2, 1)
    x = torch.tensor([[0.5], [-0.3]]).requires_grad_(True)
    p = torch.zeros(2, 1)
    force_fn = harmonic_force(k=1.0)

    leap_step = LeapfrogIntegrator().step(x, p, 0.01, force_fn, mass)
    assert leap_step.n_force_evals == 2
    assert torch.isfinite(leap_step.loss_initial)
    assert torch.isfinite(leap_step.loss_final)

    yosh_step = YoshidaIntegrator().step(x, p, 0.01, force_fn, mass)
    assert yosh_step.n_force_evals == 6
