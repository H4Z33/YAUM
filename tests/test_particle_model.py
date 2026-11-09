"""Specs for the many-body (every token a particle) model."""
from __future__ import annotations

import pytest
import torch

from yaum.models import make_model
from yaum.models.particles import HamiltonianParticleModel


def _model(dissipation="none", **kw):
    torch.manual_seed(0)
    return HamiltonianParticleModel(
        vocab_size=7, embedding_dim=5, hidden_dim=4, n_layers=2,
        charge_dim=3, dissipation=dissipation, **kw,
    )


def _universe(model, batch=3, n=6, spread=1.0):
    q = torch.randn(batch, n, model.hidden_dim) * spread
    p = torch.randn(batch, n, model.hidden_dim) * 0.5
    c = torch.randn(batch, n, model.charge_dim)
    return q, p, c


def _no_confinement(model):
    with torch.no_grad():
        model._kappa.fill_(-40.0)  # softplus(-40) ≈ 4e-18


def test_forward_shapes_and_universe_grows():
    model = _model()
    logits, (q, p, c) = model(torch.randn(2, 5, 5), None)
    assert logits.shape == (2, 5, 7)
    assert q.shape == p.shape == (2, 5, 4)
    assert c.shape == (2, 5, 3)
    # Continuing the universe adds particles to the existing ones.
    _, (q2, _, _) = model(torch.randn(2, 3, 5), (q, p, c))
    assert q2.shape[1] == 8


def test_forces_are_exact_negative_gradient_of_energy():
    model = _model()
    q, p, c = _universe(model)
    q = q.requires_grad_(True)
    (grad,) = torch.autograd.grad(model.energy(q, p, c).sum(), q)
    assert torch.allclose(model.forces(q, c), -grad, atol=1e-5)


def test_newtons_third_law_conserves_total_momentum():
    model = _model("none", dt=0.05)
    _no_confinement(model)
    q, p, c = _universe(model)
    total0 = p.sum(1)
    with torch.no_grad():
        q, p = model.flow(q, p, c, 400)
    assert torch.allclose(p.sum(1), total0, atol=1e-4)


def test_frictionless_flow_conserves_energy():
    model = _model("none", dt=0.05)
    q, p, c = _universe(model)
    with torch.no_grad():
        H0 = model.energy(q, p, c)
        worst = 0.0
        for _ in range(20):
            q, p = model.flow(q, p, c, 50)
            worst = max(worst, float(((model.energy(q, p, c) - H0).abs() / H0.abs()).max()))
    assert worst < 1e-2


def test_frictionless_flow_is_time_reversible():
    """Run in float64: many-body dynamics are chaotic and amplify float32
    round-off (~1e-3 after 150+150 steps), which would hide a real bug."""
    model = _model("none", dt=0.1).double()
    q0, p0, c = (t.double() for t in _universe(model))
    with torch.no_grad():
        q, p = model.flow(q0, p0, c, 150)
        q, p = model.flow(q, p, c, 150, dt=-0.1)
    assert torch.allclose(q, q0, atol=1e-8)
    assert torch.allclose(p, p0, atol=1e-8)


def test_like_charges_attract_and_bind_with_friction():
    """Two aligned charges starting apart end up closer and nearly at rest."""
    model = _model(1.0, dt=0.2)
    _no_confinement(model)
    q = torch.tensor([[[-0.8, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]]])
    p = torch.zeros_like(q)
    c = torch.tensor([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    with torch.no_grad():
        q1, p1 = model.flow(q, p, c, 300)
    assert (q1[0, 0] - q1[0, 1]).norm() < 0.5 * (q[0, 0] - q[0, 1]).norm()
    assert p1.abs().max() < 1e-2


def test_opposite_charges_repel():
    model = _model("none", dt=0.1)
    _no_confinement(model)
    q = torch.tensor([[[-0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]])
    c = torch.tensor([[[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
    with torch.no_grad():
        q1, _ = model.flow(q, torch.zeros_like(q), c, 20)
    assert (q1[0, 0] - q1[0, 1]).norm() > (q[0, 0] - q[0, 1]).norm()


def test_interaction_energy_is_bounded_below():
    model = _model("none")
    q, p, c = _universe(model, batch=64, n=8, spread=0.01)  # all crowded together
    with torch.no_grad():
        H = model.energy(q, torch.zeros_like(p), c)
        g = c @ c.transpose(-1, -2)
        bound = -0.5 * (g.clamp(min=0).sum((-1, -2)) - g.diagonal(dim1=-2, dim2=-1).clamp(min=0).sum(-1))
    assert torch.all(H >= bound - 1e-4)


def test_parameters_receive_gradient():
    model = _model("learned")
    logits, _ = model(torch.randn(2, 4, 5), None)
    logits.sum().backward()
    for name in ("_rho", "_sigma", "_kappa", "_mass", "_charge_scale"):
        grad = getattr(model, name).grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert model.to_charge.weight.grad.abs().sum() > 0


def test_charges_and_range_are_bounded():
    model = _model()
    c = model.charges(torch.randn(4, 9, 5) * 100)
    g = c @ c.transpose(-1, -2)
    assert g.abs().max() <= model.charge_scale ** 2 + 1e-5
    with torch.no_grad():
        model._sigma.fill_(-50.0)
    assert float(model.sigma) >= model.sigma_min


def test_registry_and_rejects_negative_dissipation():
    model = make_model(9, {"model_type": "particles", "embedding_dim": 4,
                           "hidden_dim": 6, "n_layers": 1, "particle_charge_dim": 5})
    assert isinstance(model, HamiltonianParticleModel) and model.charge_dim == 5
    with pytest.raises(ValueError):
        _model(-1.0)


# ---- Big Bang options ------------------------------------------------------

def test_defaults_keep_the_measured_model():
    model = _model()
    assert float(model.spawn_scale) == pytest.approx(1.0, rel=1e-5)
    assert float(model.kappa) > 0 and not model.expansion
    q, _, _ = model.init_hidden(2)
    assert q.shape[1] == 0


def test_dense_birth_puts_particles_within_range():
    torch.manual_seed(0)
    kw = {"vocab_size": 7, "embedding_dim": 16, "hidden_dim": 16, "charge_dim": 3}
    x = torch.randn(1, 30, 16)
    sparse = HamiltonianParticleModel(**kw)
    dense = HamiltonianParticleModel(**kw, spawn_scale_init=0.1)
    dense.load_state_dict({**sparse.state_dict(), "_spawn_scale": dense._spawn_scale.data})

    def mean_kernel(m):
        q = m.spawn_scale * m.to_position(x)
        r2 = ((q.unsqueeze(2) - q.unsqueeze(1)) ** 2).sum(-1)
        return float(torch.exp(-r2 / (2 * m.sigma**2)).mean())

    with torch.no_grad():
        assert mean_kernel(dense) > 0.5 > mean_kernel(sparse)


def test_expansion_allows_negative_kappa_and_conserves_energy():
    model = _model("none", dt=0.05, expansion=True, kappa_init=-0.05)
    assert float(model.kappa) == pytest.approx(-0.05, rel=1e-4)
    q, p, c = _universe(model, spread=0.5)
    q = q.requires_grad_(True)
    (grad,) = torch.autograd.grad(model.energy(q, p, c).sum(), q)
    assert torch.allclose(model.forces(q, c), -grad, atol=1e-5)
    with torch.no_grad():
        H0 = model.energy(q, p, c)
        q1, p1 = model.flow(q, p, c, 200)
        assert float(((model.energy(q1, p1, c) - H0).abs() / H0.abs()).max()) < 1e-2
        # A cosmological constant pushes the universe apart.
        assert float(q1.norm(dim=-1).mean()) > float(q.norm(dim=-1).mean())
    with pytest.raises(ValueError):
        _model(expansion=True, kappa_init=0.5)


def test_primordial_particles_are_learned_initial_condition():
    model = _model(n_primordial=3)
    q, _, c = model.init_hidden(2)
    assert q.shape == (2, 3, 4) and c.shape == (2, 3, 3)
    logits, (q1, _, _) = model(torch.randn(2, 4, 5), None)
    assert q1.shape[1] == 7
    logits.sum().backward()
    for name in ("primordial_q", "primordial_c", "_spawn_scale"):
        grad = getattr(model, name).grad
        assert grad is not None and float(grad.abs().sum()) > 0, name
