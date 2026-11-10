"""Specs for the Hamiltonian-in-context-time sequence model.

The model claims three physical properties; each gets a test:

* the force is the exact gradient of the potential,
* with no friction and no input the flow conserves energy and is
  time-reversible,
* with friction the energy decays.
"""
from __future__ import annotations

import os

import pytest
import torch

from yaum.core.trainer import Trainer
from yaum.models import make_model
from yaum.models.hamiltonian import HamiltonianCharModel


def _model(dissipation="none", **kw):
    torch.manual_seed(0)
    return HamiltonianCharModel(
        vocab_size=7, embedding_dim=5, hidden_dim=8, n_layers=2,
        dissipation=dissipation, **kw,
    )


def _random_state(model, batch=4, scale=1.0):
    q = torch.randn(batch, model.hidden_dim) * scale
    p = torch.randn(batch, model.hidden_dim) * scale
    return q, p


def test_forward_shapes_and_hidden():
    model = _model()
    x = torch.randn(3, 6, 5)
    logits, (q, p) = model(x, model.init_hidden(3))
    assert logits.shape == (3, 6, 7)
    assert q.shape == p.shape == (3, 8)


def test_force_is_exact_gradient_of_potential():
    model = _model()
    q = torch.randn(5, 8, requires_grad=True)
    (auto,) = torch.autograd.grad(model.potential(q).sum(), q)
    assert torch.allclose(model.grad_potential(q), auto, atol=1e-5)


def test_conservative_flow_conserves_energy():
    model = _model("none", dt=0.05)
    q, p = _random_state(model)
    with torch.no_grad():
        H0 = model.energy(q, p)
        worst = 0.0
        for _ in range(40):
            q, p = model.flow(q, p, 50)
            worst = max(worst, float(((model.energy(q, p) - H0).abs() / H0.abs()).max()))
    # Leapfrog: bounded O(dt^2) oscillation, no secular drift over 2000 steps.
    assert worst < 1e-2


def test_conservative_flow_is_time_reversible():
    model = _model("none", dt=0.1)
    q0, p0 = _random_state(model)
    with torch.no_grad():
        q, p = model.flow(q0, p0, 200)
        q, p = model.flow(q, p, 200, dt=-0.1)
    assert torch.allclose(q, q0, atol=1e-3)
    assert torch.allclose(p, p0, atol=1e-3)


def test_friction_dissipates_energy():
    model = _model(0.5, dt=0.1)
    q, p = _random_state(model, scale=2.0)
    with torch.no_grad():
        H0 = model.energy(q, p)
        q, p = model.flow(q, p, 400)
        H1 = model.energy(q, p)
        # Energy has a minimum (confining potential); it should approach it.
        q_min, p_min = model.flow(q, p, 2000)
        H_floor = model.energy(q_min, p_min)
    assert torch.all(H1 < H0)
    assert torch.all(H1 - H_floor < 0.25 * (H0 - H_floor))


def test_learned_dissipation_receives_gradient_and_starts_positive():
    model = _model("learned", gamma_init=0.05)
    assert torch.allclose(model.gamma, torch.full((8,), 0.05), atol=1e-6)
    logits, _ = model(torch.randn(2, 4, 5), None)
    logits.sum().backward()
    assert model._rho.grad is not None and model._rho.grad.abs().sum() > 0


def test_rejects_negative_dissipation():
    with pytest.raises(ValueError):
        _model(-0.1)


def test_make_model_registry_and_config_keys():
    model = make_model(
        9,
        {
            "model_type": "hamiltonian",
            "embedding_dim": 4,
            "hidden_dim": 6,
            "n_layers": 3,
            "ham_dt": 0.25,
            "ham_dissipation": "none",
        },
    )
    assert isinstance(model, HamiltonianCharModel)
    assert model.n_substeps == 3 and model.dt == 0.25
    assert torch.all(model.gamma == 0)


def test_trainer_runs_the_hamiltonian_model_end_to_end(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(
        {
            "model_type": "hamiltonian",
            "embedding_dim": 4,
            "hidden_dim": 6,
            "n_layers": 1,
            "learning_rate_W": 1e-3,
            "gradient_clip_norm_W": 1.0,
            "context_window": 6,
            "batch_size": 4,
            "num_steps": 4,
            "eval_interval": 2,
            "eval_iters": 2,
            "results_dir": str(tmp_path),
            "safe_speed": False,
        }
    )
    vocab = 8
    data = torch.arange(200) % vocab
    chars = {chr(97 + i): i for i in range(vocab)}
    trainer.setup(data, data[:120], chars, {i: c for c, i in chars.items()}, vocab, torch.ones(vocab, 1))
    events = list(trainer.train())
    assert events[-1]["status"] == "finished"
    assert len(trainer.generate("ab", 5, 1.0)) == 7
    path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")
    assert Trainer({"results_dir": str(tmp_path)}).load_checkpoint(path) is True


# ---- dynamic geometry ------------------------------------------------------

def _geo_model(dissipation="none", **kw):
    torch.manual_seed(0)
    return HamiltonianCharModel(
        vocab_size=7, embedding_dim=5, hidden_dim=8, n_layers=2,
        dissipation=dissipation, geometry="dynamic", **kw,
    )


def _joint_state(model, batch=4, scale=1.0):
    q, p = _random_state(model, batch, scale)
    phi = 0.1 * torch.randn(batch, model.n_potential)
    pi = 0.1 * torch.randn(batch, model.n_potential)
    return q, p, phi, pi


def test_dynamic_geometry_forward_shapes_and_rest_start():
    model = _geo_model()
    h0 = model.init_hidden(3)
    assert len(h0) == 4
    assert torch.all(h0[2] == 0) and torch.all(h0[3] == 0)  # starts unbent, at rest
    logits, (_q, _p, phi, pi) = model(torch.randn(3, 6, 5), h0)
    assert logits.shape == (3, 6, 7)
    assert phi.shape == pi.shape == (3, model.n_potential)


def test_geometry_force_is_exact_gradient_of_joint_energy():
    model = _geo_model()
    q, p, phi, pi = _joint_state(model)
    phi = phi.requires_grad_(True)
    (auto,) = torch.autograd.grad(model.energy(q, p, phi, pi).sum(), phi)
    assert torch.allclose(model.geometry_force(q, phi), -auto, atol=1e-5)


def test_joint_flow_conserves_joint_energy_without_friction():
    model = _geo_model("none", dt=0.05)
    q, p, phi, pi = _joint_state(model)
    with torch.no_grad():
        H0 = model.energy(q, p, phi, pi)
        worst = 0.0
        for _ in range(20):
            q, p, phi, pi = model.flow(q, p, 50, phi=phi, pi=pi)
            H = model.energy(q, p, phi, pi)
            worst = max(worst, float(((H - H0).abs() / H0.abs()).max()))
    assert worst < 1e-2


def test_joint_flow_is_time_reversible():
    model = _geo_model("none", dt=0.1)
    s0 = _joint_state(model)
    with torch.no_grad():
        s = model.flow(*s0[:2], 100, phi=s0[2], pi=s0[3])
        s = model.flow(*s[:2], 100, dt=-0.1, phi=s[2], pi=s[3])
    for a, b in zip(s, s0):
        assert torch.allclose(a, b, atol=1e-3)


def test_content_bends_the_geometry():
    """Different inputs must leave the landscape in different shapes."""
    model = _geo_model("none")
    with torch.no_grad():
        _, (_, _, phi_a, _) = model(torch.randn(1, 8, 5) * 2, None)
        _, (_, _, phi_b, _) = model(torch.randn(1, 8, 5) * 2, None)
        _, (_, _, phi_0, _) = model(torch.zeros(1, 8, 5), None)
    assert phi_a.abs().max() > 1e-2
    assert (phi_a - phi_b).abs().max() > 1e-2
    # No matter moving (q stays at 0, z = b = 0, tanh² 0 = 0): no bending.
    assert torch.allclose(phi_0, torch.zeros_like(phi_0), atol=1e-6)


def test_joint_energy_is_bounded_below():
    """Regression for the 'gravitational collapse' of the first design."""
    model = _geo_model("none")
    with torch.no_grad():
        q = torch.randn(256, model.hidden_dim) * 20
        phi = -torch.rand(256, model.n_potential) * 50
        H = model.energy(q, torch.zeros_like(q), phi, torch.zeros_like(phi))
        static_floor = model.potential(q).min()
        bound = static_floor - (1.0 / (2.0 * model.stiffness)).sum()
    assert torch.all(H >= bound - 1e-3)


def test_geometry_parameters_receive_gradient():
    model = _geo_model("learned")
    logits, _ = model(torch.randn(2, 5, 5), None)
    logits.sum().backward()
    for name in ("_stiffness", "_geom_mass", "_rho_geom", "a"):
        grad = getattr(model, name).grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_static_geometry_is_unchanged_by_default():
    assert _model().geometry == "static"
    assert len(_model().init_hidden(2)) == 2
    with pytest.raises(ValueError):
        _model(geometry="curved")


# ---- cyclic coordinates (Noether) -----------------------------------------

def test_cyclic_momenta_are_exactly_conserved_and_default_is_unchanged():
    torch.manual_seed(0)
    model = HamiltonianCharModel(7, 5, 8, 2, dissipation="learned", n_cyclic=3)
    q, p = torch.randn(4, 8), torch.randn(4, 8)
    qg = q.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model.potential(qg).sum(), qg)
    assert torch.allclose(model.grad_potential(q), grad, atol=1e-5)
    assert float(grad[:, -3:].abs().max()) == 0.0  # V ignores cyclic coordinates
    with torch.no_grad():
        _, p1 = model.flow(q, p, 500)
    # Translation symmetry → conserved momentum, even with friction elsewhere.
    assert torch.equal(p1[:, -3:], p[:, -3:])
    assert not torch.allclose(p1[:, :-3], p[:, :-3], atol=1e-2)
    logits, _ = model(torch.randn(2, 6, 5), None)
    assert logits.shape == (2, 6, 7)
    plain = HamiltonianCharModel(7, 5, 8, 2)
    assert plain.n_cyclic == 0 and bool((plain.kappa > 0).all())
    with pytest.raises(ValueError):
        HamiltonianCharModel(7, 5, 8, 2, n_cyclic=8)


def test_zero_charge_tokens_cannot_change_the_conserved_momenta():
    torch.manual_seed(0)
    model = HamiltonianCharModel(7, 5, 8, 2, dissipation="learned", n_cyclic=3, cyclic_charge=True)
    with torch.no_grad():
        model.cyclic_charge.weight.zero_()
        model.cyclic_charge.bias.fill_(-50.0)  # charge ≈ 0 for every token
        _, (_, p) = model(torch.randn(2, 30, 5), None)
        assert float(p[:, -3:].abs().max()) < 1e-12
        assert float(p[:, :-3].abs().max()) > 1e-3  # ordinary modes still driven
        model.cyclic_charge.bias.fill_(50.0)  # charge ≈ 1: behaves like no gate
        _, (_, p_on) = model(torch.randn(2, 30, 5), None)
        assert float(p_on[:, -3:].abs().max()) > 1e-3
    with pytest.raises(ValueError):
        HamiltonianCharModel(7, 5, 8, 2, cyclic_charge=True)


def test_neutral_tokens_have_exactly_zero_charge():
    torch.manual_seed(0)
    model = HamiltonianCharModel(7, 5, 8, 2, n_cyclic=3, cyclic_charge="neutral")
    x = torch.randn(2, 30, 5)
    with torch.no_grad():
        model(x, None)
        assert float(model.last_charge_mean) > 0  # starts charged
        model.cyclic_charge.weight.zero_()
        model.cyclic_charge.bias.fill_(-0.1)  # slightly negative is enough: relu gives exactly 0
        _, (_, p) = model(x, None)
        assert float(p[:, -3:].abs().max()) == 0.0
        assert float(model.last_charge_mean) == 0.0
    with pytest.raises(ValueError):
        HamiltonianCharModel(7, 5, 8, 2, n_cyclic=3, cyclic_charge="hard")
