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
