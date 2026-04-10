"""Trainer integration spec.

These run a tiny model end-to-end on synthetic data. They don't prove the
physics — that's what test_integrators and test_dynamics do — but they
verify the plumbing: setup, training yields, evaluation, checkpoint
round-trip, generation, and integrator selection.
"""
from __future__ import annotations

import os

import pytest
import torch

from yaum.core.trainer import Trainer


def _tiny_config(tmp_path, integrator="leapfrog", num_steps=4):
    return {
        "embedding_dim": 4,
        "hidden_dim": 6,
        "n_layers": 1,
        "dt": 0.01,
        "learning_rate_W": 1e-3,
        "gradient_clip_norm_W": 1.0,
        "context_window": 6,
        "batch_size": 4,
        "num_steps": num_steps,
        "eval_interval": 2,
        "eval_iters": 3,
        "eval_batch_size": 2,
        "results_dir": str(tmp_path),
        "integrator": integrator,
    }


def _tiny_setup(trainer, vocab_size=8):
    # Synthetic data: deterministic ramps so every test has enough tokens.
    train_data = torch.arange(200, dtype=torch.long) % vocab_size
    test_data = torch.arange(120, dtype=torch.long) % vocab_size
    char_to_idx = {chr(ord("a") + i): i for i in range(vocab_size)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    mass = torch.ones(vocab_size, 1)
    trainer.setup(train_data, test_data, char_to_idx, idx_to_char, vocab_size, mass)


def test_trainer_setup_instantiates_state(tmp_path):
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    assert trainer.model is not None
    assert trainer.E.shape == (8, 4)
    assert trainer.P.shape == (8, 4)
    assert trainer.integrator.name == "leapfrog"


def test_train_loop_yields_running_and_finished(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path, num_steps=4))
    _tiny_setup(trainer)
    events = list(trainer.train())
    statuses = [e["status"] for e in events]
    assert "running" in statuses
    assert statuses[-1] == "finished"
    # History must track H_drift alongside the classic stats.
    hist = events[-1]["history"]
    assert "H_drift" in hist
    assert len(hist["H_drift"]) == len(hist["test_l"])


def test_evaluate_returns_finite_loss(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    result = trainer.evaluate()
    assert torch.isfinite(torch.tensor(result["test_loss"]))


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    list(trainer.train())
    ckpt_path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")
    assert os.path.exists(ckpt_path)

    reloaded = Trainer(_tiny_config(tmp_path))
    assert reloaded.load_checkpoint(ckpt_path)
    assert reloaded.current_step == trainer.current_step
    assert torch.allclose(reloaded.E.detach(), trainer.E.detach())
    assert torch.allclose(reloaded.P, trainer.P)


def test_generate_produces_requested_length(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    out = trainer.generate(start_prompt="ab", length=10, temperature=1.0)
    assert len(out) == len("ab") + 10


def test_generate_preserves_model_mode(tmp_path):
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    trainer.model.eval()
    trainer.generate("a", length=3, temperature=1.0)
    assert trainer.model.training is False
    trainer.model.train()
    trainer.generate("a", length=3, temperature=1.0)
    assert trainer.model.training is True


@pytest.mark.parametrize("name", ["leapfrog", "yoshida4"])
def test_trainer_accepts_configured_integrator(tmp_path, name):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path, integrator=name, num_steps=2))
    _tiny_setup(trainer)
    events = list(trainer.train())
    assert events[-1]["status"] == "finished"
    assert trainer.integrator.name == name
