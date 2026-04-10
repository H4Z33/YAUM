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


def test_trainer_runs_with_multiscale_integrator(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, integrator="multiscale", num_steps=2)
    config["integrator_params"] = {
        "inner": "leapfrog",
        "n_inner": 3,
        "slow_lambda": 1e-3,
    }
    trainer = Trainer(config)
    _tiny_setup(trainer)
    events = list(trainer.train())
    assert events[-1]["status"] == "finished"
    assert trainer.integrator.name == "multiscale"
    assert trainer.integrator.n_inner == 3


def test_trainer_trains_transformer_end_to_end(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, num_steps=2)
    config["model_type"] = "transformer"
    config["embedding_dim"] = 8  # must be divisible by n_heads
    config["n_heads"] = 2
    config["max_context"] = 8
    config["context_window"] = 6
    trainer = Trainer(config)
    _tiny_setup(trainer)
    events = list(trainer.train())
    assert events[-1]["status"] == "finished"
    # Generate must also work for the transformer path.
    out = trainer.generate(start_prompt="ab", length=5, temperature=1.0)
    assert len(out) == 2 + 5


def test_trainer_runs_with_langevin_thermostat(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, integrator="langevin", num_steps=2)
    config["integrator_params"] = {
        "friction": 0.5,
        "temperature": 0.1,
        "core": "leapfrog",
    }
    trainer = Trainer(config)
    _tiny_setup(trainer)
    events = list(trainer.train())
    assert events[-1]["status"] == "finished"
    assert trainer.integrator.name == "langevin"


def test_adaptive_dt_off_by_default(tmp_path):
    trainer = Trainer(_tiny_config(tmp_path))
    assert trainer.adaptive is None


def test_reversibility_probe_records_residuals(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, num_steps=4)
    config["reversibility_check_interval"] = 1
    config["reversibility_check_steps"] = 3
    trainer = Trainer(config)
    _tiny_setup(trainer)
    list(trainer.train())
    history = trainer.get_history()
    assert "reversibility" in history
    recorded = history["reversibility"]
    assert len(recorded) == len(history["test_l"])
    # All eval ticks should have produced finite numbers, not NaN sentinels.
    assert all(not (r != r) for r in recorded)


def test_reversibility_disabled_by_default_records_nan(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path, num_steps=4))
    _tiny_setup(trainer)
    list(trainer.train())
    recorded = trainer.get_history()["reversibility"]
    # Disabled => sentinel NaN on every eval tick.
    assert len(recorded) > 0
    assert all(r != r for r in recorded)


def test_phase_transition_observables_populate_history(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, num_steps=6)
    config["observable_window"] = 3
    trainer = Trainer(config)
    _tiny_setup(trainer)
    list(trainer.train())
    history = trainer.get_history()
    n = len(history["test_l"])
    for key in ("specific_heat", "susceptibility", "corr_time", "entropy_rate"):
        assert key in history
        assert len(history[key]) == n
    # At least one eval tick should have produced a finite number once the
    # rolling window filled up.
    assert any(v == v for v in history["specific_heat"])  # not all NaN
    assert any(v == v for v in history["susceptibility"])


def test_trainer_runs_with_fisher_mass(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, num_steps=4)
    config["mass_mode"] = "fisher"
    config["fisher_refresh_every"] = 2
    config["fisher_batches_per_refresh"] = 2
    config["fisher_eps"] = 1e-12  # keep the floor out of the way for the test
    trainer = Trainer(config)
    _tiny_setup(trainer)
    events = list(trainer.train())
    assert events[-1]["status"] == "finished"
    assert trainer._fisher is not None
    assert trainer._fisher.update_count >= 1
    # Fisher mass has the full embedding shape, not the (V, 1) scalar default.
    assert trainer.mass_vector.shape == (8, 4)
    # At least one entry should have moved away from the pure-eps floor.
    assert (trainer.mass_vector > 1e-12).any()


def test_fisher_mass_survives_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, num_steps=4)
    config["mass_mode"] = "fisher"
    config["fisher_refresh_every"] = 2
    config["fisher_batches_per_refresh"] = 2
    trainer = Trainer(config)
    _tiny_setup(trainer)
    list(trainer.train())
    ckpt_path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")

    reloaded = Trainer(_tiny_config(tmp_path))
    assert reloaded.load_checkpoint(ckpt_path)
    assert reloaded._fisher is not None
    assert torch.allclose(reloaded.mass_vector, trainer.mass_vector)
    assert reloaded._fisher.current().shape == (8, 4)


def test_adaptive_dt_controller_wires_in_and_records_dt(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config(tmp_path, num_steps=4)
    config.update(
        adaptive_dt=True,
        dt_min=1e-5,
        dt_max=1.0,
        drift_high=1e-6,   # force shrinks
        drift_low=-1.0,    # never grow
        dt_shrink=0.5,
        dt_grow=2.0,
        dt_grow_after=100,
    )
    trainer = Trainer(config)
    _tiny_setup(trainer)
    start_dt = trainer._current_dt
    list(trainer.train())
    assert trainer.adaptive is not None
    assert trainer.adaptive.stats["shrinks"] >= 1
    assert trainer._current_dt < start_dt
    history = trainer.get_history()
    assert "dt" in history
    assert len(history["dt"]) == len(history["test_l"])
