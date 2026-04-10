"""Specs for the checkpoint integrity fingerprint."""
from __future__ import annotations

import os

import pytest
import torch

from yaum.core.integrity import (
    CheckpointIntegrity,
    compute_integrity,
    verify_integrity,
)
from yaum.core.trainer import Trainer


def test_fingerprint_is_bit_exact_on_roundtrip():
    torch.manual_seed(0)
    E = torch.randn(8, 4)
    P = torch.randn(8, 4)
    mass = torch.ones(8, 1)
    a = compute_integrity(E, P, mass)
    b = compute_integrity(E, P, mass)
    report = verify_integrity(a, b)
    assert report.ok
    assert report.mismatches == {}


def test_fingerprint_detects_perturbed_momentum():
    torch.manual_seed(1)
    E = torch.randn(8, 4)
    P = torch.randn(8, 4)
    mass = torch.ones(8, 1)
    saved = compute_integrity(E, P, mass)
    P_tampered = P.clone()
    P_tampered[0, 0] += 0.1
    loaded = compute_integrity(E, P_tampered, mass)
    report = verify_integrity(saved, loaded)
    assert not report.ok
    # Both kinetic and P_norm depend on P, so both should flag.
    assert "kinetic" in report.mismatches
    assert "P_norm" in report.mismatches


def test_fingerprint_detects_perturbed_embedding():
    torch.manual_seed(2)
    E = torch.randn(8, 4)
    P = torch.randn(8, 4)
    mass = torch.ones(8, 1)
    saved = compute_integrity(E, P, mass)
    E_tampered = E.clone()
    E_tampered[3, 2] -= 0.5
    loaded = compute_integrity(E_tampered, P, mass)
    report = verify_integrity(saved, loaded)
    assert not report.ok
    assert "E_norm" in report.mismatches
    assert "kinetic" not in report.mismatches


def test_serialisation_roundtrips_through_dict():
    fp = CheckpointIntegrity(kinetic=1.25, E_norm=2.5, P_norm=3.75)
    restored = CheckpointIntegrity.from_dict(fp.to_dict())
    assert restored == fp


def test_report_format_is_human_readable():
    saved = CheckpointIntegrity(kinetic=1.0, E_norm=2.0, P_norm=3.0)
    loaded = CheckpointIntegrity(kinetic=1.0, E_norm=2.0, P_norm=3.0)
    ok = verify_integrity(saved, loaded).format()
    assert "OK" in ok

    bad_loaded = CheckpointIntegrity(kinetic=99.0, E_norm=2.0, P_norm=3.0)
    msg = verify_integrity(saved, bad_loaded).format()
    assert "MISMATCH" in msg
    assert "kinetic" in msg


# ---- trainer integration -------------------------------------------------

def _tiny_config(tmp_path):
    return {
        "embedding_dim": 4,
        "hidden_dim": 6,
        "n_layers": 1,
        "dt": 0.01,
        "learning_rate_W": 1e-3,
        "gradient_clip_norm_W": 1.0,
        "context_window": 6,
        "batch_size": 4,
        "num_steps": 2,
        "eval_interval": 2,
        "eval_iters": 2,
        "eval_batch_size": 2,
        "results_dir": str(tmp_path),
    }


def _tiny_setup(trainer, vocab_size=8):
    train_data = torch.arange(200, dtype=torch.long) % vocab_size
    test_data = torch.arange(120, dtype=torch.long) % vocab_size
    char_to_idx = {chr(ord("a") + i): i for i in range(vocab_size)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    mass = torch.ones(vocab_size, 1)
    trainer.setup(train_data, test_data, char_to_idx, idx_to_char, vocab_size, mass)


def test_trainer_writes_integrity_block_to_checkpoint(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    list(trainer.train())
    ckpt_path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "integrity" in blob
    assert "mass_vector" in blob
    fp = CheckpointIntegrity.from_dict(blob["integrity"])
    # The stored fingerprint must match a fresh one computed from the
    # checkpoint's own tensors — if this fails, save and verify disagree.
    fresh = compute_integrity(blob["E"], blob["P"], blob["mass_vector"])
    assert verify_integrity(fp, fresh).ok


def test_trainer_load_accepts_untampered_checkpoint(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    list(trainer.train())
    ckpt_path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")

    reloaded = Trainer(_tiny_config(tmp_path))
    assert reloaded.load_checkpoint(ckpt_path) is True


def test_trainer_load_rejects_tampered_checkpoint(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    list(trainer.train())
    ckpt_path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    blob["P"] = blob["P"] + 1.0  # tamper without updating the fingerprint
    torch.save(blob, ckpt_path)

    reloaded = Trainer(_tiny_config(tmp_path))
    assert reloaded.load_checkpoint(ckpt_path) is False


def test_trainer_load_survives_legacy_checkpoint_without_integrity(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(_tiny_config(tmp_path))
    _tiny_setup(trainer)
    list(trainer.train())
    ckpt_path = os.path.join(trainer.save_dir, "checkpoint_latest.pt")

    # Simulate a pre-integrity checkpoint by stripping the new fields.
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    blob.pop("integrity", None)
    blob.pop("mass_vector", None)
    torch.save(blob, ckpt_path)

    reloaded = Trainer(_tiny_config(tmp_path))
    assert reloaded.load_checkpoint(ckpt_path) is True
