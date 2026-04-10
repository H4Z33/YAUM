"""Specs for the phase-transition observables module."""
from __future__ import annotations

import math

import pytest
import torch

from yaum.core.observables import (
    ObservableRecord,
    ObservableWindow,
    embedding_entropy_logdet,
    integrated_autocorr,
    snapshot,
)


# ---- entropy -------------------------------------------------------------

def test_entropy_logdet_drops_as_rows_collapse():
    torch.manual_seed(0)
    wide = torch.randn(64, 4)
    narrow = wide * 0.01
    assert embedding_entropy_logdet(narrow) < embedding_entropy_logdet(wide)


def test_entropy_logdet_is_nan_for_degenerate_shape():
    assert math.isnan(embedding_entropy_logdet(torch.zeros(1, 4)))


# ---- integrated autocorrelation time -----------------------------------

def test_integrated_autocorr_of_iid_noise_is_near_one():
    torch.manual_seed(0)
    series = torch.randn(256).tolist()
    tau = integrated_autocorr(series)
    assert 0.5 <= tau <= 2.5


def test_integrated_autocorr_of_slow_drift_is_large():
    # Strongly autocorrelated series: AR(1) with rho=0.95
    torch.manual_seed(0)
    x = [0.0]
    for _ in range(512):
        x.append(0.95 * x[-1] + float(torch.randn(1)))
    iid = integrated_autocorr(torch.randn(len(x)).tolist())
    slow = integrated_autocorr(x)
    assert slow > iid * 2.0
    assert slow >= 2.0


def test_integrated_autocorr_bails_out_on_tiny_series():
    assert math.isnan(integrated_autocorr([1.0]))
    assert math.isnan(integrated_autocorr([]))


# ---- window ---------------------------------------------------------------

def test_window_rejects_size_below_two():
    with pytest.raises(ValueError):
        ObservableWindow(size=1)


def test_window_returns_none_until_two_records_pushed():
    w = ObservableWindow(size=8)
    assert w.report() is None
    w.push(ObservableRecord(H=1.0, M=0.1, S=0.0))
    assert w.report() is None
    w.push(ObservableRecord(H=1.0, M=0.1, S=0.0))
    assert w.report() is not None


def test_specific_heat_zero_on_constant_hamiltonian():
    w = ObservableWindow(size=8)
    for _ in range(8):
        w.push(ObservableRecord(H=3.14, M=0.5, S=0.0))
    r = w.report()
    assert r.specific_heat == pytest.approx(0.0)
    assert r.susceptibility == pytest.approx(0.0)


def test_specific_heat_positive_on_oscillating_hamiltonian():
    w = ObservableWindow(size=8)
    for i in range(8):
        w.push(ObservableRecord(H=float((-1) ** i), M=0.0, S=0.0))
    r = w.report()
    assert r.specific_heat > 0.5  # Var of ±1 sequence is exactly 1.0


def test_entropy_rate_is_end_minus_start_over_intervals():
    w = ObservableWindow(size=5)
    for i, s in enumerate([1.0, 1.25, 1.5, 1.75, 2.0]):
        w.push(ObservableRecord(H=0.0, M=0.0, S=s))
    r = w.report()
    assert r.entropy_rate == pytest.approx(0.25)


def test_window_overflow_drops_oldest():
    w = ObservableWindow(size=3)
    for h in (1.0, 2.0, 3.0, 4.0):
        w.push(ObservableRecord(H=h, M=0.0, S=0.0))
    assert len(w) == 3
    # Only 2,3,4 should remain — Var of that is 2/3
    r = w.report()
    assert r.specific_heat == pytest.approx(((2 - 3) ** 2 + 0 + (4 - 3) ** 2) / 3)


def test_window_skips_nan_records_gracefully():
    w = ObservableWindow(size=4)
    w.push(ObservableRecord(H=1.0, M=0.0, S=0.0))
    w.push(ObservableRecord(H=float("nan"), M=0.0, S=0.0))
    w.push(ObservableRecord(H=2.0, M=0.0, S=0.0))
    r = w.report()
    # Two valid Hs => finite specific heat
    assert math.isfinite(r.specific_heat)


# ---- snapshot helper ------------------------------------------------------

def test_snapshot_captures_phase_state():
    torch.manual_seed(0)
    E = torch.randn(16, 4)
    rec = snapshot(E, H=2.5)
    assert rec.H == pytest.approx(2.5)
    assert rec.M == pytest.approx(float(torch.linalg.norm(E.mean(dim=0))))
    assert math.isfinite(rec.S)
