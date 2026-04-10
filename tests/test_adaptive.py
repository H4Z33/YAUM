"""Specs for the adaptive time-step controller."""
from __future__ import annotations

import pytest

from yaum.core.adaptive import AdaptiveStepController


def _make(**overrides):
    base = dict(
        dt_init=0.01,
        dt_min=1e-4,
        dt_max=0.1,
        drift_high=1e-3,
        drift_low=1e-5,
        shrink=0.5,
        grow=2.0,
        grow_after=3,
    )
    base.update(overrides)
    return AdaptiveStepController(**base)


def test_initial_dt_is_clamped():
    c = _make(dt_init=10.0)
    assert c.dt == c.dt_max
    c = _make(dt_init=1e-9)
    assert c.dt == c.dt_min


def test_drift_spike_shrinks_dt():
    c = _make()
    start = c.dt
    next_dt = c.observe(drift=1e-2)
    assert next_dt == start * 0.5
    assert c.stats["shrinks"] == 1
    assert c.stats["calm_streak"] == 0


def test_drift_spike_never_below_dt_min():
    c = _make(dt_init=2e-4)
    for _ in range(50):
        c.observe(drift=1.0)
    assert c.dt == c.dt_min


def test_sustained_calm_grows_dt_after_streak():
    c = _make()
    start = c.dt
    for _ in range(2):
        c.observe(drift=1e-7)
    assert c.dt == start  # not yet — streak not satisfied
    c.observe(drift=1e-7)
    assert c.dt == start * 2.0
    assert c.stats["grows"] == 1
    assert c.stats["calm_streak"] == 0  # reset after growing


def test_growth_is_clamped_to_dt_max():
    c = _make(dt_init=0.09, grow_after=1, grow=2.0)
    for _ in range(10):
        c.observe(drift=1e-9)
    assert c.dt == c.dt_max


def test_middle_of_range_keeps_dt_and_resets_streak():
    c = _make()
    c.observe(drift=1e-7)
    c.observe(drift=1e-7)
    assert c.stats["calm_streak"] == 2
    c.observe(drift=1e-4)  # inside (drift_low, drift_high) — neutral
    assert c.stats["calm_streak"] == 0
    assert c.dt == c.dt_init


def test_spike_after_calm_shrinks_and_resets_streak():
    c = _make()
    c.observe(drift=1e-7)
    c.observe(drift=1e-7)
    assert c.stats["calm_streak"] == 2
    c.observe(drift=1e-2)
    assert c.stats["calm_streak"] == 0
    assert c.dt < c.dt_init


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        AdaptiveStepController(
            dt_init=0.01, dt_min=0.01, dt_max=0.01,
            drift_high=1e-5, drift_low=1e-3,  # inverted
        )
    with pytest.raises(ValueError):
        AdaptiveStepController(
            dt_init=0.01, dt_min=0.0, dt_max=0.01,
            drift_high=1e-3, drift_low=1e-5,
        )
    with pytest.raises(ValueError):
        AdaptiveStepController(
            dt_init=0.01, dt_min=0.01, dt_max=0.1,
            drift_high=1e-3, drift_low=1e-5, shrink=1.5,
        )
    with pytest.raises(ValueError):
        AdaptiveStepController(
            dt_init=0.01, dt_min=0.01, dt_max=0.1,
            drift_high=1e-3, drift_low=1e-5, grow=0.9,
        )
