"""Specs for the action-functional telemetry."""
from __future__ import annotations

import pytest

from yaum.core.action import ActionAccumulator


def test_fresh_accumulator_is_zero():
    acc = ActionAccumulator()
    r = acc.report()
    assert r.S == 0.0
    assert r.L == 0.0
    assert r.dS == 0.0
    assert acc.steps == 0


def test_single_update_computes_lagrangian_and_increment():
    acc = ActionAccumulator()
    r = acc.update(kinetic=3.0, potential=1.0, dt=0.1)
    assert r.L == pytest.approx(2.0)
    assert r.dS == pytest.approx(0.2)
    assert r.S == pytest.approx(0.2)
    assert acc.steps == 1


def test_multiple_updates_integrate_over_trajectory():
    acc = ActionAccumulator()
    acc.update(2.0, 1.0, 0.1)  # L = 1,   dS = 0.1
    acc.update(5.0, 1.0, 0.1)  # L = 4,   dS = 0.4
    acc.update(0.5, 1.0, 0.1)  # L = -0.5, dS = -0.05
    r = acc.report()
    assert r.S == pytest.approx(0.1 + 0.4 - 0.05)
    assert r.L == pytest.approx(-0.5)    # last L
    assert r.dS == pytest.approx(-0.05)  # last increment
    assert acc.steps == 3


def test_reset_clears_state():
    acc = ActionAccumulator()
    acc.update(2.0, 1.0, 0.1)
    acc.update(2.0, 1.0, 0.1)
    acc.reset()
    r = acc.report()
    assert r.S == 0.0
    assert r.L == 0.0
    assert r.dS == 0.0
    assert acc.steps == 0


def test_update_rejects_negative_dt():
    acc = ActionAccumulator()
    with pytest.raises(ValueError):
        acc.update(1.0, 0.0, -0.01)


def test_action_stays_exactly_zero_when_lagrangian_vanishes():
    # T == V everywhere => L ≡ 0 => action never accumulates.
    acc = ActionAccumulator()
    for _ in range(50):
        acc.update(kinetic=2.5, potential=2.5, dt=0.01)
    assert acc.report().S == 0.0
    assert acc.steps == 50


def test_constant_lagrangian_gives_linear_action_growth():
    acc = ActionAccumulator()
    for _ in range(10):
        acc.update(kinetic=3.0, potential=1.0, dt=0.05)
    # L = 2, dS = 0.1, after 10 steps S = 1.0
    assert acc.report().S == pytest.approx(1.0)
