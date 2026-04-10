"""Specs for replica exchange / parallel tempering."""
from __future__ import annotations

import math

import pytest
import torch

from yaum.core.replica import (
    ReplicaEnsemble,
    ReplicaState,
    attempt_swap,
    metropolis_swap_prob,
)


def _make_state(val: float, H: float) -> ReplicaState:
    E = torch.full((3, 2), val)
    P = torch.full((3, 2), val)
    return ReplicaState(E=E, P=P, H=H)


# ---- metropolis probability --------------------------------------------


def test_swap_always_accepted_when_cold_has_higher_energy():
    # β_cold > β_hot, H_cold > H_hot => exponent positive => p = 1
    assert metropolis_swap_prob(beta_i=2.0, beta_j=1.0, H_i=5.0, H_j=1.0) == 1.0


def test_swap_prob_formula_when_cold_has_lower_energy():
    # (β_i − β_j)(H_i − H_j) = (2 − 1)(1 − 5) = −4  =>  p = e^{−4}
    p = metropolis_swap_prob(2.0, 1.0, 1.0, 5.0)
    assert math.isclose(p, math.exp(-4.0))


def test_swap_prob_rejects_non_positive_beta():
    with pytest.raises(ValueError):
        metropolis_swap_prob(0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        metropolis_swap_prob(1.0, -1.0, 0.0, 0.0)


# ---- attempt_swap ------------------------------------------------------


def test_attempt_swap_swaps_configs_when_accepted():
    a = _make_state(1.0, H=10.0)  # cold, high H
    b = _make_state(2.0, H=1.0)   # hot,  low  H
    # α = 1 here, so always accepted regardless of RNG.
    ok = attempt_swap(a, b, beta_a=2.0, beta_b=1.0)
    assert ok is True
    assert torch.allclose(a.E, torch.full_like(a.E, 2.0))
    assert torch.allclose(b.E, torch.full_like(b.E, 1.0))
    assert a.H == 1.0
    assert b.H == 10.0


def test_attempt_swap_rejected_keeps_state_untouched():
    a = _make_state(1.0, H=0.0)
    b = _make_state(2.0, H=100.0)
    # exponent = (2-1)*(0-100) = -100 => p ≈ 3.7e-44, always rejected.
    g = torch.Generator().manual_seed(0)
    ok = attempt_swap(a, b, 2.0, 1.0, rng=g)
    assert ok is False
    assert torch.allclose(a.E, torch.full_like(a.E, 1.0))
    assert torch.allclose(b.E, torch.full_like(b.E, 2.0))
    assert a.H == 0.0
    assert b.H == 100.0


# ---- ensemble validation -----------------------------------------------


def test_ensemble_requires_descending_betas():
    with pytest.raises(ValueError):
        ReplicaEnsemble(
            states=[_make_state(0.0, 0.0), _make_state(1.0, 0.0)],
            betas=[1.0, 2.0],  # ascending — rejected
        )


def test_ensemble_requires_at_least_two_replicas():
    with pytest.raises(ValueError):
        ReplicaEnsemble(states=[_make_state(0.0, 0.0)], betas=[1.0])


def test_ensemble_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ReplicaEnsemble(
            states=[_make_state(0.0, 0.0), _make_state(1.0, 0.0)],
            betas=[2.0, 1.0, 0.5],
        )


# ---- sweep schedule ----------------------------------------------------


def test_ensemble_alternates_even_and_odd_sweeps():
    states = [_make_state(float(i), 0.0) for i in range(4)]
    betas = [4.0, 3.0, 2.0, 1.0]
    ens = ReplicaEnsemble(states=states, betas=betas)
    # With H = 0 everywhere, every swap has p = 1.

    # Sweep 0 → even pairs (0,1) and (2,3).
    res = ens.sweep_once()
    assert res == [True, True]
    assert torch.allclose(ens.states[0].E, torch.full_like(ens.states[0].E, 1.0))
    assert torch.allclose(ens.states[1].E, torch.full_like(ens.states[1].E, 0.0))
    assert torch.allclose(ens.states[2].E, torch.full_like(ens.states[2].E, 3.0))
    assert torch.allclose(ens.states[3].E, torch.full_like(ens.states[3].E, 2.0))

    # Sweep 1 → odd pair (1,2) — exactly one attempt.
    res = ens.sweep_once()
    assert res == [True]
    # The content that was at slot 1 (originally 0.0) and slot 2 (originally 3.0)
    # should now be exchanged.
    assert torch.allclose(ens.states[1].E, torch.full_like(ens.states[1].E, 3.0))
    assert torch.allclose(ens.states[2].E, torch.full_like(ens.states[2].E, 0.0))


def test_ensemble_swap_rates_tracked_per_boundary():
    states = [_make_state(float(i), 0.0) for i in range(3)]
    betas = [3.0, 2.0, 1.0]
    ens = ReplicaEnsemble(states=states, betas=betas)
    for _ in range(10):
        ens.sweep_once()
    rates = ens.swap_rates()
    assert len(rates) == 2
    # Every attempt has p=1 (H=0 everywhere) so all accepted.
    assert rates[0] == 1.0
    assert rates[1] == 1.0


def test_ensemble_coldest_is_position_zero():
    states = [_make_state(0.0, 0.0), _make_state(1.0, 0.0)]
    betas = [2.0, 1.0]
    ens = ReplicaEnsemble(states=states, betas=betas)
    assert ens.coldest() is ens.states[0]


# ---- tunneling smoke test ----------------------------------------------


def test_pt_transports_good_configuration_from_hot_to_cold():
    """Cold replica is stuck in a lossy basin; hot replica has stumbled
    into a deep one. One sweep should hand the cold slot the good config.
    """
    states = [
        _make_state(val=99.0, H=100.0),  # cold slot, stuck in bad basin
        _make_state(val=1.0, H=0.0),     # hot slot,  in the good basin
    ]
    betas = [10.0, 1.0]
    ens = ReplicaEnsemble(states=states, betas=betas)
    # exponent = (10 − 1)(100 − 0) = +900  =>  p = 1  =>  always accepted.
    assert ens.sweep_once() == [True]
    assert torch.allclose(ens.coldest().E, torch.full_like(ens.coldest().E, 1.0))
    assert ens.coldest().H == 0.0
