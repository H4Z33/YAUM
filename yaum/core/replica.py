"""Replica exchange (parallel tempering) for embedding dynamics.

Running K trajectories at temperatures β₀ > β₁ > ... > β_{K-1} and
periodically swapping adjacent pairs via a Metropolis test gives the
cold trajectory access to basins that are only reachable through the
high-entropy detours of the hot replicas. For an error landscape with a
grokking-style phase transition — a narrow generalising basin separated
from the lossy memoriser basin by a ridge — parallel tempering is the
cleanest way to tunnel across.

The swap acceptance between replicas (i at β_i) and (j at β_j) holding
Hamiltonians H_i and H_j is the canonical PT formula
    α = min(1, exp((β_i − β_j)(H_i − H_j)))
which is exactly Metropolis-Hastings on the joint distribution
``π(x_i, x_j) ∝ exp(−β_i H(x_i) − β_j H(x_j))``. If the colder replica
usually has lower energy the swap is rejected most of the time; when a
hotter replica stumbles into a deeper basin the swap is accepted and
the cold trajectory inherits the find.

This module provides pure, integrator-agnostic primitives:

* :func:`metropolis_swap_prob` — the acceptance probability,
* :func:`attempt_swap`         — one swap trial with that probability,
* :class:`ReplicaState`        — an ``(E, P, H)`` container,
* :class:`ReplicaEnsemble`     — K states, a swap scheduler, swap stats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ReplicaState:
    """One replica's phase-space point and its current Hamiltonian."""

    E: torch.Tensor
    P: torch.Tensor
    H: float


def metropolis_swap_prob(
    beta_i: float, beta_j: float, H_i: float, H_j: float
) -> float:
    """Acceptance probability for a PT exchange between replicas i and j."""
    if beta_i <= 0.0 or beta_j <= 0.0:
        raise ValueError("betas must be positive")
    exponent = (beta_i - beta_j) * (H_i - H_j)
    if exponent >= 0.0:
        return 1.0
    return math.exp(exponent)


def attempt_swap(
    a: ReplicaState,
    b: ReplicaState,
    beta_a: float,
    beta_b: float,
    *,
    rng: Optional[torch.Generator] = None,
) -> bool:
    """Maybe exchange the configurations of two replicas in place.

    Only the tensors and the cached ``H`` are swapped — the temperatures
    stay with their replica slot, which is the standard PT convention.
    Returns ``True`` iff the swap was accepted.
    """
    p = metropolis_swap_prob(beta_a, beta_b, a.H, b.H)
    u = float(torch.rand((), generator=rng).item())
    if u < p:
        a.E, b.E = b.E, a.E
        a.P, b.P = b.P, a.P
        a.H, b.H = b.H, a.H
        return True
    return False


@dataclass
class ReplicaEnsemble:
    """K replicas at fixed temperatures with a round-robin swap scheduler.

    ``betas`` is the inverse-temperature ladder in strictly descending
    order — the coldest replica first. The swap scheduler alternates
    between even pairs ``(0,1),(2,3),...`` and odd pairs
    ``(1,2),(3,4),...`` so no single replica boundary is neglected.
    """

    states: list[ReplicaState]
    betas: list[float]
    _sweep: int = field(default=0, init=False)
    _accepts: list[int] = field(default_factory=list, init=False)
    _attempts: list[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        if len(self.states) != len(self.betas):
            raise ValueError("states and betas must have the same length")
        if len(self.states) < 2:
            raise ValueError("need at least 2 replicas")
        if any(
            self.betas[i] <= self.betas[i + 1] for i in range(len(self.betas) - 1)
        ):
            raise ValueError("betas must be strictly descending (cold first)")
        self._accepts = [0] * (len(self.states) - 1)
        self._attempts = [0] * (len(self.states) - 1)

    @property
    def n_replicas(self) -> int:
        return len(self.states)

    def sweep_once(
        self, *, rng: Optional[torch.Generator] = None
    ) -> list[bool]:
        """Attempt one round of adjacent-pair swaps (even- or odd-offset)."""
        start = self._sweep % 2
        results: list[bool] = []
        for i in range(start, self.n_replicas - 1, 2):
            ok = attempt_swap(
                self.states[i],
                self.states[i + 1],
                self.betas[i],
                self.betas[i + 1],
                rng=rng,
            )
            self._attempts[i] += 1
            if ok:
                self._accepts[i] += 1
            results.append(ok)
        self._sweep += 1
        return results

    def swap_rates(self) -> list[float]:
        """Empirical acceptance rate per inter-replica boundary."""
        return [
            (self._accepts[i] / self._attempts[i]) if self._attempts[i] else 0.0
            for i in range(self.n_replicas - 1)
        ]

    def coldest(self) -> ReplicaState:
        return self.states[0]
