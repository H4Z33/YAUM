"""Hamiltonian substrate for embedding phase-state evolution.

The trainer updates model weights with a conventional optimiser, but the
embedding table lives in a separate phase space. ``HamiltonianSubstrate``
keeps that phase state explicit so the integrator contract is not hidden
inside trainer control flow.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .integrators import ForceField, PhaseStep, SymplecticIntegrator


@dataclass
class HamiltonianSubstrate:
    """State and stepping API for a Hamiltonian embedding table."""

    E: torch.Tensor
    P: torch.Tensor
    mass: torch.Tensor
    integrator: SymplecticIntegrator
    dt: float

    @classmethod
    def random(
        cls,
        vocab_size: int,
        embedding_dim: int,
        *,
        mass: torch.Tensor,
        integrator: SymplecticIntegrator,
        dt: float,
        device,
        dtype: torch.dtype = torch.float32,
    ) -> "HamiltonianSubstrate":
        E = torch.randn(
            vocab_size,
            embedding_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        P = torch.zeros(vocab_size, embedding_dim, device=device, dtype=dtype)
        return cls(E=E, P=P, mass=mass, integrator=integrator, dt=float(dt))

    def configure(
        self,
        *,
        mass: torch.Tensor | None = None,
        integrator: SymplecticIntegrator | None = None,
        dt: float | None = None,
    ) -> None:
        """Refresh runtime knobs without replacing the phase state."""
        if mass is not None:
            self.mass = mass
        if integrator is not None:
            self.integrator = integrator
        if dt is not None:
            self.dt = float(dt)

    def propose(self, force_fn: ForceField, *, retain_final: bool = True) -> PhaseStep:
        """Return the next phase step without mutating the committed state."""
        E_leaf = self.E if self.E.requires_grad else self.E.detach().requires_grad_(True)
        return self.integrator.step(
            E_leaf,
            self.P,
            self.dt,
            force_fn,
            self.mass,
            retain_final=retain_final,
        )

    def commit(self, step: PhaseStep) -> None:
        """Commit a proposed phase step after the trainer has used its loss graph."""
        self.E = step.E.detach().requires_grad_(True)
        self.P = step.P.detach()
