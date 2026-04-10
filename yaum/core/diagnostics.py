"""Runtime diagnostics for Hamiltonian dynamics.

The central diagnostic here is *time-reversibility*: if the integrator is
symplectic and the force field is autonomous, integrating forward ``n``
steps and then backward ``n`` steps should return the system to its
starting state up to floating-point roundoff. Any residual is a direct,
quantitative signal that something has broken the conservative structure
— numerical instability, nondeterminism in the model, or a dt that no
longer fits the local curvature of the loss landscape.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .integrators import ForceField, SymplecticIntegrator


@dataclass
class ReversibilityResidual:
    """Residual of a forward/backward round trip through the integrator."""

    position: float
    momentum: float

    @property
    def total(self) -> float:
        return (self.position * self.position + self.momentum * self.momentum) ** 0.5


def measure_reversibility(
    integrator: SymplecticIntegrator,
    E0: torch.Tensor,
    P0: torch.Tensor,
    mass: torch.Tensor,
    force_fn: ForceField,
    dt: float,
    n_steps: int,
) -> ReversibilityResidual:
    """Run ``n_steps`` forward then backward; return L2 residuals.

    The backward leg uses ``-dt`` so a time-symmetric integrator inverts
    itself exactly. ``force_fn`` must be deterministic for the duration of
    the call — typically the caller freezes model weights and uses the
    same (context, target) batch throughout.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    E_ref = E0.detach().clone()
    P_ref = P0.detach().clone()

    E = E_ref.clone().requires_grad_(True)
    P = P_ref.clone()

    for _ in range(n_steps):
        step = integrator.step(E, P, dt, force_fn, mass, retain_final=False)
        E = step.E.requires_grad_(True)
        P = step.P

    for _ in range(n_steps):
        step = integrator.step(E, P, -dt, force_fn, mass, retain_final=False)
        E = step.E.requires_grad_(True)
        P = step.P

    pos_res = torch.linalg.norm(E.detach() - E_ref).item()
    mom_res = torch.linalg.norm(P - P_ref).item()
    return ReversibilityResidual(position=pos_res, momentum=mom_res)
