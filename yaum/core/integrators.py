"""Symplectic integrators for Hamiltonian dynamics in semantic phase space.

An integrator is the only primitive a time-bound system has: it decides how
the next (E, P) follows from the current one. Everything the model "knows"
is encoded in the trajectory that this stepping function traces out.

Higher-order methods are built by composing lower-order ones. Yoshida's
4th-order scheme is three leapfrog steps; each leapfrog step is two half
kicks around a drift. Turtles all the way down.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Callable, Protocol, Union

import torch


class ForceField(Protocol):
    """Evaluates (loss, force) at a given phase-space position E.

    The force must be broadcastable onto E (same shape). Sparse gradients
    — nonzero only at active tokens — are allowed and expected.
    """

    def __call__(
        self, E: torch.Tensor, *, retain_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class PhaseStep:
    """Outcome of one symplectic step.

    ``loss_final`` may still carry an autograd graph when ``retain_final``
    is True so the caller can backprop through it to update model weights.
    """

    E: torch.Tensor
    P: torch.Tensor
    loss_initial: torch.Tensor
    loss_final: torch.Tensor
    n_force_evals: int


class SymplecticIntegrator:
    order: int = 0
    name: str = "base"

    def step(
        self,
        E: torch.Tensor,
        P: torch.Tensor,
        dt: float,
        force_fn: ForceField,
        mass: torch.Tensor,
        retain_final: bool = True,
    ) -> PhaseStep:
        raise NotImplementedError


class LeapfrogIntegrator(SymplecticIntegrator):
    """Velocity-Verlet leapfrog. Second-order, time-reversible."""

    order = 2
    name = "leapfrog"

    def step(self, E, P, dt, force_fn, mass, retain_final=True):
        loss1, F1 = force_fn(E, retain_graph=False)
        P_half = P + F1 * (dt * 0.5)

        E_mid = E.detach() + (P_half / mass) * dt
        E_new = E_mid.requires_grad_(True)

        loss2, F2 = force_fn(E_new, retain_graph=retain_final)
        P_new = P_half + F2 * (dt * 0.5)

        return PhaseStep(
            E=E_new.detach(),
            P=P_new.detach(),
            loss_initial=loss1.detach(),
            loss_final=loss2,
            n_force_evals=2,
        )


class YoshidaIntegrator(SymplecticIntegrator):
    """Yoshida (1990) 4th-order composition of leapfrog.

    Three sub-steps with weights (w1, w2, w1) where

        w1 = 1 / (2 - 2**(1/3))
        w2 = 1 - 2 * w1

    The middle step is negative. It cancels the leading O(dt**3) error of
    leapfrog, giving O(dt**5) local error.
    """

    order = 4
    name = "yoshida4"

    def __init__(self):
        cbrt2 = 2.0 ** (1.0 / 3.0)
        self.w1 = 1.0 / (2.0 - cbrt2)
        self.w2 = 1.0 - 2.0 * self.w1

    def step(self, E, P, dt, force_fn, mass, retain_final=True):
        leap = LeapfrogIntegrator()
        s1 = leap.step(E, P, dt * self.w1, force_fn, mass, retain_final=False)
        E1 = s1.E.requires_grad_(True)
        s2 = leap.step(E1, s1.P, dt * self.w2, force_fn, mass, retain_final=False)
        E2 = s2.E.requires_grad_(True)
        s3 = leap.step(
            E2, s2.P, dt * self.w1, force_fn, mass, retain_final=retain_final
        )
        return PhaseStep(
            E=s3.E,
            P=s3.P,
            loss_initial=s1.loss_initial,
            loss_final=s3.loss_final,
            n_force_evals=s1.n_force_evals + s2.n_force_evals + s3.n_force_evals,
        )


class LangevinIntegrator(SymplecticIntegrator):
    """OBABO-split Langevin dynamics (Leimkuhler & Matthews, 2013).

    A deterministic symplectic core is flanked by two Ornstein–Uhlenbeck
    half-steps that exchange momentum with a heat bath at temperature
    ``kT`` with friction ``gamma``. The resulting flow samples the
    canonical ensemble, trading strict energy conservation for the
    ability to explore ``(E, P)`` configurations at a fixed temperature.

    The core defaults to leapfrog but accepts any other
    :class:`SymplecticIntegrator` — passing ``"yoshida4"`` gives
    fourth-order deterministic accuracy inside the thermostat.
    """

    name = "langevin"
    order = 2

    def __init__(
        self,
        friction: float = 1.0,
        temperature: float = 1.0,
        core: Union[str, "SymplecticIntegrator"] = "leapfrog",
    ):
        if friction < 0:
            raise ValueError("friction must be >= 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.gamma = float(friction)
        self.kT = float(temperature)
        if isinstance(core, str):
            if core not in _CORE_INTEGRATORS:
                raise ValueError(
                    f"Unknown core integrator {core!r}. "
                    f"Available: {sorted(_CORE_INTEGRATORS)}"
                )
            self.core = _CORE_INTEGRATORS[core]()
        else:
            self.core = core

    def _ou_half(self, P: torch.Tensor, half_dt: float, mass: torch.Tensor) -> torch.Tensor:
        if self.gamma == 0.0:
            return P
        c = math.exp(-self.gamma * half_dt)
        variance = mass * self.kT * (1.0 - c * c)
        sigma = torch.sqrt(torch.clamp(variance, min=0.0))
        noise = torch.randn_like(P)
        return c * P + sigma * noise

    def step(self, E, P, dt, force_fn, mass, retain_final=True):
        P_in = self._ou_half(P, dt * 0.5, mass)
        inner = self.core.step(E, P_in, dt, force_fn, mass, retain_final=retain_final)
        P_out = self._ou_half(inner.P, dt * 0.5, mass)
        return PhaseStep(
            E=inner.E,
            P=P_out,
            loss_initial=inner.loss_initial,
            loss_final=inner.loss_final,
            n_force_evals=inner.n_force_evals,
        )


_CORE_INTEGRATORS: dict[str, type[SymplecticIntegrator]] = {
    LeapfrogIntegrator.name: LeapfrogIntegrator,
    YoshidaIntegrator.name: YoshidaIntegrator,
}

INTEGRATORS: dict[str, type[SymplecticIntegrator]] = {
    **_CORE_INTEGRATORS,
    LangevinIntegrator.name: LangevinIntegrator,
}


def make_integrator(name: str, **params) -> SymplecticIntegrator:
    """Instantiate an integrator by name, forwarding any parameters it accepts.

    Unknown keyword arguments are silently dropped so callers can pass a
    loose config dict without having to know which integrator they are
    constructing.
    """
    try:
        cls = INTEGRATORS[name]
    except KeyError as err:
        raise ValueError(
            f"Unknown integrator {name!r}. Available: {sorted(INTEGRATORS)}"
        ) from err
    signature = inspect.signature(cls.__init__)
    accepted = {
        key: value
        for key, value in params.items()
        if key in signature.parameters
    }
    return cls(**accepted)
