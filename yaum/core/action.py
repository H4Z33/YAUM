"""Action functional ``S = ∫(T − V) dt`` as a trajectory invariant.

The classical principle of least action says physical trajectories
extremize ``S[q] = ∫ L(q, q̇) dt`` where ``L = T − V`` is the
Lagrangian. For a symplectic integrator the action along the discrete
trajectory is a smooth function of time — its increments are the
Lagrangian times ``dt``, and plotting the per-step ``δS`` gives a clean,
scale-free signal of trajectory regularity that is orthogonal to the
Hamiltonian drift already tracked.

Symplectic integrators of order ``p`` conserve a *shadow* Hamiltonian
exactly; the action accumulated along their trajectories is exact for
the shadow system and ``O(dt^p)`` close to the true one. So a spike in
``δS`` that does not coincide with an energy-drift spike is a hint that
something non-conservative (noise, clipping, checkpoint reload) has
perturbed the state.

The trainer calls :meth:`ActionAccumulator.accumulate` on **every**
step with the increment ``(T − V)·dt`` (kept as a device tensor so no
host sync is forced) and :meth:`ActionAccumulator.observe` on eval
steps to read the running total. :meth:`update` remains for callers
that integrate scalars directly.

Caveat: ``V`` is the minibatch loss and the weights ``W`` change every
step, so ``L`` is not the Lagrangian of a fixed system — ``S`` is a
bookkeeping integral, not something the trajectory extremizes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionReport:
    """Snapshot of the running action integral."""

    S: float   # cumulative action so far
    L: float   # current Lagrangian T − V
    dS: float  # most recent increment L · dt


class ActionAccumulator:
    """Running left-point integral of the Lagrangian along a trajectory.

    Each :meth:`update` call takes the current ``(T, V, dt)`` and
    advances the stored action by ``(T − V) · dt``. The left-point rule
    is the only increment we can evaluate *before* the next step has
    taken place, and it matches the per-step loss telemetry the trainer
    already records.
    """

    def __init__(self) -> None:
        self._S: float = 0.0
        self._L: float = 0.0
        self._dS: float = 0.0
        self._steps: int = 0
        self._last_observed_S: float = 0.0

    def reset(self) -> None:
        self._S = 0.0
        self._L = 0.0
        self._dS = 0.0
        self._steps = 0
        self._last_observed_S = 0.0

    def update(self, kinetic: float, potential: float, dt: float) -> ActionReport:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        L = float(kinetic) - float(potential)
        dS = L * float(dt)
        self._L = L
        self._dS = dS
        self._S += dS
        self._steps += 1
        return ActionReport(S=self._S, L=L, dS=dS)

    def accumulate(self, dS) -> None:
        """Add one step's increment ``(T − V)·dt``.

        ``dS`` may be a float or a 0-d tensor; tensors stay on their
        device until :meth:`observe` or :meth:`report` reads them.
        """
        self._S = self._S + dS
        self._steps += 1

    def observe(self, kinetic: float, potential: float) -> ActionReport:
        """Record the current Lagrangian and return the running total."""
        L = float(kinetic) - float(potential)
        S = float(self._S)
        self._dS = S - getattr(self, "_last_observed_S", 0.0)
        self._last_observed_S = S
        self._S = S
        self._L = L
        return ActionReport(S=S, L=L, dS=self._dS)

    def report(self) -> ActionReport:
        return ActionReport(S=float(self._S), L=self._L, dS=self._dS)

    @property
    def steps(self) -> int:
        return self._steps
