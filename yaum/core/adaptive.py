"""Adaptive time-step controller driven by Hamiltonian drift.

"Time is all it has": the system's only knob is ``dt``, so we let it decide
its own stride from the only feedback it can measure — how well the current
integrator preserves the total energy ``H = T + V``.

The controller is a pure function of observed drift. It never rolls state
back; it simply proposes the ``dt`` for the *next* step. That keeps the
training loop simple and the controller unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveStepController:
    """Shrink ``dt`` on drift spikes; grow it after sustained calm.

    Parameters
    ----------
    dt_init:
        Starting step size.
    dt_min, dt_max:
        Hard clamps.
    drift_high:
        If observed drift exceeds this, shrink immediately.
    drift_low:
        If observed drift stays below this for ``grow_after`` consecutive
        steps, grow.
    shrink, grow:
        Multiplicative factors (``0 < shrink < 1 < grow``).
    grow_after:
        Number of consecutive "calm" steps required before growing.
    """

    dt_init: float
    dt_min: float
    dt_max: float
    drift_high: float
    drift_low: float
    shrink: float = 0.5
    grow: float = 1.2
    grow_after: int = 5

    def __post_init__(self):
        if not 0.0 < self.shrink < 1.0:
            raise ValueError("shrink must be in (0, 1)")
        if self.grow <= 1.0:
            raise ValueError("grow must be > 1")
        if self.dt_min <= 0 or self.dt_max < self.dt_min:
            raise ValueError("require 0 < dt_min <= dt_max")
        if self.drift_low >= self.drift_high:
            raise ValueError("require drift_low < drift_high")
        self.dt = max(self.dt_min, min(self.dt_max, self.dt_init))
        self._calm_streak = 0
        self._n_shrinks = 0
        self._n_grows = 0

    def observe(self, drift: float) -> float:
        """Record one step's drift and return the dt to use next."""
        if drift > self.drift_high:
            new_dt = max(self.dt_min, self.dt * self.shrink)
            if new_dt < self.dt:
                self._n_shrinks += 1
            self.dt = new_dt
            self._calm_streak = 0
        elif drift < self.drift_low:
            self._calm_streak += 1
            if self._calm_streak >= self.grow_after:
                new_dt = min(self.dt_max, self.dt * self.grow)
                if new_dt > self.dt:
                    self._n_grows += 1
                self.dt = new_dt
                self._calm_streak = 0
        else:
            self._calm_streak = 0
        return self.dt

    @property
    def stats(self) -> dict:
        return {
            "dt": self.dt,
            "shrinks": self._n_shrinks,
            "grows": self._n_grows,
            "calm_streak": self._calm_streak,
        }
