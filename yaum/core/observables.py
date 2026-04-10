"""Phase-transition telemetry for a Hamiltonian embedding run.

The trainer already tracks ``|ΔH/H|`` — that tells you the integrator is
stable. It does not tell you whether the system is about to reorganise,
which is what we care about if we treat training as a substrate where
understanding crystallises.

This module maintains a rolling window of (H, E_mean, S) records and
exposes the four observables that the statistical mechanics of a finite
system uses to detect phase transitions:

    specific_heat   = Var(H)                 — susceptibility to energy exchange
    susceptibility  = Var(||E_mean||)        — order-parameter fluctuations
    corr_time       = τ_int of H autocorr    — proxy for correlation length
    entropy_rate    = d/dt  0.5 log det Σ(E) — rate of information crystallisation

Near criticality the first three spike and the fourth swings. A grokking
event should be visible in this panel *before* it is visible in loss.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch


def embedding_entropy_logdet(E: torch.Tensor, eps: float = 1e-6) -> float:
    """Gaussian differential entropy of the row distribution of ``E``.

    Returns ``0.5 * log det(Σ + εI)`` where Σ is the sample covariance of
    the rows. Constant offsets (``0.5 d (1 + log 2π)``) are dropped — we
    only care about *changes* in entropy over time.
    """
    if E.ndim != 2 or E.shape[0] < 2:
        return float("nan")
    with torch.no_grad():
        centred = E - E.mean(dim=0, keepdim=True)
        n = centred.shape[0]
        cov = (centred.t() @ centred) / max(n - 1, 1)
        d = cov.shape[0]
        cov = cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
        sign, logabsdet = torch.linalg.slogdet(cov)
        if sign <= 0:
            return float("nan")
        return 0.5 * float(logabsdet)


def integrated_autocorr(series: list[float], max_lag: Optional[int] = None) -> float:
    """Integrated autocorrelation time of a scalar series.

    τ_int = 1 + 2 Σ_k ρ(k) truncated at the first non-positive ρ. For an
    iid sequence this returns ~1; for a strongly correlated one it grows
    without bound. Near a phase transition it diverges — that's the
    "critical slowing down" signal we want to plot.
    """
    n = len(series)
    if n < 4:
        return float("nan")
    x = torch.tensor(series, dtype=torch.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 1.0
    if max_lag is None:
        max_lag = max(1, n // 4)
    tau = 1.0
    for k in range(1, min(max_lag, n - 1) + 1):
        rho = float((x[:-k] * x[k:]).sum()) / denom
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return tau


@dataclass
class ObservableRecord:
    """One snapshot pushed into the rolling window."""

    H: float
    M: float        # order parameter: ||mean(E)||_2
    S: float        # log-det entropy of E's row distribution (Gaussian proxy)


@dataclass
class ObservableReport:
    specific_heat: float
    susceptibility: float
    corr_time: float
    entropy_rate: float

    def as_dict(self) -> dict:
        return {
            "specific_heat": self.specific_heat,
            "susceptibility": self.susceptibility,
            "corr_time": self.corr_time,
            "entropy_rate": self.entropy_rate,
        }


class ObservableWindow:
    """Rolling-window statistics over ``ObservableRecord`` pushes.

    ``size`` is the number of consecutive snapshots kept. ``report()``
    returns ``None`` until the window holds at least two records, so the
    caller should be prepared for a warm-up period.
    """

    def __init__(self, size: int = 32):
        if size < 2:
            raise ValueError("window size must be >= 2")
        self.size = int(size)
        self._buf: Deque[ObservableRecord] = deque(maxlen=self.size)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, record: ObservableRecord) -> None:
        self._buf.append(record)

    def report(self) -> Optional[ObservableReport]:
        if len(self._buf) < 2:
            return None
        Hs = [r.H for r in self._buf if r.H == r.H]  # drop NaN
        Ms = [r.M for r in self._buf if r.M == r.M]
        Ss = [r.S for r in self._buf if r.S == r.S]
        if len(Hs) < 2 or len(Ms) < 2:
            return None

        H_t = torch.tensor(Hs, dtype=torch.float64)
        M_t = torch.tensor(Ms, dtype=torch.float64)
        specific_heat = float(H_t.var(unbiased=False))
        susceptibility = float(M_t.var(unbiased=False))
        corr_time = integrated_autocorr(Hs)
        if len(Ss) >= 2:
            entropy_rate = (Ss[-1] - Ss[0]) / (len(Ss) - 1)
        else:
            entropy_rate = float("nan")

        return ObservableReport(
            specific_heat=specific_heat,
            susceptibility=susceptibility,
            corr_time=corr_time,
            entropy_rate=entropy_rate,
        )


def snapshot(
    E: torch.Tensor, H: float, eps: float = 1e-6
) -> ObservableRecord:
    """Compute an :class:`ObservableRecord` from the current phase state."""
    with torch.no_grad():
        centroid = E.mean(dim=0)
        M = float(torch.linalg.norm(centroid))
    S = embedding_entropy_logdet(E.detach(), eps=eps)
    return ObservableRecord(H=float(H), M=M, S=S)
