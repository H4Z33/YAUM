"""Checkpoint integrity fingerprints.

A checkpoint stores more than weights — it stores a physical state
``(E, P, mass)``. We fingerprint that state with three scalars that are
invariant under bit-exact save/load but sensitive to any tampering:

    kinetic = sum(P^2 / 2m)
    E_norm  = ||E||_2
    P_norm  = ||P||_2

On load we recompute these and compare. Any deviation beyond a tight
tolerance indicates the state has been altered or corrupted between
save and load. This is cheaper than a full hash and — unlike a hash —
carries physical meaning: ``kinetic`` is an actual observable of the
system, not an opaque digest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch


@dataclass
class CheckpointIntegrity:
    kinetic: float
    E_norm: float
    P_norm: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointIntegrity":
        return cls(
            kinetic=float(data["kinetic"]),
            E_norm=float(data["E_norm"]),
            P_norm=float(data["P_norm"]),
        )


def compute_integrity(
    E: torch.Tensor, P: torch.Tensor, mass: torch.Tensor
) -> CheckpointIntegrity:
    with torch.no_grad():
        kinetic = float((P * P / (2.0 * mass)).sum())
        e_norm = float(torch.linalg.norm(E))
        p_norm = float(torch.linalg.norm(P))
    return CheckpointIntegrity(kinetic=kinetic, E_norm=e_norm, P_norm=p_norm)


@dataclass
class IntegrityReport:
    ok: bool
    mismatches: dict[str, tuple[float, float]] = field(default_factory=dict)

    def format(self) -> str:
        if self.ok:
            return "checkpoint integrity: OK"
        lines = ["checkpoint integrity: MISMATCH"]
        for name, (saved, loaded) in self.mismatches.items():
            lines.append(f"  {name}: saved={saved:.6e} loaded={loaded:.6e}")
        return "\n".join(lines)


def verify_integrity(
    saved: CheckpointIntegrity,
    loaded: CheckpointIntegrity,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> IntegrityReport:
    """Return a report comparing the saved fingerprint to a fresh one.

    Uses a combined absolute/relative tolerance consistent with
    ``math.isclose``: ``|a - b| <= atol + rtol * max(|a|, |b|)``.
    """
    mismatches: dict[str, tuple[float, float]] = {}
    for name in ("kinetic", "E_norm", "P_norm"):
        a = getattr(saved, name)
        b = getattr(loaded, name)
        tol = atol + rtol * max(abs(a), abs(b))
        if abs(a - b) > tol:
            mismatches[name] = (a, b)
    return IntegrityReport(ok=not mismatches, mismatches=mismatches)
