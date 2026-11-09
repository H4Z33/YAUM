"""A many-body universe: every token is a particle that stays and interacts.

``HamiltonianCharModel`` compresses the whole context into the motion of
*one* phase point. Here each token, once read, becomes its own particle
with a position ``qᵢ``, a momentum ``pᵢ`` and a fixed **charge** ``cᵢ``
(computed from its content, like an electric charge: an attribute, not a
coordinate). All particles that have entered so far keep moving and
interact through one Hamiltonian:

    H = Σᵢ |pᵢ|² / 2m  +  ½ κ Σᵢ |qᵢ|²  +  Σ_{i<j} φᵢⱼ
    φᵢⱼ = −(cᵢ · cⱼ) · exp(−|qᵢ − qⱼ|² / 2σ²)

* ``cᵢ · cⱼ > 0`` is **attraction** (a well between the two particles),
  ``< 0`` is **repulsion**. Which pairs bind is learned through the
  charge map, so clusters (for example a key bound to its value) can form
  as composite objects that nobody designs: "particles forming
  particles".
* The pair force is symmetric, ``Fᵢⱼ = −Fⱼᵢ`` (Newton's third law). With
  ``κ = 0`` and no friction, total momentum is conserved; with any
  ``κ`` and no friction, total energy is.
* Charges are unit directions times one learned strength ``s``, so
  ``|cᵢ·cⱼ| ≤ s²``, and the range has a floor ``σ ≥ sigma_min``. With the
  bounded Gaussian kernel, every pair's energy and force are bounded (no
  gravitational collapse, no needle-sharp close encounters), and
  ``κ > 0`` keeps particles from escaping. The first draft had neither
  bound. Gradients exploded to 1e9–1e13 on 40-token sequences, the
  classic chaos of close N-body encounters.
* Friction ``γ`` (learned per dimension by default) lets clusters settle.
  Earlier results say some dissipation is needed for binding.

Reading token ``t``: a new particle is created at ``q = A xₜ`` with
``p = B xₜ`` and charge ``c = C xₜ``. Then the whole system (all particles
so far) evolves for ``n_substeps`` damped-leapfrog steps. The prediction
is read from the **newest** particle, ``(q, p/m)`` plus the pair force
it currently feels, after it has felt the pull of everything already in
the system. (Without the force term, two short substeps barely move the
newest particle and the readout mostly sees its own starting point; the
first draft learned nothing on ``assoc`` in 100 steps.) Its momentum after a short time
is a distance- and charge-weighted sum of the directions to the other
particles, a conservative, second-order cousin of attention.

There is no positional encoding: the only clock is the dynamics itself
(older particles have moved and settled more). That is a deliberate,
testable limitation.

Cost: O(N²) per force evaluation for N particles, so O(T³) per sequence.
Fine for the short synthetic tasks, not for long text.

``hidden`` is ``(q, p, c)`` for the particles already in the system,
each shaped ``(B, N, ·)``. ``init_hidden`` returns an empty universe
(N = 0). Passing a non-empty one continues an existing universe, which is
the hook for persisting it between sessions (future work).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class HamiltonianParticleModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        *,
        charge_dim: int = 16,
        dt: float = 0.5,
        dissipation: str | float = "learned",
        gamma_init: float = 0.05,
        sigma_init: float = 1.0,
        kappa_init: float = 0.1,
        charge_scale_init: float = 1.0,
        sigma_min: float = 0.25,
        readout_force: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers (substeps per token) must be >= 1")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_substeps = n_layers
        self.charge_dim = charge_dim
        self.dt = float(dt)

        self.to_position = nn.Linear(embedding_dim, hidden_dim)
        self.to_momentum = nn.Linear(embedding_dim, hidden_dim)
        self.to_charge = nn.Linear(embedding_dim, charge_dim)

        self._mass = nn.Parameter(torch.full((hidden_dim,), _inv_softplus(1.0)))
        self._kappa = nn.Parameter(torch.tensor(_inv_softplus(kappa_init)))
        self.sigma_min = float(sigma_min)
        self._sigma = nn.Parameter(torch.tensor(_inv_softplus(max(sigma_init - sigma_min, 1e-3))))
        self._charge_scale = nn.Parameter(torch.tensor(_inv_softplus(charge_scale_init)))
        self.readout_force = bool(readout_force)

        if dissipation == "learned":
            self._rho = nn.Parameter(torch.full((hidden_dim,), _inv_softplus(gamma_init)))
            self._fixed_gamma = None
        elif dissipation == "none":
            self._rho, self._fixed_gamma = None, 0.0
        else:
            value = float(dissipation)
            if value < 0:
                raise ValueError("dissipation must be >= 0")
            self._rho, self._fixed_gamma = None, value
        self.dissipation = dissipation

        width = (3 if self.readout_force else 2) * hidden_dim
        self.readout_norm = nn.LayerNorm(width)
        self.fc = nn.Linear(width, vocab_size)

    # ---- physical quantities ---------------------------------------------

    @property
    def mass(self) -> torch.Tensor:
        return F.softplus(self._mass) + 1e-3

    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self._kappa)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self._sigma) + self.sigma_min

    @property
    def charge_scale(self) -> torch.Tensor:
        return F.softplus(self._charge_scale)

    def charges(self, embedded: torch.Tensor) -> torch.Tensor:
        """Unit-direction charges times one learned strength, so |cᵢ·cⱼ| ≤ s²."""
        return self.charge_scale * F.normalize(self.to_charge(embedded), dim=-1)

    def pair_force(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Interaction part of the force on every particle (no confinement)."""
        w = self._pair_weights(q, c)
        return (w @ q - w.sum(-1, keepdim=True) * q) / self.sigma**2

    @property
    def gamma(self) -> torch.Tensor:
        if self._rho is not None:
            return F.softplus(self._rho)
        return torch.full(
            (self.hidden_dim,), self._fixed_gamma,
            device=self._mass.device, dtype=self._mass.dtype,
        )

    def _pair_weights(self, q: torch.Tensor, c: torch.Tensor):
        """Return ``g_ij · exp(−r²/2σ²)`` with the diagonal removed, shape (B, N, N).

        Distances use ``|qᵢ|² + |qⱼ|² − 2 qᵢ·qⱼ`` so nothing of shape
        (B, N, N, d) is ever materialised.
        """
        sq = (q * q).sum(-1)
        r2 = (sq.unsqueeze(-1) + sq.unsqueeze(-2) - 2.0 * q @ q.transpose(-1, -2)).clamp(min=0.0)
        g = c @ c.transpose(-1, -2)
        w = g * torch.exp(-r2 / (2.0 * self.sigma**2))
        n = q.shape[-2]
        eye = torch.eye(n, device=q.device, dtype=torch.bool)
        return w.masked_fill(eye, 0.0)

    def forces(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``−∂H/∂q`` for every particle, shape (B, N, d)."""
        # Σⱼ wᵢⱼ (qⱼ − qᵢ) / σ²: positive w pulls i toward j.
        return self.pair_force(q, c) - self.kappa * q

    def energy(self, q: torch.Tensor, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Total energy per batch element."""
        kinetic = 0.5 * (p * p / self.mass).sum((-1, -2))
        confine = 0.5 * self.kappa * (q * q).sum((-1, -2))
        w = self._pair_weights(q, c)
        interaction = -0.5 * w.sum((-1, -2))  # each pair counted twice
        return kinetic + confine + interaction

    # ---- dynamics -------------------------------------------------------

    def flow(self, q, p, c, n_steps: int, dt: float | None = None):
        """Advance all particles by ``n_steps`` damped-leapfrog steps."""
        dt = self.dt if dt is None else dt
        m = self.mass
        gamma = self.gamma
        damp = bool((gamma > 0).any())
        decay = torch.exp(-gamma * (abs(dt) * 0.5)) if damp else None
        for _ in range(n_steps):
            if damp:
                p = p * decay
            p = p + (0.5 * dt) * self.forces(q, c)
            q = q + dt * p / m
            p = p + (0.5 * dt) * self.forces(q, c)
            if damp:
                p = p * decay
        return q, p

    def forward(self, embedded: torch.Tensor, hidden=None):
        """``embedded`` (B, T, d_emb) → logits (B, T, vocab), universe ``(q, p, c)``."""
        batch, seq_len, _ = embedded.shape
        if hidden is None:
            hidden = self.init_hidden(batch)
        q, p, c = hidden
        new_q = self.to_position(embedded)
        new_p = self.to_momentum(embedded)
        new_c = self.charges(embedded)
        outputs = []
        for t in range(seq_len):
            q = torch.cat([q, new_q[:, t : t + 1]], dim=1)
            p = torch.cat([p, new_p[:, t : t + 1]], dim=1)
            c = torch.cat([c, new_c[:, t : t + 1]], dim=1)
            q, p = self.flow(q, p, c, self.n_substeps)
            observed = [q[:, -1], p[:, -1] / self.mass]
            if self.readout_force:
                # The force the newest particle feels is a physical
                # observable of the current state: what everything already
                # in the universe is pulling it toward.
                observed.append(self.pair_force(q, c)[:, -1])
            outputs.append(torch.cat(observed, dim=-1))
        states = torch.stack(outputs, dim=1)
        logits = self.fc(self.readout_norm(states))
        return logits, (q, p, c)

    def init_hidden(self, batch_size: int):
        ref = self._mass
        return (
            ref.new_zeros(batch_size, 0, self.hidden_dim),
            ref.new_zeros(batch_size, 0, self.hidden_dim),
            ref.new_zeros(batch_size, 0, self.charge_dim),
        )
