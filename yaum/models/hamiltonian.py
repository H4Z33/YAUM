"""A sequence model whose hidden state is a Hamiltonian system in *context time*.

Every other piece of YAUM puts physics on *training* time: the embedding
table drifts across optimiser steps. This module puts it where the
"small universe that moves with context" picture says it belongs — in
the forward pass. Reading token ``t`` advances an internal phase state
``(q, p)`` by a fixed amount of time under a learned energy function.

State and energy
----------------
``q, p ∈ R^h`` (``h = hidden_dim``) with

    H(q, p) = Σ_i p_i² / (2 m_i)  +  V(q)
    V(q)    = Σ_k a_k · log cosh(w_kᵀ q + b_k)  +  ½ Σ_i κ_i q_i²

* ``a_k`` is unconstrained, so ``V`` can have many wells and barriers;
* ``κ_i = softplus(·) > 0`` confines the motion (``V → ∞`` as ``|q| → ∞``,
  because ``log cosh`` grows only linearly);
* ``∇V`` has a closed form, so the force is an exact gradient field — the
  dynamics are conservative by construction, not by a learned
  approximation, and no nested autograd is needed.

Tokens are forces
-----------------
Token ``t`` (its embedding ``x_t``) delivers an impulse ``p ← p + B x_t``,
then the system evolves for ``n_substeps`` leapfrog steps of size ``dt``.
Between tokens the motion is free Hamiltonian flow — information is
carried by the trajectory, not overwritten by a gate.

Dissipation as a measured quantity
----------------------------------
Each mode has a friction ``γ_i ≥ 0`` applied as ``p ← p · exp(−γ_i dt/2)``
around every leapfrog step (conformally symplectic integration: exact
energy bookkeeping for the conservative part, exponential decay for the
dissipative part). ``dissipation`` selects:

* ``"learned"``  — ``γ_i = softplus(ρ_i)`` trained with everything else;
* ``"none"``     — ``γ = 0``: a purely conservative model;
* a float        — fixed ``γ`` for every mode.

With ``"learned"``, the trained ``γ`` *is* an experimental result: it says
how much conservation the task wants. ``n_layers`` in the shared config
is reused as ``n_substeps``.

Dynamic geometry (``geometry="dynamic"``)
-----------------------------------------
With static geometry the landscape ``V`` is fixed after training: the state
moves through it, but nothing it reads bends it. ``geometry="dynamic"``
adds the missing half of "matter tells space how to curve; space tells
matter how to move". Each potential feature gets a deformation field
``φ_k`` with momentum ``π_k``, one copy per sequence, and everything
lives in one Hamiltonian:

    V(q; φ) = Σ a_k log cosh(z_k) + Σ φ_k tanh²(z_k) + ½ Σ κ_i q_i²
    H       = Σ p²/2m + V(q; φ) + Σ π²/2M + ½ Σ λ_k φ_k²

* the trained landscape (``φ = 0``) is the **rest shape**: training
  "jumpstarts the universe". ``λ`` is the stiffness of space and ``M`` its
  inertia;
* the force on the matter is ``−∂V/∂q``, which now includes the bend;
* the force on the geometry is ``−∂H/∂φ_k = −tanh²(z_k) − λ_k φ_k``. It
  depends on where the matter is, so the content bends the landscape.
  Because ``tanh² ≥ 0``, matter pushes ``φ`` negative and lowers the
  energy of the region it occupies: an attraction, like gravity. Nothing
  about this coupling is hand-designed; it follows from putting both in
  the same ``H``;
* the bend enters through the *bounded* feature ``tanh²``. A first
  version let ``φ`` scale the unbounded ``log cosh`` directly. Its joint
  energy had no lower bound (a deeper well pulls matter further in,
  which deepens the well): gravitational collapse, seen in tests as
  energy reaching 1e34. With ``tanh²`` the joint energy is bounded below
  by ``−Σ 1/(2λ_k)`` plus the static part;
* ``H`` is separable (kinetic depends on momenta only), so the joint
  leapfrog is exactly symplectic and, without friction, conserves the
  joint energy;
* the geometry has its own friction ``γ_Φ`` (learned or fixed alongside
  the matter friction), which sets how long a bend persists.

Every sequence starts flat, ``φ = π = 0`` (``init_hidden``): the bending is
in-context memory only. Persisting it across sessions is future work
(see ``docs/LESSONS.md``).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class HamiltonianCharModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        *,
        n_potential: int | None = None,
        dt: float = 0.5,
        dissipation: str | float = "learned",
        gamma_init: float = 0.05,
        geometry: str = "static",
        stiffness_init: float = 1.0,
        geometry_mass_init: float = 1.0,
        n_cyclic: int = 0,
        cyclic_charge: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers (substeps per token) must be >= 1")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_substeps = n_layers
        self.dt = float(dt)

        k = int(n_potential or 2 * hidden_dim)
        self.W = nn.Parameter(torch.randn(k, hidden_dim) / math.sqrt(hidden_dim))
        self.b = nn.Parameter(torch.zeros(k))
        self.a = nn.Parameter(torch.randn(k) * 0.5)
        self._kappa = nn.Parameter(torch.full((hidden_dim,), _inv_softplus(0.5)))
        self._mass = nn.Parameter(torch.full((hidden_dim,), _inv_softplus(1.0)))

        # Cyclic coordinates (Noether): the last ``n_cyclic`` coordinates do
        # not appear in V at all (no wells, no confinement) and feel no
        # friction. H is then invariant under translations along them, so
        # their momenta are *exactly* conserved between tokens: a memory
        # that neither oscillates nor decays. Their positions drift
        # linearly and are not observables, so the readout sees only their
        # velocities. Without this every mode has κ > 0 and oscillates, so
        # a stored token is a rotating phase the readout can only decode at
        # the trained length (ARTICLE §4.2).
        if not 0 <= n_cyclic < hidden_dim:
            raise ValueError("n_cyclic must be in [0, hidden_dim)")
        self.n_cyclic = int(n_cyclic)
        active = torch.ones(hidden_dim)
        if self.n_cyclic:
            active[-self.n_cyclic:] = 0.0
        self.register_buffer("_active", active, persistent=False)

        self.impulse = nn.Linear(embedding_dim, hidden_dim, bias=False)
        # Selection rule: a content-derived charge g(x) in (0, 1) scales
        # each token's impulse on the cyclic coordinates. A token with zero
        # charge cannot change the conserved momenta at all. Without it,
        # distractor tokens push the conserved memory by ~8% of a real
        # write each step, and the drift swamps it over hundreds of tokens
        # (ARTICLE §4.2). This plays the role of an LSTM input gate; the
        # difference is that the memory it guards is conserved exactly by
        # a symmetry instead of approximately by a learned forget gate.
        if cyclic_charge and not self.n_cyclic:
            raise ValueError("cyclic_charge needs n_cyclic > 0")
        self.cyclic_charge = nn.Linear(embedding_dim, 1) if cyclic_charge else None

        if dissipation == "learned":
            self._rho = nn.Parameter(
                torch.full((hidden_dim,), _inv_softplus(gamma_init))
            )
            self._fixed_gamma = None
        elif dissipation == "none":
            self._rho = None
            self._fixed_gamma = 0.0
        else:
            value = float(dissipation)
            if value < 0:
                raise ValueError("dissipation must be >= 0")
            self._rho = None
            self._fixed_gamma = value
        self.dissipation = dissipation

        if geometry not in ("static", "dynamic"):
            raise ValueError("geometry must be 'static' or 'dynamic'")
        self.geometry = geometry
        self.n_potential = k
        if geometry == "dynamic":
            self._stiffness = nn.Parameter(torch.full((k,), _inv_softplus(stiffness_init)))
            self._geom_mass = nn.Parameter(torch.full((k,), _inv_softplus(geometry_mass_init)))
            if dissipation == "learned":
                self._rho_geom = nn.Parameter(torch.full((k,), _inv_softplus(gamma_init)))
            else:
                self._rho_geom = None

        self.readout_norm = nn.LayerNorm(2 * hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    # ---- physical quantities ---------------------------------------------

    @property
    def mass(self) -> torch.Tensor:
        return F.softplus(self._mass) + 1e-3

    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self._kappa) * self._active

    @property
    def W_active(self) -> torch.Tensor:
        """``W`` with the cyclic columns zeroed, so V ignores those coordinates."""
        return self.W * self._active if self.n_cyclic else self.W

    @property
    def gamma(self) -> torch.Tensor:
        if self._rho is not None:
            return F.softplus(self._rho) * self._active
        return torch.full(
            (self.hidden_dim,), self._fixed_gamma, device=self.W.device, dtype=self.W.dtype
        ) * self._active

    @property
    def stiffness(self) -> torch.Tensor:
        return F.softplus(self._stiffness)

    @property
    def geometry_mass(self) -> torch.Tensor:
        return F.softplus(self._geom_mass) + 1e-3

    @property
    def gamma_geometry(self) -> torch.Tensor:
        if self._rho_geom is not None:
            return F.softplus(self._rho_geom)
        return torch.full(
            (self.n_potential,), self._fixed_gamma, device=self.W.device, dtype=self.W.dtype
        )

    @staticmethod
    def _logcosh(z: torch.Tensor) -> torch.Tensor:
        # log cosh(z) = z + softplus(-2z) - log 2, numerically stable
        return z + F.softplus(-2.0 * z) - math.log(2.0)

    def potential(self, q: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        """``V(q; φ)``; ``phi=None`` is the trained (flat, unbent) landscape."""
        z = q @ self.W_active.t() + self.b
        V = (self._logcosh(z) * self.a).sum(-1) + 0.5 * (self.kappa * q * q).sum(-1)
        if phi is not None:
            V = V + (torch.tanh(z).pow(2) * phi).sum(-1)
        return V

    def grad_potential(self, q: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        z = q @ self.W_active.t() + self.b
        t = torch.tanh(z)
        coeff = t * self.a
        if phi is not None:
            coeff = coeff + 2.0 * t * (1.0 - t * t) * phi
        return coeff @ self.W_active + self.kappa * q

    def geometry_force(self, q: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """``−∂H/∂φ``: matter lowers the energy where it is; stiffness flattens."""
        z = q @ self.W_active.t() + self.b
        return -torch.tanh(z).pow(2) - self.stiffness * phi

    def energy(self, q, p, phi=None, pi=None) -> torch.Tensor:
        """Total energy per batch element (joint matter + geometry when given)."""
        H = 0.5 * (p * p / self.mass).sum(-1) + self.potential(q, phi)
        if phi is not None:
            H = H + 0.5 * (pi * pi / self.geometry_mass).sum(-1)
            H = H + 0.5 * (self.stiffness * phi * phi).sum(-1)
        return H

    # ---- dynamics -------------------------------------------------------

    def flow(self, q, p, n_steps: int, dt: float | None = None, phi=None, pi=None):
        """Advance the state by ``n_steps`` damped-leapfrog steps (no input).

        Returns ``(q, p)``, or ``(q, p, phi, pi)`` when a geometry state is given.
        """
        dt = self.dt if dt is None else dt
        if phi is not None:
            return self._flow_joint(q, p, phi, pi, n_steps, dt)
        m = self.mass
        gamma = self.gamma
        decay = torch.exp(-gamma * (abs(dt) * 0.5)) if bool((gamma > 0).any()) else None
        for _ in range(n_steps):
            if decay is not None:
                p = p * decay
            p = p - (0.5 * dt) * self.grad_potential(q)
            q = q + dt * p / m
            p = p - (0.5 * dt) * self.grad_potential(q)
            if decay is not None:
                p = p * decay
        return q, p

    def _flow_joint(self, q, p, phi, pi, n_steps: int, dt: float):
        m, M = self.mass, self.geometry_mass
        g, g_phi = self.gamma, self.gamma_geometry
        damp = bool((g > 0).any()) or bool((g_phi > 0).any())
        if damp:
            decay = torch.exp(-g * (abs(dt) * 0.5))
            decay_phi = torch.exp(-g_phi * (abs(dt) * 0.5))
        h = 0.5 * dt
        for _ in range(n_steps):
            if damp:
                p, pi = p * decay, pi * decay_phi
            p = p - h * self.grad_potential(q, phi)
            pi = pi + h * self.geometry_force(q, phi)
            q = q + dt * p / m
            phi = phi + dt * pi / M
            p = p - h * self.grad_potential(q, phi)
            pi = pi + h * self.geometry_force(q, phi)
            if damp:
                p, pi = p * decay, pi * decay_phi
        return q, p, phi, pi

    def forward(self, embedded: torch.Tensor, hidden=None):
        """``embedded``: (B, T, embedding_dim) → logits (B, T, vocab), (q, p)."""
        batch, seq_len, _ = embedded.shape
        if hidden is None:
            hidden = self.init_hidden(batch)
        dynamic = self.geometry == "dynamic"
        if dynamic:
            q, p, phi, pi = hidden
        else:
            q, p = hidden
        kicks = self.impulse(embedded)  # (B, T, h)
        if self.cyclic_charge is not None:
            charge = torch.sigmoid(self.cyclic_charge(embedded))  # (B, T, 1)
            kicks = kicks * (self._active + (1.0 - self._active) * charge)
        # Mean squared impulse on the conserved coordinates in this call: a
        # training loop may add it to the loss as a "write cost", so that
        # only tokens that must be remembered pay to change the conserved
        # momenta. ``None`` when there are no cyclic coordinates.
        self.last_cyclic_write = (
            kicks[..., -self.n_cyclic:].pow(2).mean() if self.n_cyclic else None
        )
        outputs = []
        for t in range(seq_len):
            p = p + kicks[:, t]
            if dynamic:
                q, p, phi, pi = self._flow_joint(q, p, phi, pi, self.n_substeps, self.dt)
            else:
                q, p = self.flow(q, p, self.n_substeps)
            # The readout sees only the matter: the geometry acts on the
            # prediction solely by changing how the matter moves.
            outputs.append(torch.cat([q * self._active, p / self.mass], dim=-1))
        states = torch.stack(outputs, dim=1)
        logits = self.fc(self.readout_norm(states))
        return logits, ((q, p, phi, pi) if dynamic else (q, p))

    def init_hidden(self, batch_size: int):
        ref = self.W
        zeros = ref.new_zeros(batch_size, self.hidden_dim)
        if self.geometry == "dynamic":
            # Every sequence starts from the trained, unbent landscape.
            flat = ref.new_zeros(batch_size, self.n_potential)
            return (zeros, zeros.clone(), flat, flat.clone())
        return (zeros, zeros.clone())
