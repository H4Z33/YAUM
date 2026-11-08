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

        self.impulse = nn.Linear(embedding_dim, hidden_dim, bias=False)

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

        self.readout_norm = nn.LayerNorm(2 * hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    # ---- physical quantities ---------------------------------------------

    @property
    def mass(self) -> torch.Tensor:
        return F.softplus(self._mass) + 1e-3

    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self._kappa)

    @property
    def gamma(self) -> torch.Tensor:
        if self._rho is not None:
            return F.softplus(self._rho)
        return torch.full(
            (self.hidden_dim,), self._fixed_gamma, device=self.W.device, dtype=self.W.dtype
        )

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        z = q @ self.W.t() + self.b
        # log cosh(z) = z + softplus(-2z) - log 2, numerically stable
        logcosh = z + F.softplus(-2.0 * z) - math.log(2.0)
        return (logcosh * self.a).sum(-1) + 0.5 * (self.kappa * q * q).sum(-1)

    def grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        z = q @ self.W.t() + self.b
        return (torch.tanh(z) * self.a) @ self.W + self.kappa * q

    def energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Total energy ``H(q, p)`` per batch element."""
        return 0.5 * (p * p / self.mass).sum(-1) + self.potential(q)

    # ---- dynamics -------------------------------------------------------

    def flow(self, q, p, n_steps: int, dt: float | None = None):
        """Advance ``(q, p)`` by ``n_steps`` damped-leapfrog steps (no input)."""
        dt = self.dt if dt is None else dt
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

    def forward(self, embedded: torch.Tensor, hidden=None):
        """``embedded``: (B, T, embedding_dim) → logits (B, T, vocab), (q, p)."""
        batch, seq_len, _ = embedded.shape
        if hidden is None:
            hidden = self.init_hidden(batch)
        q, p = hidden
        kicks = self.impulse(embedded)  # (B, T, h)
        outputs = []
        for t in range(seq_len):
            p = p + kicks[:, t]
            q, p = self.flow(q, p, self.n_substeps)
            outputs.append(torch.cat([q, p / self.mass], dim=-1))
        states = torch.stack(outputs, dim=1)
        logits = self.fc(self.readout_norm(states))
        return logits, (q, p)

    def init_hidden(self, batch_size: int):
        ref = self.W
        zeros = ref.new_zeros(batch_size, self.hidden_dim)
        return (zeros, zeros.clone())
