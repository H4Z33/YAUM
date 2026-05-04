"""Information-geometric mass for the embedding particle system.

Treating ``mass`` as ``torch.ones(vocab, 1)`` is an unforced geometric
choice. The natural metric on any probability-output model is the Fisher
information: the quadratic form ``F_{ij} = E[∂ log p_i · ∂ log p_j]``
that measures how sharply the output distribution responds to a
parameter perturbation. Using it as mass turns the Hamiltonian flow on
embeddings into natural-gradient dynamics — steep directions become
heavy and move slowly, flat directions stay light and explore freely.

The full Fisher matrix on the embedding table has shape
``(V*d, V*d)`` which is unusable. We maintain a **diagonal empirical
Fisher** of shape ``(V, d)`` — one scalar per embedding coordinate —
updated by an EMA over recent minibatches. That tensor broadcasts
elementwise against ``P`` and ``E`` in every integrator step, so
swapping ``mass_mode: "fisher"`` into the config needs no integrator
changes at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class FisherMassConfig:
    beta: float = 0.9
    eps: float = 1e-3
    refresh_every: int = 50
    batches_per_refresh: int = 4

    def __post_init__(self):
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        if self.batches_per_refresh < 1:
            raise ValueError("batches_per_refresh must be >= 1")


def fisher_diagonal_sample(
    model,
    criterion,
    E: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Return ``(∂L/∂E)²`` for one minibatch, same shape as ``E``.

    This is the diagonal empirical Fisher: the expectation is replaced by
    a sample mean over one batch, and the outer product ``g gᵀ`` is
    collapsed to its diagonal. For a well-calibrated model it converges
    to the true Fisher diagonal; for an underfit one it is biased low on
    the rare tokens, which is harmless for use as an inertia floor.
    """
    E_leaf = E.detach().requires_grad_(True)
    embedded = F.embedding(context_indices, E_leaf)
    
    # cuDNN RNN backward pass requires training mode on CUDA.
    # We switch back to cuDNN for performance (5x speedup) now that VRAM is stable.
    was_training = model.training
    if not was_training:
        model.train()
    try:
        hidden = model.init_hidden(context_indices.shape[0])
        logits, _ = model(embedded, hidden)
        loss = criterion(
            logits.reshape(-1, model.vocab_size), target_indices.reshape(-1)
        )
        (grad,) = torch.autograd.grad(loss, E_leaf)
    finally:
        if not was_training:
            model.eval()
        
    return grad * grad


class FisherMassEstimator:
    """EMA-smoothed diagonal empirical Fisher on the embedding table.

    Initialised to the ``eps`` floor everywhere (so unseen tokens still
    have a finite mass). Calling :meth:`update` blends a new sample into
    the running estimate with decay ``beta`` and re-applies the floor.
    """

    def __init__(self, shape: tuple[int, int], config: FisherMassConfig):
        self.shape = tuple(shape)
        self.config = config
        self.mass: Optional[torch.Tensor] = None
        self._update_count = 0

    def initialise(self, device, dtype) -> torch.Tensor:
        self.mass = torch.full(
            self.shape, self.config.eps, device=device, dtype=dtype
        )
        return self.mass

    def current(self) -> torch.Tensor:
        if self.mass is None:
            raise RuntimeError("FisherMassEstimator not initialised")
        return self.mass

    def update(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.shape != self.shape:
            raise ValueError(
                f"Fisher sample shape {tuple(sample.shape)} != expected {self.shape}"
            )
        if self.mass is None:
            self.initialise(sample.device, sample.dtype)
        beta = self.config.beta
        self.mass.mul_(beta).add_(sample.detach(), alpha=1.0 - beta)
        self.mass.clamp_(min=self.config.eps)
        self._update_count += 1
        return self.mass

    @property
    def update_count(self) -> int:
        return self._update_count
