"""Information-geometric mass for the embedding particle system.

Treating ``mass`` as ``torch.ones(vocab, 1)`` is an unforced geometric
choice. A data-dependent alternative is to make coordinates that feel
large, persistent forces heavy and coordinates that feel small forces
light.

We keep an EMA of the **squared force** per embedding coordinate,
``v ≈ E[g²]`` (the diagonal empirical Fisher of the loss w.r.t. ``E``),
and use

    mass = sqrt(v) + eps

This is the RMSProp/Adam preconditioner expressed as an inertia: the
acceleration ``F / m ≈ g / sqrt(E[g²])`` is O(1) whatever the raw
gradient scale, so ``dt`` has the same meaning across batch sizes,
context lengths and vocabularies. (Using ``mass = v`` instead gives an
acceleration ~ 1/g that blows up in flat directions — and because the
raw ``v`` of a mean loss is ~1e-7, any fixed floor such as 1e-3 would
swallow it entirely.)

Two details keep the time-varying mass from injecting energy:

* the EMA is **sparse** — only rows that appeared in the batch are
  updated, so the mass of an unseen token never decays while it still
  carries momentum (a shrinking mass at fixed ``P`` would speed it up);
* a row's first observation initialises ``v`` directly instead of being
  blended with zero, which removes the EMA start-up bias.

A changing mass still makes the dynamics only piecewise-Hamiltonian:
each step is symplectic for the mass it was taken with.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FisherMassConfig:
    beta: float = 0.9
    #: Absolute floor added to ``sqrt(v)``; the analogue of Adam's epsilon.
    eps: float = 1e-8
    #: Kept for config compatibility; the trainer now updates every step.
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
    """Sparse EMA of the squared force, exposed as an RMS mass.

    ``second_moment`` holds ``v``; :meth:`current` returns
    ``sqrt(v) + eps``. Before any observation every coordinate has
    ``v = 0`` and therefore mass ``eps``; the trainer warms the estimator
    up on a few batches before the first step.
    """

    def __init__(self, shape: tuple[int, int], config: FisherMassConfig):
        self.shape = tuple(shape)
        self.config = config
        self.second_moment: torch.Tensor | None = None
        self.mass: torch.Tensor | None = None
        self._update_count = 0

    def initialise(self, device, dtype) -> torch.Tensor:
        self.second_moment = torch.zeros(self.shape, device=device, dtype=dtype)
        self._refresh_mass()
        return self.mass

    def load_mass(self, mass: torch.Tensor) -> None:
        """Restore state from a saved mass tensor (inverts :meth:`current`)."""
        if tuple(mass.shape) != self.shape:
            raise ValueError(
                f"mass shape {tuple(mass.shape)} != expected {self.shape}"
            )
        root = torch.clamp(mass.detach() - self.config.eps, min=0.0)
        self.second_moment = root * root
        self._refresh_mass()
        self._update_count = max(self._update_count, 1)

    def _refresh_mass(self) -> None:
        self.mass = torch.sqrt(self.second_moment) + self.config.eps

    def current(self) -> torch.Tensor:
        if self.mass is None:
            raise RuntimeError("FisherMassEstimator not initialised")
        return self.mass

    def update(
        self, sample: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Blend ``sample`` (a squared force) into ``v``; return the new mass.

        ``mask`` selects the coordinates that were observed. By default a
        coordinate counts as observed when its sample is non-zero, which
        for a scattered embedding force means "the token was in the batch".
        """
        if tuple(sample.shape) != self.shape:
            raise ValueError(
                f"Fisher sample shape {tuple(sample.shape)} != expected {self.shape}"
            )
        if self.second_moment is None:
            self.initialise(sample.device, sample.dtype)
        sample = sample.detach()
        if mask is None:
            mask = sample > 0
        v = self.second_moment
        beta = self.config.beta
        blended = beta * v + (1.0 - beta) * sample
        first_seen = v == 0
        new_v = torch.where(first_seen, sample, blended)
        v.copy_(torch.where(mask, new_v, v))
        self._refresh_mass()
        self._update_count += 1
        return self.mass

    @property
    def update_count(self) -> int:
        return self._update_count
