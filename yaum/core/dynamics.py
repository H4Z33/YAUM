"""Force fields and Hamiltonian observables for embedding dynamics.

This module is the bridge between the neural network and the integrators in
``yaum.core.integrators``. It exposes:

* ``calculate_loss_and_grads_rnn`` — forward/backward through an RNN to
  extract the gradient of the loss w.r.t. the input embedding tensor.
* ``make_rnn_force_fn`` — turn that gradient into a closure usable by any
  symplectic integrator.
* ``kinetic_energy`` / ``total_hamiltonian`` — observables that let the
  trainer monitor how well the integrator preserves the energy of the
  embedding system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .integrators import ForceField, LeapfrogIntegrator

try:
    from .utils import device
except ImportError:  # pragma: no cover - device fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


def calculate_loss_and_grads_rnn(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    E_matrix: torch.Tensor,
    model,
    criterion,
    retain_graph: bool = False,
):
    """Forward the RNN on embedded context and grab d(loss)/d(batch_embeddings).

    Returns ``(loss, embedding_grads, batch_embeddings)``. ``embedding_grads``
    is ``None`` only if autograd raises — the caller decides how to handle it.
    """
    batch_size, _ = context_indices.shape
    hidden = model.init_hidden(batch_size)

    batch_embeddings = F.embedding(context_indices, E_matrix)
    logits, _ = model(batch_embeddings, hidden)
    loss = criterion(logits.view(-1, model.vocab_size), target_indices.view(-1))

    try:
        (embedding_grads,) = torch.autograd.grad(
            outputs=loss,
            inputs=batch_embeddings,
            grad_outputs=torch.ones_like(loss),
            retain_graph=retain_graph,
        )
    except RuntimeError as err:
        print(f"autograd.grad failed: {err}")
        return loss, None, batch_embeddings

    return loss, embedding_grads, batch_embeddings


def make_rnn_force_fn(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    model,
    criterion,
) -> ForceField:
    """Build a :class:`ForceField` that updates only the active tokens.

    The returned closure scatters the per-token-position gradient back into
    a full ``(vocab_size, embedding_dim)`` force tensor. Token positions not
    seen in this batch feel zero force, which is exactly what we want: they
    stay on whatever trajectory they were on.
    """
    indices_flat = context_indices.reshape(-1)

    def force_fn(E: torch.Tensor, *, retain_graph: bool):
        E_in = E if E.requires_grad else E.detach().requires_grad_(True)
        loss, grad_batch, _ = calculate_loss_and_grads_rnn(
            context_indices,
            target_indices,
            E_in,
            model,
            criterion,
            retain_graph=retain_graph,
        )
        if grad_batch is None:
            raise RuntimeError("Gradient computation failed during force evaluation.")
        force = torch.zeros_like(E_in)
        force.index_add_(0, indices_flat, -grad_batch.reshape(-1, E_in.shape[1]))
        return loss, force

    return force_fn


@dataclass
class EnergyReport:
    kinetic: float
    potential: float
    total: float

    def drift(self, reference: "EnergyReport") -> float:
        ref = abs(reference.total)
        if ref < 1e-12:
            return abs(self.total - reference.total)
        return abs(self.total - reference.total) / ref


def kinetic_energy(P: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """T = sum(P^2 / (2 m)). ``mass`` broadcasts against ``P``."""
    return (P * P / (2.0 * mass)).sum()


def total_hamiltonian(
    P: torch.Tensor, mass: torch.Tensor, potential: torch.Tensor
) -> EnergyReport:
    """Assemble (T, V, H) with V taken from the current loss value.

    Treating the loss as potential energy is the defining move of this
    project: the model's error field *is* the landscape the embeddings
    roll through.
    """
    T = float(kinetic_energy(P, mass).detach())
    V = float(potential.detach()) if torch.is_tensor(potential) else float(potential)
    return EnergyReport(kinetic=T, potential=V, total=T + V)


def hamiltonian_step_rnn(
    context_indices,
    target_indices,
    E,
    P,
    model,
    criterion,
    mass_vector,
    dt,
):
    """Backward-compatible leapfrog step. New code should use integrators directly."""
    force_fn = make_rnn_force_fn(context_indices, target_indices, model, criterion)
    E_leaf = E if E.requires_grad else E.detach().requires_grad_(True)
    step = LeapfrogIntegrator().step(
        E_leaf, P, dt, force_fn, mass_vector, retain_final=True
    )
    return step.E, step.P, step.loss_initial, step.loss_final
