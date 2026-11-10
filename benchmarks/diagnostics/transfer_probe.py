"""Diagnostic: does a readout fitted at T = 40 on the conserved velocities transfer to long lengths?

``cyclic_probe`` re-fits a readout at every length (and re-centres the
features), which hides two different failure modes of a *fixed* readout:

  drift   distractor impulses move the conserved momenta inside the
          directions that discriminate first tokens
  norm    the momenta only grow in other directions, and the model's own
          readout fails because its LayerNorm rescales everything

A linear readout fitted once at T = 40 on the raw cyclic velocities and
applied unchanged at long lengths separates them: if it still works, the
discriminating directions are clean and the model's readout (LayerNorm)
is the problem; if it works only after subtracting the per-length mean,
the damage is a deterministic drift; if neither, it is diffusion.

Usage: python -m benchmarks.diagnostics.transfer_probe MODEL SEED [WRITE_COST]
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from benchmarks.context_dynamics import N_SYMBOLS, TASKS, build, masked_loss_acc

torch.set_num_threads(1)
name, seed = sys.argv[1], int(sys.argv[2])
cost = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
args = SimpleNamespace(embedding_dim=16, hidden_dim=64, particle_dim=32, substeps=2, charge_dim=16,
                       dt=0.5, gamma_init=0.05, lstm_forget_bias=3.0, n_cyclic=16)
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed + 1000)
model = build(name, args)
core = model.core
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
make = TASKS["recall_first"]
for _ in range(1500):
    length = int(torch.randint(20, 41, (1,), generator=gen))
    x, y, mask = make(64, length, gen)
    loss, _ = masked_loss_acc(model(x), y, mask)
    if cost:
        loss = loss + cost * core.last_cyclic_write
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
model.eval()
nc = core.n_cyclic


def velocities(n, length, g):
    vs, labels, preds = [], [], []
    with torch.no_grad():
        for _ in range(n // 512):
            x, y, _ = make(512, length, g)
            logits, (_, p) = core(model.embed(x), core.init_hidden(512))
            vs.append((p / core.mass)[:, -nc:].double())
            labels.append(y[:, -1])
            preds.append(logits[:, -1].argmax(-1))
    return torch.cat(vs), torch.cat(labels), torch.cat(preds)


v40, y40, _ = velocities(4096, 40, torch.Generator().manual_seed(70_000 + seed))
mu40 = v40.mean(0)
a = torch.cat([v40 - mu40, torch.ones(len(v40), 1, dtype=torch.double)], 1)
w = torch.linalg.solve(a.T @ a + 1e-3 * torch.eye(nc + 1, dtype=torch.double),
                       a.T @ torch.nn.functional.one_hot(y40, N_SYMBOLS).double())
# Directions that discriminate first tokens (row space of the class weights, minus their mean).
disc = torch.linalg.svd((w[:-1] - w[:-1].mean(1, keepdim=True)).T, full_matrices=False).Vh
print(f"{name} seed={seed} write_cost={cost}")
for length in (40, 100, 400, 1000):
    v, y, pred = velocities(2048, length, torch.Generator().manual_seed(71_000 + seed + length))
    def acc(feat, labels=y):
        ones = torch.ones(len(feat), 1, dtype=torch.double)
        return float(((torch.cat([feat, ones], 1) @ w).argmax(-1) == labels).float().mean())

    shift = v.mean(0) - mu40
    inside = float((disc @ shift).norm() / (shift.norm() + 1e-12))
    print(f"  T={length:5d} own={float((pred == y).float().mean()):.3f} fixed probe={acc(v - mu40):.3f} "
          f"fixed probe, re-centred={acc(v - v.mean(0)):.3f} | mean shift {float(shift.norm()):.2f} "
          f"(fraction inside discriminating directions {inside:.2f}), spread x{float(v.std(0).mean() / v40.std(0).mean()):.2f}",
          flush=True)
