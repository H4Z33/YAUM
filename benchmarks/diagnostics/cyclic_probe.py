"""Diagnostic: why does ham_cyclic still lose recall at long lengths?

Two suspects for the conserved (cyclic) momenta p_c, trained at T = 20-40:
  (a) distractor tokens kick p_c a little each step, so it random-walks;
  (b) p_c is intact, but the readout (a LayerNorm over all features) is
      thrown off because the other, non-conserved modes heat up.
For each length this prints the model's own accuracy, a fresh linear
readout on the cyclic velocities only, the ratio of distractor to
first-token impulse size in the cyclic directions, and the RMS size of
the non-cyclic state relative to T = 40 (heating).

Usage: python -m benchmarks.diagnostics.cyclic_probe MODEL SEED
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from benchmarks.context_dynamics import N_SYMBOLS, TASKS, build, masked_loss_acc

torch.set_num_threads(1)
name, seed = sys.argv[1], int(sys.argv[2])
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
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
model.eval()
nc = core.n_cyclic
with torch.no_grad():
    kicks = core.impulse(model.embed.weight)[:, -nc:]  # (16 tokens, nc)
    first, distract = kicks[:N_SYMBOLS], kicks[N_SYMBOLS:]
    print(f"{name} seed={seed} cyclic impulse RMS: first-token {float(first.pow(2).mean().sqrt()):.4f} "
          f"distractor {float(distract.pow(2).mean().sqrt()):.4f} "
          f"distractor mean (drift per token) {float(distract.mean(0).pow(2).mean().sqrt()):.4f}")


def state(n, length, g):
    vs, rest, labels, preds = [], [], [], []
    with torch.no_grad():
        for _ in range(n // 512):
            x, y, _ = make(512, length, g)
            logits, (q, p) = core(model.embed(x), core.init_hidden(512))
            vs.append((p / core.mass)[:, -nc:].double())
            rest.append(torch.cat([q[:, :-nc], p[:, :-nc]], 1))
            labels.append(y[:, -1])
            preds.append(logits[:, -1].argmax(-1))
    return torch.cat(vs), torch.cat(rest), torch.cat(labels), torch.cat(preds)


base = None
for length in (40, 100, 200, 400, 1000):
    g = torch.Generator().manual_seed(50_000 + seed + length)
    v_tr, _, y_tr, _ = state(4096, length, g)
    v_te, rest, y_te, pred = state(2048, length, g)
    mu, sd = v_tr.mean(0), v_tr.std(0) + 1e-8
    a_tr = torch.cat([(v_tr - mu) / sd, torch.ones(len(v_tr), 1, dtype=torch.double)], 1)
    a_te = torch.cat([(v_te - mu) / sd, torch.ones(len(v_te), 1, dtype=torch.double)], 1)
    onehot = torch.nn.functional.one_hot(y_tr, N_SYMBOLS).double()
    w = torch.linalg.solve(a_tr.T @ a_tr + 1e-3 * torch.eye(nc + 1, dtype=torch.double), a_tr.T @ onehot)
    probe = float(((a_te @ w).argmax(-1) == y_te).float().mean())
    # Signal-to-noise of the conserved velocities: spread between first tokens vs. within one.
    means = torch.stack([v_te[y_te == k].mean(0) for k in range(N_SYMBOLS)])
    within = torch.stack([v_te[y_te == k].var(0) for k in range(N_SYMBOLS)]).mean().sqrt()
    snr = float(means.std(0).mean() / within)
    rms = float(rest.pow(2).mean().sqrt())
    base = base or rms
    print(f"  T={length:5d} own={float((pred == y_te).float().mean()):.3f} probe(cyclic only)={probe:.3f} "
          f"cyclic SNR={snr:.2f} non-cyclic RMS x{rms / base:.2f}", flush=True)
