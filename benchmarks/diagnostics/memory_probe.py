"""Diagnostic: is the memory gone at unseen lengths, or only unreadable?

Background: trained on ``recall_first`` at T = 40, the Hamiltonian models'
accuracy *oscillates* with evaluation length (1.00 at T = 30, 40, 50;
0.000, below chance, at T = 36 and 44). That suggests the first token is
stored in the phase of an oscillation, and the trained readout only fits
the phase at T = 40.

Prediction written before the first run of this script (2026-11-10): with
the trained dynamics frozen, a *new linear readout* fitted at each length
recovers the first token with accuracy ≥ 0.9 at T = 400 for
``ham_conservative``; for ``ham_learned`` (friction ~0.035 per token) it
recovers less at T = 400 than at T = 100. If the probe also fails for
``ham_conservative``, the memory really is lost (chaos or heating).

For each length the script reports:
  own      accuracy of the model's own trained readout
  probe    accuracy of a ridge-regression readout fitted on 4096 fresh
           sequences of that length and tested on 2048 others
  consist  how deterministic the model's own answer is given the true
           token (1.0 = each true token always maps to one answer, even if
           it is the wrong one; 1/8 = unrelated)

Usage: python -m benchmarks.diagnostics.memory_probe MODEL SEED [LENGTHS...]
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from benchmarks.context_dynamics import N_SYMBOLS, TASKS, build, masked_loss_acc

torch.set_num_threads(1)
name, seed = sys.argv[1], int(sys.argv[2])
lengths = [int(v) for v in sys.argv[3:]] or [40, 44, 100, 200, 400, 1000]
args = SimpleNamespace(embedding_dim=16, hidden_dim=64, particle_dim=32, substeps=2, charge_dim=16,
                       dt=0.5, gamma_init=0.05, lstm_forget_bias=3.0)
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed + 1000)
model = build(name, args)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
make = TASKS["recall_first"]
for _ in range(1500):
    x, y, mask = make(64, 40, gen)
    loss, _ = masked_loss_acc(model(x), y, mask)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
model.eval()

captured = {}
model.core.fc.register_forward_pre_hook(lambda _m, inp: captured.__setitem__("f", inp[0][:, -1]))


def features(n, length, g):
    feats, labels, preds = [], [], []
    with torch.no_grad():
        for _ in range(n // 512):
            x, y, _ = make(512, length, g)
            logits = model(x)
            feats.append(captured["f"].double())
            labels.append(y[:, -1])
            preds.append(logits[:, -1].argmax(-1))
    return torch.cat(feats), torch.cat(labels), torch.cat(preds)


for length in lengths:
    g = torch.Generator().manual_seed(40_000 + seed + length)
    f_tr, y_tr, _ = features(4096, length, g)
    f_te, y_te, pred = features(2048, length, g)
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-8
    a_tr = torch.cat([(f_tr - mu) / sd, torch.ones(len(f_tr), 1, dtype=torch.double)], 1)
    a_te = torch.cat([(f_te - mu) / sd, torch.ones(len(f_te), 1, dtype=torch.double)], 1)
    onehot = torch.nn.functional.one_hot(y_tr, N_SYMBOLS).double()
    w = torch.linalg.solve(a_tr.T @ a_tr + 1e-3 * torch.eye(a_tr.shape[1], dtype=torch.double), a_tr.T @ onehot)
    probe = float(((a_te @ w).argmax(-1) == y_te).float().mean())
    own = float((pred == y_te).float().mean())
    consist = sum(
        float(torch.bincount(pred[y_te == k], minlength=N_SYMBOLS).max()) for k in range(N_SYMBOLS)
    ) / len(y_te)
    print(f"{name:17s} seed={seed} T={length:5d} own={own:.3f} probe={probe:.3f} consist={consist:.3f}",
          flush=True)
