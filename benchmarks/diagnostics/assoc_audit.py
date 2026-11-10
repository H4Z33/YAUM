"""Audit diagnostics on assoc (benchmark settings: emb 16, lr 3e-3, batch 64, 1500 steps).
Usage: python -m benchmarks.diagnostics.assoc_audit MODEL SEED STEPS
1. Transformer baseline (never benchmarked). 2. Accuracy by which pair is queried."""
import sys

import torch

torch.set_num_threads(1)
from types import SimpleNamespace

from benchmarks.context_dynamics import (
    N_SYMBOLS,
    TASKS,
    Wrapped,
    build,
    masked_loss_acc,
)
from yaum.models.transformer import TransformerCharModel

args = SimpleNamespace(embedding_dim=16, hidden_dim=64, particle_dim=32, substeps=2, charge_dim=16,
                       dt=0.5, gamma_init=0.05, lstm_forget_bias=3.0)
name, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed + 1000)
if name == "transformer":
    model = Wrapped(TransformerCharModel(N_SYMBOLS, 16, 64, 2, n_heads=2, max_context=64, dropout=0.0), 16)
else:
    model = build(name, args)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
make = TASKS["assoc"]
for _ in range(steps):
    x, y, mask = make(64, 40, gen)
    loss, _ = masked_loss_acc(model(x), y, mask)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
model.eval()
with torch.no_grad():
    x, y, mask = make(4096, 40, torch.Generator().manual_seed(10_000 + seed))
    pred = model(x)[:, -1].argmax(-1)
    ok = (pred == y[:, -1]).float()
    which = (x[:, 0:-1:2] == x[:, -1:]).float().argmax(1)
    by = [round(float(ok[which == k].mean()), 3) for k in range(4)]
print(name, "seed", seed, "steps", steps, "acc", round(float(ok.mean()), 3), "by queried pair (oldest→newest)", by,
      "params", sum(p.numel() for p in model.parameters()), flush=True)
