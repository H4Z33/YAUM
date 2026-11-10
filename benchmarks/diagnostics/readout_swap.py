"""Diagnostic: which part of the state breaks the trained readout at long lengths?

The cyclic models keep the first token perfectly in their conserved
momenta at T = 1000, yet their own readout fails by T = 400, and a
charge gating distractor impulses did not help (hypothesis 7). This
script splits the readout's input into the conserved part (cyclic
velocities) and the rest (ordinary positions and velocities) and mixes
them between a long run and its own 40-token prefix (same first token):

  long/long    the model as it is at the long length
  short/short  the model at T = 40
  cons=long    conserved part from the long run, the rest from T = 40
  rest=long    conserved part from T = 40, the rest from the long run

If ``cons=long`` stays high and ``rest=long`` collapses, the conserved
memory is fine and the ordinary modes are what derail the readout.

Usage: python -m benchmarks.diagnostics.readout_swap MODEL SEED [TRAIN_LEN_MAX] [CHARGE_COST]
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from benchmarks.context_dynamics import N_SYMBOLS, TASKS, build, masked_loss_acc

torch.set_num_threads(1)
name, seed = sys.argv[1], int(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
charge_cost = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
args = SimpleNamespace(embedding_dim=16, hidden_dim=64, particle_dim=32, substeps=2, charge_dim=16,
                       dt=0.5, gamma_init=0.05, lstm_forget_bias=3.0, n_cyclic=16)
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed + 1000)
model = build(name, args)
core = model.core
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
make = TASKS["recall_first"]
for _ in range(1500):
    length = int(torch.randint(20, top + 1, (1,), generator=gen))
    x, y, mask = make(64, length, gen)
    loss, _ = masked_loss_acc(model(x), y, mask)
    if charge_cost:
        loss = loss + charge_cost * core.last_charge_mean
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
model.eval()
nc = core.n_cyclic
active = core._active

with torch.no_grad():
    emb = model.embed.weight
    kick = core.impulse(emb)[:, -nc:]
    if core.cyclic_charge is not None:
        raw = core.cyclic_charge(emb).squeeze(-1)
        g = torch.relu(raw) if core.charge_kind == "neutral" else torch.sigmoid(raw)
        kick = kick * g.unsqueeze(-1)
        print(f"{name} seed={seed} charge: first tokens {[round(float(v), 3) for v in g[:N_SYMBOLS]]}")
        print(f"{'':{len(name)}}        charge: distractors  {[round(float(v), 3) for v in g[N_SYMBOLS:]]}")
    print(f"{name} seed={seed} effective cyclic impulse RMS: first {float(kick[:N_SYMBOLS].pow(2).mean().sqrt()):.4f} "
          f"distractor {float(kick[N_SYMBOLS:].pow(2).mean().sqrt()):.4f} "
          f"drift/token {float(kick[N_SYMBOLS:].mean(0).pow(2).mean().sqrt()):.4f}")

    def final_features(x):
        _, (q, p) = core(model.embed(x), core.init_hidden(x.shape[0]))
        return torch.cat([q * active, p / core.mass], dim=-1)

    def accuracy(feat, y):
        return float((core.fc(core.readout_norm(feat)).argmax(-1) == y).float().mean())

    h = core.hidden_dim
    cons = torch.zeros(2 * h, dtype=torch.bool)
    cons[-nc:] = True  # cyclic velocities; cyclic positions are already masked out
    for length in (100, 400, 1000):
        x, y, _ = make(1024, length, torch.Generator().manual_seed(60_000 + seed + length))
        target = y[:, -1]
        long, short = final_features(x), final_features(x[:, :40])
        mix_a = torch.where(cons, long, short)
        mix_b = torch.where(cons, short, long)
        print(f"  T={length:5d} long/long={accuracy(long, target):.3f} short/short={accuracy(short, target):.3f} "
              f"cons=long={accuracy(mix_a, target):.3f} rest=long={accuracy(mix_b, target):.3f} "
              f"| rest RMS long/short {float(long[:, ~cons].pow(2).mean().sqrt() / short[:, ~cons].pow(2).mean().sqrt()):.2f} "
              f"cons RMS long/short {float(long[:, cons].pow(2).mean().sqrt() / short[:, cons].pow(2).mean().sqrt()):.2f}",
              flush=True)
