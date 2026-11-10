"""Diagnostic (not a benchmark): Big Bang variants of the particle model.

Usage: python -m benchmarks.diagnostics.bigbang_variants TASK VARIANT SEED STEPS
Prints one JSON line: grad-norm stats (first 100 steps), mean kernel felt
by the newest particle at init and at the end, accuracy curve, final eval.
"""
import json
import sys
import time

import torch

torch.set_num_threads(1)
from benchmarks.context_dynamics import TASKS, Wrapped, masked_loss_acc
from yaum.models.particles import HamiltonianParticleModel

VARIANTS = {
    "base": {},
    "dense": {"spawn_scale_init": 0.1},
    "dense_expand": {"spawn_scale_init": 0.1, "expansion": True, "kappa_init": -0.02},
    "dense_primordial": {"spawn_scale_init": 0.1, "n_primordial": 4},
    "bigbang": {"spawn_scale_init": 0.1, "expansion": True, "kappa_init": -0.02, "n_primordial": 4},
}
task, variant, seed, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed + 1000)
core = HamiltonianParticleModel(8, 32, 32, 2, charge_dim=16, dt=0.5, dissipation="learned",
                                gamma_init=0.05, **VARIANTS[variant])
model = Wrapped(core, 32)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
make = TASKS[task]

def newest_kernel():
    """Mean kernel between the last token's birth point and earlier particles (no dynamics)."""
    g = torch.Generator().manual_seed(7)
    x, _, _ = make(64, 40, g)
    with torch.no_grad():
        q = core.spawn_scale * core.to_position(model.embed(x))
        r2 = ((q[:, -1:] - q[:, :-1]) ** 2).sum(-1)
        return float(torch.exp(-r2 / (2 * core.sigma**2)).mean())

k0 = newest_kernel()
gnorms, curve, t0 = [], [], time.perf_counter()
for step in range(steps):
    x, y, mask = make(64, 40, gen)
    loss, acc = masked_loss_acc(model(x), y, mask)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if step < 100:
        gnorms.append(gn)
    opt.step()
    if (step + 1) % 100 == 0:
        curve.append(round(float(acc), 3))
eval_gen = torch.Generator().manual_seed(10_000 + seed)
model.eval()
with torch.no_grad():
    x, y, mask = make(1024, 40, eval_gen)
    _, acc = masked_loss_acc(model(x), y, mask)
gn = torch.tensor(gnorms)
print(json.dumps({"task": task, "variant": variant, "seed": seed, "steps": steps, "acc": round(float(acc), 3),
    "gn_median": round(float(gn.median()), 3), "gn_max": round(float(gn.max()), 3),
    "kernel_init": round(k0, 4), "kernel_end": round(newest_kernel(), 4),
    "sigma": round(float(core.sigma), 3), "spawn": round(float(core.spawn_scale), 3),
    "kappa": round(float(core.kappa), 4), "gamma_med": round(float(core.gamma.median()), 3),
    "curve": curve, "wall": round(time.perf_counter() - t0)}), flush=True)
