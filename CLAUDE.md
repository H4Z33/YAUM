# YAUM — guide for agents and contributors

Read this first, then `docs/LESSONS.md` (what went wrong before and what is
still missing) and `docs/ARTICLE.md` (the research narrative and current
results). This file is loaded automatically by Claude Code; `AGENTS.md`
points here for other tools.

## What this project is

A research codebase testing one hypothesis: **Hamiltonian mechanics
(energy, momentum, conservation, controlled dissipation) is a good
substrate — a set of rules — for a model to grow in.** It is
exploratory research, not a product. Correctness of *claims* matters
more than features.

The physics lives in two places, and keeping them apart is essential:

| | training time | context time |
|---|---|---|
| clock | optimiser steps | tokens read |
| what moves | embedding table `E` (with momentum `P`, mass `m`) | hidden state `(q, p)` |
| code | `yaum/core/` (trainer, integrators) | `yaum/models/hamiltonian.py` |
| status | works; ties a tuned Adam on `E` | new; under evaluation |

The owner's thesis is mostly about **context time** ("a model builds a
small universe that moves with context as time"). Training-time physics
is a well-behaved optimiser, not a new kind of model.

## Layout

```
yaum/core/integrators.py   leapfrog, yoshida4, langevin (OBABO; T=0 = damping), multiscale
yaum/core/dynamics.py      make_force_fn (loss → force on E), kinetic_energy, total_hamiltonian
yaum/core/substrate.py     HamiltonianSubstrate: explicit (E, P, m, dt) state, propose/commit
yaum/core/trainer.py       Trainer: the training loop, telemetry, checkpoints
yaum/core/geometry.py      RMS-Fisher mass estimator (sparse EMA of force²)
yaum/core/adaptive.py      dt controller driven by one-step energy error
yaum/core/diagnostics.py   reversibility probe
yaum/core/observables.py   Var(H), Var(|mean E|), τ_int, entropy rate (exploratory)
yaum/core/action.py        S = Σ (T − V) dt (bookkeeping only)
yaum/core/integrity.py     checkpoint corruption check (NOT security)
yaum/core/replica.py       parallel tempering primitives (tested, NOT wired into trainer)
yaum/models/               rnn, transformer, hamiltonian — shared interface, registry in __init__
yaum/ui/app.py             Gradio UI (global APP_STATE, background training thread)
benchmarks/compare_training.py   training-time comparison → results.md
benchmarks/context_dynamics.py   context-time comparison → context_dynamics_results.md
docs/ARTICLE.md            narrative + results + corrections table
docs/LESSONS.md            lessons learned, open problems, backlog
```

## Commands

```bash
# environment (CPU; use requirements.txt instead on an NVIDIA machine)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt

python -m pytest -q                     # ~160 tests, ~15 s on CPU
ruff check .                            # only BLE001 (deliberate catch-alls at UI/thread boundaries) should remain
python -m benchmarks.compare_training   # ~1 min, appends to benchmarks/results.md
python -m benchmarks.context_dynamics   # ~25 min on CPU, appends to benchmarks/context_dynamics_results.md
python -m yaum.ui.app                   # UI on 127.0.0.1:7860; --share / --server-name / --server-port
```

Benchmarks **append** to their results files. Pass `--output ''` for
scratch runs so you don't pollute the record.

## Contracts (don't break these)

* **Model interface**: a model has `vocab_size`, `init_hidden(batch_size)`
  and `forward(embedded, hidden) -> (logits, hidden)`, taking *already
  embedded* input `(B, T, d)`. The embedding table is owned by the trainer
  (as `E`), not by the model. Register new models in `yaum/models/__init__.py`.
* **ForceField**: `force_fn(E, *, retain_graph) -> (loss, force)` with
  `force = −∇_E loss` scattered onto the full `(V, d)` table.
* **Integrator**: `step(E, P, dt, force_fn, mass, retain_final) -> PhaseStep`.
  `loss_final` keeps its graph when `retain_final=True` — the trainer
  backprops it into `W`. Wrapping integrators that add non-conservative
  pieces must set `PhaseStep.P_conservative_in/out` so energy error is
  measured on the conservative core only.
* **Trainer step order**: propose `(E, P)` step on a batch → measure
  one-step energy error with the *same* batch, `W` and mass → update Fisher
  mass (and rescale `P` by `m_new/m_old` to keep velocity continuous) →
  backprop `loss_final` into `W` → commit `E, P`. Moving any of these
  changes what the diagnostics mean.
* **Checkpoints** are loaded with `weights_only=True`
  (`load_checkpoint_file`). Store only tensors and plain containers. Never
  make `trusted=True` the default.

## Physics rules of thumb (each one was learned the hard way)

* During training **H is not conserved**: each step uses a new batch and
  Adam changes `W`. Any "drift vs. reference" number is mostly loss
  progress. Only same-batch, same-`W`, one-step comparisons measure the
  integrator.
* **Check magnitudes before trusting a knob.** The gradient of a mean loss
  is ~1e-4 per coordinate; its square is ~1e-7. A floor, eps or clamp of
  1e-3 silently replaces such a quantity with a constant.
* **Mass sets the scale of motion.** With `m = 1` the embeddings barely
  move; `m = sqrt(EMA[F²]) + ε` gives O(1) acceleration (RMSProp-like).
* **Undamped dynamics never settle.** Defaults are damped (Langevin with
  `temperature=0`, friction 2.0). Pure leapfrog is for experiments.
* **Time-varying mass with fixed momentum injects energy** (velocity `P/m`
  jumps). Rescale `P` when the mass changes, and keep the mass of tokens
  absent from the batch fixed (sparse EMA).
* **Pure Hamiltonian flow can't contract** (volume preservation): no
  attractors, no forgetting. Expect computation to need some dissipation;
  measure how much rather than assuming zero.

## Research hygiene (required for any claim that ends up in docs)

1. **Controls**: include the trivial baseline (e.g. `frozen_E`) so you
   know whether the mechanism does anything at all.
2. **Tuned baselines**: sweep the baseline's learning rate before saying
   YAUM beats it. An untuned Adam will flatter anything.
3. **Seeds**: ≥ 3, report mean ± std. Call differences inside seed noise a tie.
4. **Fixed evaluation set**: identical held-out windows for all variants.
5. **Pre-register** the hypothesis in the benchmark docstring *before*
   the first run, and report the result even when it goes against the
   thesis. `docs/ARTICLE.md` §6 is the corrections table — add to it
   rather than silently rewriting history.
6. **Tests check properties, not just plumbing**: e.g. "adaptive dt does
   not collapse while loss falls", "Fisher mass is not pinned to its
   floor", "frictionless flow conserves energy". A green suite of
   unit tests did not catch any of the original physics bugs.

## Environment notes

* The owner develops on **Windows with an NVIDIA GPU**. The
  `safe_speed` / `step_delay` / `cuda_sync_interval` / `disable_cudnn` /
  `cudnn_benchmark` knobs were added while chasing GPU crashes that turned
  out to be a **driver issue on the owner's PC (since fixed)** — not a
  YAUM bug. Don't attribute such crashes to the code, and don't add more
  throttling for them; the existing knobs may be relaxed (see backlog).
* `requirements.txt` pins CUDA 12.4 PyTorch; `requirements-cpu.txt` is
  for CPU/macOS; both include `requirements-common.txt`. `run.sh` picks
  automatically; the Windows scripts (`run.bat`, `run.ps1`, `installer/*`)
  always use the CUDA file and are untested by agents (no Windows here).
* Cloud agents usually have no GPU: everything must run on CPU.

## Conventions

* Python ≥ 3.10, `from __future__ import annotations`, type hints, docstrings
  that say what a quantity *means* and what it does *not* mean.
* Keep physics modules free of UI concerns; keep `print` out of import time.
* Git: develop on the branch you are given; commit messages explain *why*;
  don't put model identifiers in commits or docs.
* License is PolyForm Noncommercial 1.0.0 everywhere (README, LICENSE,
  CONTRIBUTING must agree).
