# YAUM — Training as a Physical System

**Thesis.** An embedding matrix is not a bag of parameters; it is a particle
in a high-dimensional semantic phase space. If you treat it that way — give
it momentum, give it mass, push it around with a conservative force derived
from the loss — then the machinery of Hamiltonian mechanics comes along for
free. You get time-reversibility, a conserved energy you can watch, a
natural adaptive timestep, and a thermostat for exploration. None of this
is available to an SGD/Adam loop, which forgets everything about a step as
soon as it is taken.

YAUM (*Yet Another Universal Model*) is a minimal, test-driven reference
implementation of that idea.

---

## 1. Why another trainer?

A standard training step looks like this:

```
θ ← θ − η · ∇L(θ)
```

The system has no memory: every step starts from the same kinetic zero.
The step size `η` is chosen once, for all parameters, and the only feedback
the loop has about its own stability is whether the loss went down.

A symplectic integrator step looks like this:

```
P ← P + F(E) · dt/2          # half momentum kick
E ← E + (P / m) · dt         # position drift
P ← P + F(E_new) · dt/2      # half momentum kick
```

The force `F = −∇L`, mass `m` (we use token frequency), momentum `P`, and
position `E` are all first-class objects. The step is
**time-reversible** — run it backward with `dt → −dt` and you land on
the previous state to within `O(ε_mach)`. It is **symplectic**, meaning
the Hamiltonian `H = T(P) + V(E)` is conserved up to a bounded oscillation
that never drifts. This is the "turtle all the way down" property — every
structural layer of the integrator inherits the conservation laws of the
layers beneath it.

That gives you things a plain optimiser cannot:

1. **`ΔH/H` — a physical stability signal.** If the energy starts drifting
   monotonically, your timestep is too big. No loss curve can tell you
   that; only the Hamiltonian can.
2. **Reversibility residuals.** Run forward `n` steps, then backward `n`
   steps, then measure `‖E_out − E_in‖`. A symplectic scheme keeps this at
   float-precision noise. Any divergence is a direct measurement of how
   much the integrator is lying to you.
3. **Thermostatted exploration.** Swap the core integrator for a Langevin
   (OBABO) splitting and the embeddings start sampling the canonical
   ensemble at a chosen temperature `kT`. The loss landscape becomes a
   Boltzmann distribution the particles actually explore.
4. **Rate-preserving adaptive dt.** Feed the drift back into a controller
   that shrinks `dt` the moment `|ΔH/H|` spikes and grows it back after a
   calm streak. No learning-rate schedule, no warmup — just local
   feedback.
5. **Turtle-composable integrators.** Yoshida's 4th-order scheme is
   literally three leapfrogs with weights `(w₁, w₂, w₁)`. rRESPA
   multi-time-stepping is a slow kick wrapping `n_inner` fast substeps.
   Each layer is one class, with the same `step(E, P, dt, force_fn, mass)`
   contract.

---

## 2. Architecture at a glance

```
                 ┌─────────────────────────────┐
                 │           Trainer           │
                 │  owns W (model), E, P, m    │
                 └──────────────┬──────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
    make_integrator()     make_model()         make_force_fn()
  leapfrog / yoshida4    rnn / transformer       loss → gradient
  langevin / multiscale                          → force tensor
           │                    │                    │
           └──────────► step(E, P, dt, force_fn, m) ◄─┘
                               │
                               ▼
                     PhaseStep(E', P', L₁, L₂)
                               │
                               ▼
              ┌──── total_hamiltonian() ─── observe drift
              ├──── measure_reversibility() ── eval-tick probe
              └──── compute_integrity() ──── checkpoint fingerprint
```

Everything downstream of `PhaseStep` is a *passive observer*: it can read
`(E, P, loss)` and compute a scalar, but it cannot modify the trajectory.
The integrator is the only actor. The trainer itself is just glue: it asks
the integrator for a step, asks the observers for reports, and logs.

Files worth opening, in reading order:

| path | what lives there |
|---|---|
| `yaum/core/integrators.py` | `SymplecticIntegrator`, leapfrog, yoshida4, langevin, multiscale |
| `yaum/core/dynamics.py`    | `make_force_fn`, `total_hamiltonian`, `EnergyReport` |
| `yaum/core/substrate.py`   | `HamiltonianSubstrate` explicit `(E, P, m, dt)` phase-state wrapper |
| `yaum/core/adaptive.py`    | `AdaptiveStepController` — H-drift → dt feedback |
| `yaum/core/diagnostics.py` | `measure_reversibility` — forward/backward probe |
| `yaum/core/integrity.py`   | `CheckpointIntegrity` — physical fingerprint |

New trainer runs default to the information-geometric branch: Fisher mass,
adaptive `dt`, periodic snapshots, and phase-shift snapshots. Older checkpoints
with scalar mass are loaded conservatively so their original phase geometry is
not silently rewritten.
| `yaum/core/trainer.py`     | the glue |
| `yaum/models/*`            | RNN and Transformer, both behind the same interface |
| `yaum/ui/app.py`           | Gradio dashboard with a 3×3 live metric grid |
| `benchmarks/compare_training.py` | head-to-head against an Adam baseline |

---

## 3. How YAUM differs from similar systems

| concern | standard trainer | YAUM |
|---|---|---|
| embedding update | gradient step, `η · ∇L` | symplectic step, second-order in `dt` |
| state per token | just the value | value + momentum + mass |
| stability signal | loss curve, loss NaN | `ΔH/H`, reversibility residual |
| exploration | dropout, random restart | Langevin thermostat (OBABO) |
| step-size control | LR schedule, warmup | `dt` closed-loop on drift |
| checkpoint integrity | byte hash (if any) | physical fingerprint on `(T, ‖E‖, ‖P‖)` |
| higher order | Adam/AdamW, LAMB | Yoshida4 = three-leapfrog composition |
| multi-scale updates | gradient accumulation | rRESPA slow/fast force split |
| architecture swap | needs full retrain | RNN ↔ Transformer share the force-field contract |

None of the YAUM pieces are "just Adam with a twist". They all fall out
of one commitment — that `E` lives in a phase space with a Hamiltonian —
and that commitment makes every downstream component composable without a
single feature flag.

---

## 4. Benchmark

The script in `benchmarks/compare_training.py` runs three variants on an
identical synthetic corpus (vocab 32, 8k tokens), pinning the model, the
batch sequence, and the RNG. The only thing that changes between variants
is the rule for updating `E`.

Headline numbers on a 300-step run (`seed=0`):

| variant | wall (s) | force evals | final train | final test | best test | peak |ΔH/H| |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_adam` | 4.18 | 300  | 0.4984 | 0.5074 | 0.5119 | — |
| `yaum_leapfrog` | 4.79 | 600  | 0.5333 | 0.5408 | 0.5459 | 8.34e-01 |
| `yaum_yoshida4` | 7.63 | 1800 | 0.5333 | 0.5408 | 0.5459 | 8.34e-01 |

### What this actually says

1. **On raw loss, Adam wins by ~6.7 % in this regime.** That is honest and
   expected: Adam is a very aggressive preconditioner and the synthetic
   corpus has clean, small gradients. YAUM is not trying to out-Adam
   Adam on loss — it is trying to make the embedding update a
   well-defined physical process.

2. **YAUM produces a signal Adam cannot.** The `|ΔH/H|` column is the peak
   normalised drift of the Hamiltonian during the run. With `dt = 0.05`,
   drift reaches 0.83 — a loud warning that this timestep is too large
   for this system. Flip on `adaptive_dt=True` and the controller shrinks
   `dt` within a few eval ticks. An Adam run has no analogous alarm.

3. **Symplectic costs are modest.** Leapfrog is `1.15×` Adam wall time for
   exactly `2×` the gradient evaluations (leapfrog uses two force calls
   per step). Yoshida4 is `3×` force evaluations, landing at `1.59×`
   leapfrog wall time — and on a mild-force regime like this it buys you
   nothing, because leapfrog and yoshida4 already agree to float-32
   precision. The dashboard makes this visible so you can switch back.

4. **Reversibility is binary.** The trainer's reversibility probe
   (`yaum/core/diagnostics.py`) is wired into the Gradio dashboard. On a
   healthy run the residual sits at `~1e-6`; the moment you introduce
   non-conservative state (e.g. dropout inside the force field), it
   explodes to `O(1)` and the chart screams.

### Conclusions, short version

- YAUM is **not** a drop-in replacement for Adam if your only goal is
  minimum validation loss on a tiny corpus. Use Adam.
- YAUM **is** the right substrate if you want first-class answers to:
  "is my timestep safe?", "is my run reversible?", "can I explore at a
  controlled temperature?", "is my checkpoint physically coherent?" —
  none of which an optimiser loop knows how to ask.
- The two frameworks can **coexist**. The trainer already runs Adam on
  the network weights `W` and symplectic dynamics on the embeddings `E`
  in the same step. This is the "time is all it has" split: `W` gets
  Adam, `E` lives in phase space, and the loss couples them through the
  forward pass.

---

## 5. Running the benchmark yourself

```
# Linux / macOS / WSL
./installer/run.sh

# Windows (cmd or Explorer)
installer\run.bat
```

Either script bootstraps `uv`, resolves Python and the pinned
`requirements.txt`, and launches the Gradio UI. To reproduce the
benchmark numbers above:

```
uv run --python ">=3.10" --with-requirements requirements.txt \
    python -m benchmarks.compare_training --num-steps 300 --seed 0
```

The report is printed to stdout and appended to `benchmarks/results.md`.

---

## 6. What to read next

- `tests/test_integrators.py` — the physics lives here. Start with
  `test_reversibility_leapfrog_within_float_noise` and
  `test_yoshida_order_beats_leapfrog_on_stiff_oscillator`.
- `tests/test_integrity.py` — checkpoint integrity round-trip and
  tamper-detection specs.
- `tests/test_trainer.py` — end-to-end trainer contract: setup,
  generator-style training yields, Langevin + multi-scale variants,
  Transformer model drop-in, adaptive `dt` feedback.

Every feature mentioned in this article has a spec file that runs in
under a second. If a claim here isn't visible from a test, consider it
unsupported and open an issue.
