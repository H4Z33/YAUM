# YAUM — Training as a Physical System

**Thesis.** Neural networks need more than data and mechanics; they need
*rules*. Physics has spent four centuries finding good rules for systems
that evolve in time: energy, momentum, conservation laws, and the
symmetries those laws come from. YAUM asks whether those rules make a
good substrate for a model to grow in.

A language model is more than a next-token picker. While it reads, it
builds a small world that moves forward with each token. Word2vec showed
that knowledge has geometry (`king − man + woman ≈ queen`); attention lets
the parts of that world interact coherently. The bet is that there are
more structural relationships to find — the way particles combine into
other particles — and that giving the model's internal dynamics physical
structure is a way to find them.

This document records what has been built, what was measured, and where
the measurements went against the original framing. It was revised in
November 2026 after an audit found that several earlier claims were
measurement artefacts; §6 lists what changed.

---

## 1. Two places to put the physics

There are two different "times" in a language model:

| | training time | context time |
|---|---|---|
| clock | optimiser steps | tokens read |
| what moves | the parameters | the hidden state |
| YAUM component | `yaum/core/` trainer + integrators | `yaum/models/hamiltonian.py` |

YAUM started with the first: treat the embedding table `E` as particles
with momentum and mass, moved by the gradient of the loss. That turns out
to be a respectable *optimiser* (§3) — but an optimiser is not where "a
small universe that moves with context" lives. The second place puts the
physics in the forward pass itself (§4), and it is the more direct test of
the thesis.

---

## 2. The core machinery

A symplectic step (velocity Verlet / leapfrog):

```
P ← P + F(E) · dt/2          # half momentum kick
E ← E + (P / m) · dt         # position drift
P ← P + F(E_new) · dt/2      # half momentum kick
```

For a *fixed* force field this step is time-reversible and conserves a
"shadow" energy close to `H = T + V`, so the energy error stays bounded
instead of drifting. YAUM provides:

* `leapfrog`, `yoshida4` (three leapfrogs with weights `w₁, w₂, w₁`),
  `multiscale` (rRESPA: slow L2 force around fast substeps), and
  `langevin` (OBABO: friction + noise around any core; `temperature=0`
  gives pure damping);
* `HamiltonianSubstrate` — explicit `(E, P, m, dt)` state;
* `AdaptiveStepController` — shrinks `dt` when the energy error spikes;
* `measure_reversibility` — forward-then-backward round trip;
* phase-transition observables (`Var(H)`, `Var(‖mean E‖)`, `τ_int`, log-det
  entropy rate) and an action integral `S = Σ (T − V) dt`.

### What the diagnostics can and cannot tell you

During training the force field is **not fixed**: every step uses a new
minibatch, and Adam changes the weights `W` between steps. So `H` is not
conserved by the training trajectory, and nothing in the code should
pretend it is.

* **`|ΔH/H|`** is measured per step on the *conservative core*: same
  batch, same `W`, same mass, before vs. after one integrator step. That
  is a real integrator-error signal and it is what drives the adaptive
  `dt`. (An earlier version compared `H` to its value at the first eval;
  since `H ≈ V = loss`, that number was mostly "how much has the loss
  fallen", and it made the controller shrink `dt` to its floor as soon as
  the model learned.)
* **Reversibility** is checked with `W` frozen and one batch held fixed.
  For leapfrog it is reversible by construction, so the probe detects
  numerical trouble (precision, nondeterministic kernels, absurd `dt`),
  not anything about learning. It reports NaN for stochastic integrators.
* **`Var(H)` and friends** are computed over a window of eval snapshots
  and include the downward trend of the loss. Treat them as exploratory
  telemetry. There is currently no evidence that they anticipate
  grokking-style transitions.
* **`S = Σ (T − V) dt`** is a bookkeeping integral. Because `V` changes
  with every batch and every weight update, the training trajectory does
  not extremise it.

---

## 3. Physics in training time: results

`benchmarks/compare_training.py` holds everything fixed except how `E` is
updated (RNN char model, Adam on `W`, synthetic vocab-32 corpus, 600 steps,
3 seeds, fixed held-out windows):

| variant | held-out loss | ‖E−E₀‖/‖E₀‖ |
|---|---:|---:|
| `frozen_E` (E never updated) | 0.2206 ± 0.0073 | 0.00 |
| `adam_E` (lr 1e-3) | 0.1978 ± 0.0049 | 0.29 |
| `adam_E_tuned` (lr 1e-2, best of sweep) | 0.1708 ± 0.0046 | 1.88 |
| `yaum_static` (leapfrog, mass 1) | 0.1989 ± 0.0038 | 0.33 |
| `yaum_fisher` (leapfrog, RMS-Fisher mass) | 0.1686 ± 0.0080 | 7.61 |
| `yaum_damped` (defaults: RMS-Fisher + friction) | 0.1699 ± 0.0065 | 2.76 |

What this says:

1. **Dynamics on `E` help** (every non-frozen variant beats `frozen_E`).
2. **The mass is what matters.** With `m = 1` the force (the gradient of
   a mean loss, ~1e-4 per coordinate) barely moves anything. With
   `m = sqrt(EMA[F²]) + ε` the acceleration `F/m` is O(1) whatever the
   gradient scale — the same trick that makes RMSProp/Adam work, expressed
   as inertia. (The earlier "Fisher mass" used `m = EMA[F²]`, which was
   ~1e-7 and sat entirely under a 1e-3 floor: in practice a constant.)
3. **Against a tuned Adam it is a tie**, not a win: 0.1699 vs 0.1708 with
   seed std ≈ 0.007. An untuned Adam baseline (lr 1e-3) makes YAUM look
   0.03 better; that is a statement about the baseline, not about physics.
4. **Undamped motion does not settle.** Without friction, `E` keeps moving
   (7.6× its initial norm here). Friction keeps the same loss with a third
   of the travel, which is why the defaults are damped.

So in training time YAUM is a heavy-ball optimiser with an RMS
preconditioner and a clean energy-error signal for its step size. That is
useful and well-behaved, but it is not a new kind of model.

---

## 4. Physics in context time

`HamiltonianCharModel` makes the hidden state itself a mechanical system.

```
H(q, p) = Σᵢ pᵢ² / 2mᵢ + V(q)
V(q)    = Σₖ aₖ log cosh(wₖᵀ q + bₖ) + ½ Σᵢ κᵢ qᵢ²
```

* Reading token `t` applies an impulse `p ← p + B xₜ` — tokens are forces.
* Between tokens the state follows leapfrog steps of the flow. The force
  `−∇V` has a closed form, so the dynamics are conservative by
  construction.
* `aₖ` is unconstrained (many wells and barriers are possible) and
  `κᵢ > 0` confines the motion.
* Each mode has a friction `γᵢ ≥ 0`, applied as `p ← p·exp(−γᵢ dt/2)` on
  both sides of each leapfrog step. That is conformally symplectic
  integration: exact bookkeeping for the conservative part and exponential
  decay for the dissipative part. `dissipation="learned"` trains `γ` with
  everything else; `"none"` forces a purely conservative model.

The tests (`tests/test_hamiltonian_model.py`) check the physics: the force
is the exact gradient of `V`, the frictionless flow conserves energy over
thousands of steps and runs backwards to its start, and friction drains
energy toward the potential's minimum.

### Why friction is part of the design

A purely Hamiltonian system preserves phase-space volume. It can carry
information forever, but it can never *contract*: no attractors, no
"deciding", no forgetting. Computation needs both. The physical universe
works the same way — its microscopic laws are conservative, yet every
structure we care about (stars, chemistry, life) forms through
dissipation. So the refined thesis is **a conservative core for memory
and coherence plus controlled dissipation for decisions**. Learning `γᵢ`
per mode turns that from a slogan into a measurement.

### The experiment

`benchmarks/context_dynamics.py` trains each model with plain Adam and an
ordinary `nn.Embedding` on three synthetic tasks:

* `recall_first` — remember the first token across 39 distractors.
  Needs lossless long memory.
* `count` — track the depth of a random bracket walk. Needs exact
  accumulation.
* `lag` — output the token from 5 steps ago. Needs *short* memory: old
  tokens must be forgotten.

**Hypothesis, registered before the first run:** the learned `γ` stays
near zero on `recall_first` and `count` and grows on `lag`; the
conservative model does well on the first two and worst on `lag`.

*Results: see `benchmarks/context_dynamics_results.md` (this section is updated once the first full run completes).*

---

## 5. Related work worth reading

* Hamiltonian Neural Networks (Greydanus et al., 2019); Neural ODEs
  (Chen et al., 2018); reversible ResNets (Gomez et al., 2017).
* Oscillator RNNs: coRNN (Rusch & Mishra, 2021); AntisymmetricRNN
  (Chang et al., 2019).
* Transformers as interacting-particle systems: Lu et al. (2019);
  Geshkovski, Letrouit, Polyanskiy & Rigollet, *A Mathematical
  Perspective on Transformers* (2023).
* Attention as energy minimisation: Ramsauer et al., *Hopfield Networks
  Is All You Need* (2020).
* Geometry and composition of representations: Poincaré embeddings
  (Nickel & Kiela, 2017); *Toy Models of Superposition* (Elhage et al.,
  2022); vector-symbolic / holographic reduced representations;
  geometric deep learning (Bronstein et al.).

---

## 6. Corrections to earlier versions of this document

| earlier claim | status |
|---|---|
| "`ΔH/H` = 0.83 is a loud warning that dt is too large" | Wrong. It was `(L₀ − L)/L₀`, i.e. learning progress. |
| "adaptive dt shrinks dt within a few ticks" (presented as a feature) | It shrank `dt` to its floor because the loss fell, which froze `E`. Fixed. |
| "Fisher mass turns the flow into natural-gradient dynamics" | The old mass was a constant at its floor. The new one is an RMS (Adam-style) preconditioner, not full natural gradient. |
| "Adam wins by 6.7 %" / "YAUM beats Adam" | Both were artefacts (a broken mass; an untuned baseline). With both sides working, it is a tie. |
| "leapfrog and yoshida4 agree to float32 precision" | True, but only because the forces were too small to move `E` at all. |
| "checkpoint fingerprint is sensitive to any tampering" | No: anyone can recompute three norms. It catches corruption. Security comes from `weights_only=True` loading. |
| "None of this is available to SGD/Adam, which forgets everything" | Adam keeps momentum and second-moment state; heavy-ball with an RMS mass is close kin to it. |

---

## 7. Reproducing

```
python -m pytest -q
python -m benchmarks.compare_training          # §3
python -m benchmarks.context_dynamics          # §4
```

Every claim in this document should be visible either in a test or in one
of the two benchmark reports. If it isn't, treat it as unsupported and
open an issue.
