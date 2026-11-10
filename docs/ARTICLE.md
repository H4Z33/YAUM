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

### Results (1500 steps, T = 40, hidden 64, 3 seeds)

Accuracy on 1024 fresh sequences. Chance is 0.125, except ≈ 0.2 for
`count`. The LSTM uses a forget-gate bias of 3, the setting that makes it
a fair baseline (see below). Parameters: LSTM 21.8k, Hamiltonian models
11.2–11.6k. Raw runs are in `benchmarks/context_dynamics_results.md`.

| task | LSTM | Ham., γ = 0 | Ham., learned γ | Ham. + dynamic geometry | learned γ |
|---|---:|---:|---:|---:|---:|
| `recall_first` | **1.000** | **1.000** | **1.000** | 0.318 ± 0.272 | 0.035 |
| `count` | **1.000** | 0.738 ± 0.051 | **1.000** | 0.277 ± 0.003 | 0.153 |
| `lag` | **1.000** | 0.934 ± 0.012 | 0.996 ± 0.002 | 0.516 ± 0.033 | 0.236 |
| `assoc` | **0.688 ± 0.040** | 0.398 ± 0.013 | 0.399 ± 0.014 | 0.404 ± 0.007 | 0.085 |

**What held up:**

1. **The learned friction orders the tasks** by how much forgetting or
   settling they need. Across all three seeds: `recall_first` 0.035
   (*below* its 0.05 start) < `assoc` 0.085 < `count` 0.153 < `lag`
   0.236. When it can choose, the model keeps lossless memory nearly
   conservative and adds dissipation where old information must fade.
2. **Purely conservative is never best.** The γ = 0 model is the weakest
   static Hamiltonian variant wherever there is a difference. This
   supports "conservative core + controlled dissipation" over "purely
   Hamiltonian".
3. A single moving phase point, with half the LSTM's parameters, solves
   long recall, bounded counting and lag copying.

**What did not:**

4. **No advantage over a fair LSTM.** The first runs showed the LSTM at
   chance on `recall_first` while the Hamiltonian models were perfect. That
   gap came from PyTorch's LSTM initialisation: a forget gate ≈ 0.5
   erases a 40-step memory (0.5⁴⁰ ≈ 1e-12). With the standard fix, a
   forget-gate bias of 3, the LSTM solves `recall_first` in every setting
   tried. With that baseline, the LSTM matches or beats every Hamiltonian
   variant on every task. On `assoc` it leads clearly (0.69 vs 0.40).
5. **The `count` prediction was wrong.** It was expected to want γ ≈ 0
   (accumulation is what momentum does). It wanted 0.15. A guess: a
   discrete, bounded state needs attractors, which only dissipation
   provides. Untested.
6. **Dynamic geometry failed as implemented, and the second registered
   hypothesis is not supported.** On `assoc` it ties the static model
   (0.404 vs 0.399) instead of beating it. On the other tasks it is
   *much* worse. The diagnosis is a training pathology, not a verdict on
   the idea: the coupled matter–geometry dynamics make gradients explode
   (median norm ~1e3 vs ~2, spikes to 1e5), and the geometry parameters
   hardly move from their initial values. A coupled nonlinear system like
   this is prone to chaos (positive Lyapunov exponents), and
   backpropagating through chaos is exactly where gradients explode.
   Taming it (weaker initial coupling, slower and stiffer geometry,
   smaller `dt`, truncated backprop) is the next step if the idea is
   pursued.
7. **One particle is probably the real bottleneck on `assoc`.** Every
   Hamiltonian model here compresses the whole context into the motion of
   *one* point. Storing several bindings in one trajectory is hard,
   whether or not the landscape bends. The next design (§4.1) makes every
   token its own particle.

**Caveats:** synthetic tasks, short sequences, unmatched parameter
counts, and a Hamiltonian model about 10× slower per step (Python loop
over tokens).

### 4.1 A many-body universe

Instead of one particle in a (possibly bending) landscape, every token
becomes a particle that stays in the system, with its own position and
momentum. Particles attract or repel each other through a learned,
content-dependent pair interaction:

```
H = Σᵢ |pᵢ|²/2m + Σᵢ U(qᵢ) + Σᵢ<ⱼ φ(qᵢ, qⱼ)
```

Clusters such as bound key–value pairs can then form as composite
objects ("particles forming particles"). The query enters as a new
particle and is pulled toward its matching cluster. Compared with
attention, the forces are symmetric (Newton's third law, so total momentum
is conserved), second-order (particles carry momentum), and derived from
one total energy. The open question is whether a conservative,
momentum-carrying, symmetric interaction can do what attention does.

**Built and measured** (`yaum/models/particles.py`, 2026-11-09). The
interaction is φᵢⱼ = −(cᵢ·cⱼ)·exp(−|qᵢ−qⱼ|²/2σ²) with content-derived
charges, confinement U = ½κ|q|², learned per-dimension friction, and a
readout from the newest particle (position, velocity and the force it
feels). Physics tests pass: momentum and energy conservation, float64
reversibility, like charges bind, opposite charges repel, bounded energy.

The first draft was untrainable (gradient norms 1e9–1e13 on 40-token
sequences: close-encounter chaos). Unit-direction charges with one learned
strength, a floor on σ, and the force readout brought gradients to ~1.

Pre-registered: beats `ham_learned` on `assoc` by more than noise, learns
γ > 0.05 there, ≥ `ham_learned` on recall and count, worse on lag.
Result (particle dim 32, 3 seeds, 1500 steps):

| task | particles | per seed | `ham_learned` | LSTM |
|---|---:|---|---:|---:|
| `assoc` | 0.392 ± 0.231 | 0.50, 0.55, 0.13 | 0.399 | **0.688** |
| `recall_first` | 0.125 ± 0.015 | 0.12, 0.14, 0.11 | **1.000** | **1.000** |
| `count` | 0.250 ± 0.029 | 0.22, 0.27, 0.26 | **1.000** | **1.000** |
| `lag` | 0.150 ± 0.018 | 0.13, 0.16, 0.16 | 0.996 | **1.000** |

**Largely not supported.** On `assoc` two seeds are the best any
Hamiltonian model has scored (0.50, 0.55) and γ rose slightly (0.064),
but one seed never left chance, so the mean is a tie. On the three
40-token tasks it stays at chance and γ never moves. Unverified
explanations, each with a cheap test in the backlog: new particles land
outside the interaction range of the old ones and feel nothing; the
learning plateau seen on `assoc` is never crossed within 1500 steps of
40-token sequences; with no positional signal, "first" and "k steps ago"
are hard to read from drift alone.

**"We need a Big Bang"** (owner, 2026-11-09). A probe confirmed the first
explanation: at initialisation a typical pair of particles is ~3σ apart
and feels ~2e-5 of the full interaction, so each new particle only feels
copies of its own token, and the first token of `recall_first` is
effectively invisible. The universe starts cold and sparse. Three
cosmological options were added (off by default): a learned birth scale
(dense, "hot" start), a signed confinement acting as a cosmological
constant (expansion, distance as a clock), and learned primordial
particles (an initial condition before any token). A 3-seed diagnostic
(results file) showed:

* **density enables binding:** `assoc` 0.81 ± 0.31 with two seeds at 0.98–0.99;
* **expansion calms gradients but erases the past:** max gradient norm < 2
  everywhere, yet chance on recall. The model learned to expand on
  recall and to confine on `assoc` when allowed both;
* **primordial particles gave the only recall signal** (best seed 0.36),
  with the worst gradients (up to 4e6).

The pre-registered 5-seed check at the benchmark's settings (embedding
16) did **not** confirm the `assoc` win: dense birth 0.571 ± 0.206 (one
seed 0.93) versus LSTM 0.678 ± 0.060 and sparse-born particles
0.333 ± 0.175. Dense birth probably helps; it does not beat the LSTM,
and its outcome is strongly seed-dependent (bimodal: a seed either finds
the binding solution or stalls near 0.5).

### 4.2 Memory at lengths never trained on: rotation, then a conservation law

Every task above is one a standard model solves, so a tie was the best
possible outcome. Conservation predicts something sharper: memory that
does not fade with length. Models were trained on `recall_first` at
T = 40 and evaluated on longer sequences (5 seeds; results file,
2026-11-10).

**The registered prediction failed.** `ham_conservative` falls to chance
by T = 100 (0.146); the LSTM holds 0.865 at T = 400. (The friction model
did keep `count` and `lag` at 0.92 and 0.99 out to T = 400, where the
transformer is at chance on both.)

**Why: the memory rotates.** Accuracy oscillates with length: 1.00 at
T = 30, 40, 50 and exactly 0.000, *below chance*, at T = 36 and 44. A
fresh linear readout fitted on the frozen model scores 1.000 at T = 44
and T = 100. The first token is stored as the phase of an oscillation
(period ≈ 10 tokens) and the trained readout only matches that phase at
the trained length. The cause is in the design: every coordinate has a
restoring force (κ > 0), so every mode oscillates and nothing can hold
still. The memory also genuinely fades (probe 0.64 at T = 400).

**The cure physics names: a symmetry.** If H does not depend on a
coordinate, the conjugate momentum is exactly conserved (Noether).
`n_cyclic=16` removes 16 of the 64 coordinates from V and from friction;
a test confirms their momenta are bit-for-bit unchanged by the flow.

**Result (hypothesis 6):** not supported as stated, and the first result
in this project that favours the physics on any measure.

| at T = 1000, trained on 20–40 | own readout | fresh linear readout (what the state still holds) |
|---|---:|---:|
| LSTM | **0.869 ± 0.092** | 0.899 ± 0.107 |
| `ham_conservative` | 0.140 | 0.327 ± 0.197 |
| `ham_cyclic` | 0.125 | **1.000 ± 0.000** |
| `ham_cyclic_learned` | 0.131 | **1.000 ± 0.000** |

With the symmetry, the first token is perfectly recoverable at 25× the
training length on all 10 runs, where the LSTM's state has started to
lose it. But the model cannot *use* it end to end: its own readout is at
chance by T = 400, and the LSTM is far better there. The reason is
measured: distractor tokens push the conserved momentum by ~8% of a
first token's push, with a small consistent bias, and over 1000 tokens
that bias moves the memory ~7× the size of the signal. A re-fitted
readout re-centres; a fixed one cannot.

**What is missing** is a rule for which tokens may couple to the
conserved quantity, a *selection rule*. The LSTM has one (its input
gate). Note the honest reading: this shows a conservation law protects
information better than a learned gate does, not that the model is
better; and "a fresh readout per length" is a probe, not a usable model.

**Finding the selection rule took four attempts** (hypotheses 7–10, all
pre-registered; results file, 2026-11-10). Each failure narrowed the
cause:

| attempt | idea | recall at T = 400 (trained on 20–40) |
|---|---|---:|
| 7 | a learned charge in (0, 1) scales each token's push on the conserved coordinates | 0.145 |
| 8 | the same, plus a cost on every write | 0.49 at best |
| 9 | a cost on the *mean* push only (the measured cause: a fixed readout re-centred scores 1.000 at T = 1000) | 0.32–0.40 |
| 10 | a charge that can be **exactly zero** (ReLU), plus a cost per charged token | **1.000** |

The first three made the leak smaller, never zero, and over 1000 tokens
a leak must be below ~1e-4 of a write to be harmless. A sigmoid cannot
reach zero; a ReLU can. With it the model learns *neutral* tokens:
charge 0.5–1.3 for the eight symbols worth remembering and exactly 0.0
for the eight distractors. A neutral token cannot move a conserved
momentum (there is a test for this), so after the write nothing touches
the memory at any length.

| trained on 20–40 tokens, own readout, 5 seeds | T = 1000 | T = 10,000 |
|---|---:|---:|
| `ham_neutral` (charge cost 0.1) | **1.000 ± 0.000** | **1.000 ± 0.000** |
| LSTM, forget bias 3 (the setting used so far) | 0.875 ± 0.084 | 0.825 ± 0.070 |
| LSTM, best of 13 settings (forget bias 3/5/8 × lr × steps) | 0.972 ± 0.063 | 0.948 ± 0.071 |

This is the first end-to-end result where the physics does something a
tuned standard model does not: recall at 250× the training length is
exact on every seed, because of a conservation law plus a selection
rule, not because of a longer training budget. What it does **not**
show:

* The margin over the best LSTM is about one LSTM standard deviation on
  five seeds. The difference in kind (exact on all seeds vs 0.95 with
  some seeds degrading) is the claim, not the mean.
* One task. Its first token and its distractors come from different
  symbol sets, so a charge that depends only on the current token is
  enough by construction. A task where the same symbol must be stored
  once and ignored later needs a charge that depends on the state.
* The charge cost needed a sweep: with no cost the model fails like its
  predecessors, at 0.01 one seed in five never learns to write.
* A conserved memory cannot be overwritten or forgotten. That is the
  price of the guarantee, and real use needs both kinds of coordinate.
* Training an LSTM on lengths 20–100 also fixes its T = 1000 recall
  (0.983). The Hamiltonian model gets there without seeing long
  sequences.

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
| (first context-time run) "Hamiltonian models solve long recall where the LSTM stays at chance" | Artefact of PyTorch's LSTM forget-gate init. With forget bias 3 the LSTM solves it too, and it matches or beats the Hamiltonian models on every task. |
| (pre-registered) "the many-body model beats single-particle models on `assoc` and is at least as good on recall/count" | Not supported: a noisy tie on `assoc` (0.39 ± 0.23), chance on recall, count and lag. |
| (diagnostic, 3 seeds, embedding 32) "dense-born particles solve `assoc` (0.81, two seeds ≈ 0.99), above the LSTM" | Not confirmed by the pre-registered 5-seed run at embedding 16: 0.571 ± 0.206 vs LSTM 0.678. |
| (pre-registered) "the conservative model keeps recall ≥ 0.9 at 10× the training length and degrades less than the LSTM" | Not supported: chance by T = 100. The memory is a rotating phase the readout cannot follow (§4.2). |
| (pre-registered) "with cyclic coordinates the model's own recall is ≥ 0.9 at T = 400" | Not supported (0.18–0.35). The state still holds the token perfectly at T = 1000 (fresh readout 1.000 on 10/10 runs vs LSTM 0.90), but distractors drift the conserved momentum and the trained readout cannot follow. |
| (pre-registered, three times) "a soft charge / a write cost / a net-force penalty lets the model read its conserved memory at T = 400" | Not supported (0.15–0.49). Each shrank the per-token leak without making it zero. A charge that can be exactly zero (hypothesis 10) was needed: 1.000 to T = 10,000. |
| "LSTM 0.875 at T = 1000" as the bar for long recall | Under-tuned for this test. Forget bias 8 or lr 1e-2 gives 0.95–0.97 at T = 1000 and 0.95 at T = 10,000. Comparisons now quote the best of 13 settings. |
| "On `assoc` the LSTM leads clearly (0.69)" as the bar; "dense birth is bimodal: a seed finds the binding solution or stalls near 0.5" | Both are artefacts of the 1500-step budget. At 6000 steps the LSTM reaches 0.99 and the "stalled" dense seeds reach 0.79–1.00 (they were slow, not stuck). A small transformer, never benchmarked before, scores 1.000 at 1500 steps on all seeds and is the real bar. See the 2026-11-10 audit diagnostic in the results file. |

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
