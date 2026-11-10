# Lessons learned and open backlog

A running record for future sessions (human or agent). Add to it; don't
rewrite past entries. Newest session first.

---

## Session 2026-11-10 — second audit

20. **A fixed step budget turns "slow" into "can't".** Every `assoc`
    number in the article was read as ability. At 4× the steps the LSTM
    goes 0.69 → 0.99 and the "stalled" dense-birth seeds go 0.5 → 0.79–1.00.
    Plot (or at least sample) the learning curve and run one longer
    budget before describing a model as stuck, bimodal or unable.
21. **Benchmark the mechanism you compare yourself to.** The particle
    model was described as "a cousin of attention" for two days without
    a transformer in the table. It solves `assoc` in 1500 steps on every
    seed with 7k parameters.
23. **Test where the idea should win, not where everything ties.** Two
    days of tasks that standard models solve could only produce ties.
    One off-distribution test (longer sequences) produced the project's
    first surprising results in an afternoon.
24. **Evaluate next to the training point, not only far from it.**
    Accuracy 1.000 at T=40 and 0.000 at T=44 exposed a rotating memory
    that "0.15 at T=100" only reported as failure. Below-chance accuracy
    means information is present and misread.
25. **Separate "is the information there" from "can the model use it".**
    A linear probe on the frozen state answered the first; the model's
    own accuracy answers the second. They disagreed completely (1.000 vs
    0.13 at T=1000), and each alone would have told the wrong story.
    Report both, and never present the probe number as model accuracy.
26. **Read your own design for forced behaviour.** `κ > 0` on every
    coordinate meant every mode had to oscillate. No experiment was
    needed to know there was no place for a static memory.
22. **Diagnostics that back a recorded table belong in the repo.** The
    Big Bang script lived in a scratch directory; it is now
    `benchmarks/diagnostics/`.

---

## Session 2026-11-09 — many-body particle model

14. **Bound the strengths, floor the ranges.** The first particle draft
    had free charges and a free interaction range. Close encounters
    between strongly charged particles made backprop explode (1e9–1e13).
    Unit charges times one learned scale, `σ ≥ 0.25`, and reading out the
    force on the newest particle brought gradient norms to ~1. The
    gradient check (CLAUDE.md rule 7) found this in minutes. The full
    benchmark would have taken an hour to show "learns nothing".
15. **A plateau on the easy task predicts failure on the hard one.**
    `assoc` (9 tokens) sat flat for hundreds of steps before learning. The
    40-token tasks get the same step budget, are ~10× slower per step, and
    never left chance. Look at the learning curve on the shortest task
    before launching long ones.
16. **Physics tests don't predict learnability (again, see 0c).** All 11
    physics tests pass; the model is at chance on three of four tasks.
17. **Measure whether the parts can see each other.** One probe (kernel
    value between the newest particle and the rest, at init) explained
    the many-body failure: pairs felt ~2e-5 of the interaction. Any
    interaction with a length scale needs its initial distances checked
    against that scale, the same lesson as #3 (magnitudes) in geometry.
18. **Stability and memory trade off, physically.** Expansion (negative
    κ) made gradients perfectly tame and made recall impossible;
    primordial particles made recall possible and gradients explode.
    Treat "calm gradients" as necessary, not as progress.
19. **Pre-register before replicating a diagnostic.** A 3-seed diagnostic
    gave `assoc` 0.81 with two seeds at 0.99. Registering the claim first
    and rerunning at the benchmark's settings gave 0.57: bimodal seeds
    plus a config difference. Without the registration step it would
    have been reported as "beats the LSTM".

---

## Session 2026-11-08 — audit, fixes, context-time model

### Background

The project was originally built by delegating most of the implementation
to a less capable model while the owner was learning how LLMs work. The
code was clean and well tested at the unit level, but the *system* did
not do what the docs said. An audit (rated 4/10 overall, 2/10 for
scientific soundness) found the problems below; all are fixed in commits
`5c3b507`, `ed92aa1` and `cd9535d` on branch `ccr-6526aa7d-7gkap1`.

### Lessons

0. **The first result that bears on the real thesis came from moving the
   physics into context time.** A Hamiltonian hidden state with learned
   friction learns usable memory, and the friction it learns orders the
   tasks by how much forgetting they need. That part survived 3 seeds.
   The "beats the LSTM on long recall" part did **not**; see 0b.

0b. **Baselines can fail for initialisation reasons, too.** The LSTM sat at
   chance on 40-step recall across 12 lr/steps/size settings. Every one of
   them used PyTorch's default forget gate (≈ 0.5 per step, so 0.5⁴⁰ of the
   memory survives). Forget bias 1.0 still failed; 3.0 solved it in every
   setting. "Tune the baseline" means its *initialisation* as well as its
   learning rate. A baseline stuck at exactly chance everywhere is a red
   flag, not a result.

0c. **Coupled dynamics can be untrainable even when the physics is right.**
   The dynamic-geometry model passes every physics test (exact gradients,
   energy conservation, reversibility, bounded energy). It still trains
   badly: gradient norms ~1e3 with spikes to 1e5, versus ~2 for the static
   model. Backprop through a chaotic system explodes. Check gradient norms
   in the first 100 steps of any new dynamics before running the full
   benchmark.

0d. **An unbounded coupling collapses.** The first geometry draft let φ
   scale the unbounded `log cosh`, so the joint energy had no lower bound
   (energy 1e34 in tests). Couplings between subsystems need bounded
   features, or a stiffness proven to dominate.

1. **A diagnostic can be precise and still measure the wrong thing.**
   `|ΔH/H|` was computed against the first eval's `H`. Since `H ≈ loss`,
   the "energy drift" of 0.83 was just `(L₀ − L)/L₀`. The docs read it as
   "dt too large". *Always ask: if the system were working perfectly, what
   value would this number take?* Here the answer would have been "large",
   which exposes the bug.

2. **Feedback loops amplify a wrong signal.** The adaptive dt consumed the
   bogus drift and shrank `dt` to its floor within ~12 evals of normal
   learning, freezing the embeddings. A wrong telemetry number is cosmetic;
   a wrong number inside a controller breaks training.

3. **Check the magnitude of every quantity that meets a constant.** The
   "Fisher mass" was `EMA[g²]` with a floor of 1e-3; real `g²` was ~1e-7,
   so 100% of entries sat on the floor and the mass was a constant. One
   `print(tensor.max())` would have shown it.

4. **Unit tests pass while the system is broken.** 136/137 tests passed.
   They checked each part in isolation (the EMA blends, the controller
   shrinks on high drift) but never asked a system-level question ("does dt
   survive a run that is learning?"). Property tests of that kind are now
   in `tests/test_trainer.py` (`test_adaptive_dt_does_not_collapse_while_loss_falls`,
   `test_h_drift_is_a_one_step_error_not_loss_progress`,
   `test_fisher_mass_is_not_pinned_to_its_floor`).

5. **No control, no conclusion.** The benchmark compared YAUM to Adam but
   not to *doing nothing*. Adding `frozen_E` showed the original YAUM
   moved `E` by ~10% and sat between frozen and Adam.

6. **Tune the baseline before you celebrate.** After the fixes, YAUM
   seemed to beat Adam-on-E by 0.028. Sweeping Adam's lr (1e-3 → 1e-2)
   closed the gap to 0.001, inside seed noise: a tie.

7. **Put the physics where the idea lives.** The owner's intuition is
   about *context time* (the model's state evolving as it reads), but the
   code put Hamiltonian dynamics on *training time* (the optimiser). That
   mismatch, not any single bug, is why the project felt stuck. See
   `yaum/models/hamiltonian.py`.

8. **Conservation alone can't compute.** Hamiltonian flow preserves
   phase-space volume: it can't settle, decide or forget. Undamped
   training runs kept moving `E` (7.6× its initial norm). Design principle
   now: *conservative core + controlled dissipation*, with the amount of
   dissipation learned and measured.

9. **Changing mass at fixed momentum injects energy.** When the RMS mass
   of a coordinate drops, velocity `P/m` jumps. Without rescaling `P` (and
   without keeping unseen tokens' mass fixed), `E` grew to ~1000× its
   initial norm.

10. **Secure defaults matter even for research tools.** The UI launched a
    public `gradio.live` link by default and loaded checkpoints with full
    pickle, meaning anyone with the link could run code on the owner's PC.
    It now binds to localhost and loads with `weights_only=True`.

11. **Docs must not outrun evidence.** README/ARTICLE claimed grokking
    precursors, natural-gradient dynamics, tamper detection, "dual
    licensing", CLI flags that didn't exist, and files that didn't exist.
    `docs/ARTICLE.md` §6 now keeps a corrections table.

12. **Not every crash is a code bug.** The `safe_speed`, `step_delay`,
    `cuda_sync_interval`, `disable_cudnn` and `cudnn_benchmark` knobs were
    added while chasing Windows GPU crashes. The owner later confirmed the
    cause was **a driver issue on their PC, now fixed**. The knobs are
    harmless but cost throughput (1 ms sleep plus a CUDA sync every step
    by default).

13. **Read tool output carefully.** The agent misattributed a
    `git ls-files` line to `.gitignore`, which briefly led to a wrong fix
    and a slightly wrong commit message (`cd9535d` says the old ignore
    line "had no trailing newline match issue"; in fact `.gradio/` was
    simply never ignored).

---

## Open problems and backlog

Priority order within each group is rough. Mark items done with the commit
hash instead of deleting them.

### Research (context time — the main line)

- [x] Read the first `context_dynamics` results and write them into
      `docs/ARTICLE.md` §4. Learned γ: recall 0.036 < count 0.15 < lag 0.24.
      Hypothesis supported on recall and lag, **wrong on count**: discrete
      bounded state seems to need attractors, so it needs dissipation.
- [x] Tune the LSTM on `recall_first`: forget bias 3 solves it; the
      long-memory advantage is gone (ARTICLE §6 corrections table).
- [x] Third seed for the context table (results unchanged).
- [x] Dynamic geometry, version A (bending resets every sequence):
      `geometry="dynamic"`. Registered hypothesis **not supported**: it ties
      on `assoc` and trains badly elsewhere (exploding gradients).
- [x] **Many-body model** (owner's idea, ARTICLE §4.1):
      `yaum/models/particles.py` (`daf5720`). Pre-registered hypothesis
      **largely not supported**: assoc 0.39 ± 0.23 (seeds 0.50/0.55/0.13),
      chance on recall, count and lag.
- [ ] Diagnose the many-body failure before changing the design. Short
      runs (100–200 steps), logging gradient norms and the distance from
      the newest particle to its nearest neighbour in units of σ:
      (a) spawn new particles near the centroid of the existing ones
      (does the new particle feel anything?); (b) add a fixed age/
      position component to the charge (a minimal clock); (c) 4–6k steps
      on `recall_first` (is it just the plateau?). Also more seeds on
      `assoc` (is seed 2's failure common?).
      Done for (a) as a probe plus "Big Bang" options (dense birth,
      expansion, primordial particles); see ARTICLE §4.1. (b) and (c)
      still open.
- [ ] Big Bang follow-ups: why is dense birth bimodal across seeds
      (what differs in the stalled ones: σ, charge scale, spawn scale)?
      Embedding 32 vs 16 at 5 seeds. Tame the primordial-particle
      gradient explosions (friction on primordials, smaller dt,
      truncated backprop) and recheck its recall signal. A schedule
      ("inflation then cooling": expansion only for the first few
      tokens, or friction rising with time) instead of constant κ.
      Give the LSTM a longer budget too before comparing at more steps.
- [ ] Stabilise dynamic geometry before judging the idea: weaker initial
      coupling (scale tanh² by a learned ε ≈ 0.1), heavier/stiffer geometry
      (slower timescale), smaller `dt`, truncated backprop, gradient-norm
      monitoring.
- [ ] **Version B / persistent universe** (owner, 2026-11-08): keep the
      bent geometry or particle state between sessions, possibly as a
      saved phase-state *cache* to resume from instead of textual
      compaction. Watch for drift and forgetting (continual-learning risks).
- [ ] **Selection rule for the conserved momenta** (top priority; ARTICLE
      §4.2). The cyclic momenta hold the first token perfectly at T=1000
      but distractors drift them (~8% of a first token's impulse, biased).
      Candidates, in order of physical cleanliness: (a) a content-derived
      charge that scales each token's impulse on the cyclic coordinates
      (zero charge = no coupling), the physics counterpart of an input
      gate; (b) a readout invariant to a uniform shift of the cyclic
      momenta; (c) training lengths up to 100+ so drift is penalised.
      Pre-register: own-readout recall ≥ 0.9 at T=400 and ≥ LSTM at T=1000.
- [ ] Give the particle model the same treatment: which of its
      quantities are conserved, and does a probe find the first token
      in them on the 40-token tasks where its own readout is at chance?
- [ ] Does `ham_learned` holding `count`/`lag` to T=400 (transformer at
      chance) survive a transformer with relative positions?
- [ ] Param-matched LSTM with forget bias 3 (hidden 44) on all tasks.
- [ ] Test the "discrete state needs attractors" reading of `count`, e.g.
      unbounded counting (no clipping) should want less γ than bounded.
- [ ] Per-mode analysis: with learned γ, do some modes go to ~0 (memory)
      while others go high (scratch)? Plot the γ distribution per task.
- [ ] Parameter-matched comparison (the LSTM has ~2× the parameters of the
      Hamiltonian model at the same `hidden_dim`).
- [ ] Longer sequences (T = 100, 200) where LSTMs are known to struggle;
      a sweep over `dt` and substeps.
- [ ] **Speed**: the Hamiltonian model loops over tokens and substeps in
      Python (~30× slower than cuDNN LSTM). Options: TorchScript /
      `torch.compile`, chunked integration, or a linear-potential variant
      that admits a parallel scan.
- [ ] Real text: train `model_type="hamiltonian"` on a char corpus and
      compare perplexity and sample quality with RNN/transformer.
- [ ] Inspect the learned potential: wells, bound states, which modes are
      conservative vs. dissipative after training. Are there stable
      "composite" states (the "particles forming particles" question)?
- [ ] Symmetry first: choose invariances the energy should have and
      derive conservation laws from them (Noether), instead of adding them
      by hand.
- [ ] Hybrid: Hamiltonian state inside a transformer block (attention as
      interaction force between token-particles).

### Research (training time)

- [ ] Wire `replica.py` (parallel tempering) into the trainer, or delete it.
- [ ] `multiscale` energy error ignores the analytic `V_slow` term.
- [ ] Observables include the loss trend; detrend before computing
      `Var(H)` / `τ_int`, or drop them if they never predict anything.
- [ ] Benchmark on real text and larger vocabularies; check whether the
      RMS-Fisher + friction default still ties Adam.

### Engineering

- [ ] **Relax GPU throttling defaults** now that the driver issue is
      fixed: `safe_speed=False`, `step_delay=0` (UI default is 0.001).
      Needs a run on the owner's Windows/NVIDIA machine to confirm.
- [ ] Windows launchers (`run.bat`, `run.ps1`, `installer/*`) always use
      the CUDA requirements; mirror `run.sh`'s CPU fallback and test on
      Windows.
- [ ] UI: expose `ham_dt` and `ham_dissipation` for the Hamiltonian model.
- [ ] Trainer writes `checkpoint_latest.pt` on every eval; make it a
      separate interval.
- [ ] `trainer.py` and `ui/app.py` are ~1000 lines each; split telemetry
      and checkpointing out of the trainer, and replace the UI's global
      `APP_STATE`.
- [ ] Generation recomputes the whole window for every token (O(n·ctx));
      use the recurrent state for RNN/Hamiltonian models.
- [ ] 16 deliberate `except Exception` blocks (ruff BLE001) at UI/thread
      boundaries; narrow them where possible.
- [ ] Old checkpoint configs may carry `fisher_eps=1e-3`, which under the
      new mass formula is a large absolute floor; warn or migrate on load.
