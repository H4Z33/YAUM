"""Does a sequence model want to conserve energy in context time?

Compares, on three synthetic tasks, an LSTM against two variants of
``HamiltonianCharModel`` (hidden state = phase point moving under a
learned energy as tokens arrive):

    ham_conservative   friction γ = 0: a purely Hamiltonian model
    transformer        2 layers, 2 heads, sinusoidal positions (the
                       mechanism the particle model is compared to)
    ham_cyclic         ham_conservative with 16 cyclic coordinates whose
                       momenta are exactly conserved (Noether);
                       ham_cyclic_learned adds learned friction elsewhere;
                       ham_charge adds a selection rule (a content-derived
                       charge gating what may change the conserved momenta);
                       ham_neutral uses a charge that can be exactly zero
    ham_learned        per-mode friction γ_i learned with the weights
    ham_geometry       ham_learned + dynamic geometry: the landscape is
                       bent by the content as it is read (joint
                       matter/geometry Hamiltonian; resets every sequence)
    particles          many-body universe: every token becomes a particle
                       with momentum and a content-derived charge;
                       symmetric pair attraction/repulsion; readout from
                       the newest particle (yaum/models/particles.py)
    particles_dense    the same, born dense (spawn_scale_init=0.1)

All models use an ordinary ``nn.Embedding`` and plain Adam — this
experiment is about the dynamics *inside* the forward pass, not the
phase-space optimiser in ``yaum.core``.

Tasks (sequence length ``T``, fresh random batches every step):

    recall_first   first token is one of 8 symbols, then T-1 distractors;
                   predict the first symbol at the last position.
                   Needs lossless long memory.
    count          a random walk of "(" / ")" with depth in [0, 7];
                   predict the depth after every token.
                   Needs exact accumulation of impulses.
    lag            predict the token seen 5 steps ago, at every position.
                   Needs memory that is *short* — old tokens must be
                   forgotten or they interfere.
    assoc          K (key, value) pairs with random bindings, then a query
                   key; predict its value. The bindings change every
                   sequence, so they cannot be stored in trained weights:
                   the rules must be written *while reading*.
                   Length 2K+1 (ignores --seq-len).

Pre-registered hypothesis (written before the first run): the learned
friction stays near zero on ``recall_first`` and ``count`` and grows on
``lag``; the conservative model does well on the first two and worst on
``lag``. If the learned γ does not separate the tasks, the "conservative
core + controlled dissipation" story is not supported by this setup.

Second pre-registered hypothesis (written 2026-11-08, before ``assoc``
or ``ham_geometry`` had ever been run): ``ham_geometry`` beats
``ham_learned`` on ``assoc`` by a margin larger than seed noise, and
ties it (within noise) on ``recall_first``, ``count`` and ``lag``. If
the dynamic geometry does not help on ``assoc``, "content must bend the
landscape" is not supported by this implementation.

Third pre-registered hypothesis (written 2026-11-09, before
``particles`` had ever been run on any task): ``particles`` beats
``ham_learned`` on ``assoc`` by more than seed noise and learns matter
friction above its 0.05 start there (binding needs clusters to settle).
It is at least as good as ``ham_learned`` on ``recall_first`` and
``count``, and *worse* on ``lag``, because it has no positional encoding
and "5 steps ago" can only be read off the dynamics. Whether it reaches
the LSTM on ``assoc`` (0.69) is left open.
(Result: largely not supported; see ARTICLE §4.1.)

Fourth pre-registered hypothesis (written 2026-11-09, after a 3-seed
*diagnostic* with embedding dim 32 gave ``assoc`` 0.98 / 0.46 / 0.99 for
dense birth, but before ``particles_dense`` had ever been run in this
benchmark): ``particles_dense`` (the particle model born dense,
``spawn_scale_init=0.1``) reaches a mean ``assoc`` accuracy above the
LSTM's at the same settings, over 5 seeds, with both run here. Because
the diagnostic showed a bimodal outcome, also report how many seeds
exceed 0.9. If the mean does not exceed the LSTM's, the diagnostic was
noise or depended on the embedding size.

Fifth pre-registered hypothesis (written 2026-11-10, before any model
had been evaluated at a length other than the one it was trained on).
Every earlier task is one a standard model solves, so the best outcome
was a tie. Conservation predicts something specific: memory that does not
fade with length. Train at T = 40, evaluate at T = 100, 200, 400
(``--eval-lengths``), 5 seeds:

* on ``recall_first``, ``ham_conservative`` keeps accuracy ≥ 0.9 at
  T = 400 and loses less from T = 40 to T = 400 than the LSTM and the
  transformer;
* ``ham_learned`` loses more than ``ham_conservative`` on
  ``recall_first`` (its friction, ~0.035 per token, erases a momentum
  memory over hundreds of tokens) but holds ``count`` and ``lag`` better
  than ``ham_conservative`` does.

No prediction is made for the LSTM versus the transformer. If the
conservative model degrades as fast as the LSTM, "conservation gives
length-robust memory" is not supported by this implementation (chaos in
the learned potential would be the first suspect).

(Result: not supported. ``ham_conservative`` is at chance by T = 100.)

Sixth pre-registered hypothesis (written 2026-11-10, before any model
with cyclic coordinates had been trained). The failure above has a
specific cause (``benchmarks/diagnostics/memory_probe.py``): every mode
has a restoring force, so the stored token is a *rotating* phase. At
T = 44 the trained readout scores 0.000 while a freshly fitted linear
readout scores 1.000. Noether's theorem names the cure: make H invariant
under translation of some coordinates and their momenta are exactly
conserved, a memory that does not rotate. ``ham_cyclic`` is
``ham_conservative`` with 16 of its 64 coordinates cyclic;
``ham_cyclic_learned`` is ``ham_learned`` with the same. Prediction, on
``recall_first``, 5 seeds, own readout (no probe):

* trained at T = 40, ``ham_cyclic`` scores ≥ 0.9 at T = 400 (where
  ``ham_conservative`` scored 0.127) and at least matches the LSTM there
  (0.865 ± 0.168);
* if it does not (the model may still prefer its rotating modes, which
  work equally well at one fixed length), then with training lengths
  drawn from 20–40 (``--train-len-min 20``) ``ham_cyclic`` reaches ≥ 0.9
  at T = 400 and ``ham_conservative`` does not.

If neither holds, a conserved momentum is not enough: the distractor
impulses would be the first suspect (they add to the conserved momentum
unless the model learns to zero them).

(Result: not supported as stated; own readout 0.18–0.35 at T = 400. But a
fresh linear readout recovers the token perfectly at T = 1000 on 10/10
runs, against 0.90 for the LSTM: the memory is conserved and the model
cannot read it, because distractor impulses drift the conserved momenta.)

Seventh pre-registered hypothesis (written 2026-11-10, before any model
with a cyclic charge had been trained). ``ham_charge`` is
``ham_cyclic_learned`` plus a selection rule: a content-derived charge
g(x) ∈ (0, 1) scales each token's impulse on the cyclic coordinates.
Prediction, ``recall_first``, trained on lengths 20–40, 5 seeds, the
model's *own* readout: accuracy ≥ 0.9 at T = 400, and at T = 1000 at
least the LSTM's under the same training (0.875 ± 0.084). If the charge
does not saturate for distractors (training at ≤ 40 tokens gives little
pressure to), the fallback condition is training lengths 20–100; report
both, and report the learned charges per token class.

(Result: not supported. The charge changed nothing (0.145 at T = 400).
``benchmarks/diagnostics/readout_swap.py`` shows the conserved part alone
derails the readout, and that the charge never learned to close:
distractor charges end anywhere from 0.01 to 0.98, because a small leak
costs nothing at the trained lengths.)

Eighth pre-registered hypothesis (written 2026-11-10, before any run with
a write cost). Add ``λ ·`` (mean squared impulse on the conserved
coordinates) to the loss (``--cyclic-write-cost λ``): the first token
pays once, distractors would pay 39 times, so their writes should go to
zero. Same criteria as hypothesis 7, ``recall_first``, trained on 20–40,
5 seeds, own readout, for ``ham_charge`` and ``ham_cyclic_learned`` at
λ = 0.1 and λ = 1.0 (both declared here; report both): accuracy ≥ 0.9 at
T = 400 and at least the LSTM's at T = 1000 (0.875 ± 0.084).

(Result: not supported, though in the right direction: at λ = 1.0
``ham_charge`` reaches 0.490 ± 0.215 at T = 400, from 0.145.)

Ninth pre-registered hypothesis (written 2026-11-10, before any run with
a net-force penalty). ``benchmarks/diagnostics/transfer_probe.py`` shows
the failure is entirely a *deterministic drift*: a readout fitted once at
T = 40 fails at long lengths, but the same readout on re-centred features
scores 1.000 at T = 1000 (4/4 runs). The random part of the distractor
impulses is harmless; their mean is not. So penalise only the mean:
``--cyclic-net-force μ`` adds ``μ · |mean impulse on the conserved
coordinates|²`` (mean over the batch and time) to the loss. Individual
writes stay free. Because training batches have different lengths, the
mean can only vanish for all of them if the distractor mean and the
first-token mean vanish separately. Same criteria as hypotheses 7 and 8,
``recall_first``, trained on 20–40, 5 seeds, own readout, for
``ham_cyclic_learned`` and ``ham_charge`` at μ = 10 and μ = 100 (both
declared): accuracy ≥ 0.9 at T = 400 and at least the LSTM's at T = 1000
(0.875 ± 0.084).

(Result: not supported; 0.32–0.40 at T = 400 at both strengths.)

Tenth pre-registered hypothesis (written 2026-11-10, before any model
with a ReLU charge had been trained). Three penalties (hypotheses 7–9)
each made the leak smaller and none made it small enough: over 1000
tokens a per-token drift must be below ~1e-4 of a write, and a sigmoid
charge or a batch-mean estimate cannot get there. ``ham_neutral`` is
``ham_charge`` with charge g(x) = relu(w·x + b), which can be exactly
zero, plus an L1 cost on the charge (``--cyclic-charge-cost``) that pays
for every charged token. Same criteria, ``recall_first``, trained on
20–40, 5 seeds, own readout, at cost 0 (control), 0.01, 0.1 and 1.0 (all
declared): accuracy ≥ 0.9 at T = 400 and at least the LSTM's at T = 1000
(0.875 ± 0.084) for at least one non-zero cost; the control at cost 0 is
predicted to fail like ``ham_charge``. Risk noted in advance: a first
token whose charge dies early cannot recover (zero gradient).

(Result: supported. Cost 0.1: 1.000 at every length to T = 10,000 on 5/5
seeds, distractor charges exactly zero; cost 1.0: 0.963 at T = 1000; the
cost-0 control fails (0.142); at cost 0.01 one seed fails, the risk
noted above. Best of 13 LSTM settings: 0.948 +- 0.071 at T = 10,000.)

Eleventh pre-registered hypothesis (written 2026-11-10, before either
new task had been run with any model). Hypothesis 10 passed on a task
where the item and the distractors come from different symbol sets, so
a charge that sees only the current token is enough by construction.
Two tasks probe that boundary; trained on 20–40, 5 seeds, own readout,
``ham_neutral`` at charge cost 0.1 against the LSTM at forget bias 3
and 8, evaluated at T = 400, 1000 and 4000:

* ``recall_mid``: the item (symbols 0–7) sits at a uniformly random
  position among distractors (symbols 8–15). Prediction: ``ham_neutral``
  ≥ 0.99 at every length (position should not matter to a per-token
  charge).
* ``recall_same``: every token is drawn from symbols 0–7 and the answer
  is the first one. Prediction: ``ham_neutral`` **fails**, below the
  better LSTM at T = 400, because no per-token charge can tell the first
  token from a later copy of the same symbol. If it passes, the model
  found a route I did not foresee and that must be explained.

(Result: supported, both halves. ``recall_mid``: 1.000 at every length;
LSTM 0.95 at T = 4000. ``recall_same``: 0.307 at T = 40 and chance
beyond; the LSTM is exact to T = 4000.)

Twelfth pre-registered hypothesis (written 2026-11-10, before any model
with a state-dependent charge had been trained). Hypothesis 11 located
the limit: on ``recall_same`` no charge that sees only the current token
can separate the first token from later copies. ``ham_exclusion`` makes
the charge depend on the conserved memory itself,
g = relu(w·x + b − u·n), where n is the mean squared conserved momentum
(the "occupation") just before the token arrives and u ≥ 0 is learned:
an occupied memory refuses further charge, an exclusion principle. When
g is exactly zero the momenta stay exactly conserved, as before.
Trained on 20–40, 5 seeds, own readout, charge cost 0 and 0.1 (both
declared), evaluated at T = 400, 1000 and 4000. Predictions:

* ``recall_same``: ≥ 0.99 at T = 4000 for at least one cost. The LSTM
  scores 1.000 there, so a tie is the best possible outcome.
* ``recall_first`` and ``recall_mid`` at the same cost: still ≥ 0.99 at
  T = 4000 (the new term must not cost what hypothesis 10 gained).

Risk noted in advance: training lengths of 20–40 only punish a leak
that matters within 40 tokens, so the blocked charge may be small rather
than exactly zero and show up only at T = 4000; the charge cost is the
pressure toward exact zero.

Measuring-stick change (2026-11-10): accuracy after a fixed 1500 steps
turned out to measure learning *speed* on ``assoc`` (see the audit
diagnostic in the results file). ``--eval-every N --target A`` now
records the first step at which accuracy on a fixed validation set
reaches ``A`` and stops there; ``--steps`` is the cap.

Run:

    python -m benchmarks.context_dynamics
    python -m benchmarks.context_dynamics --models lstm transformer ham_conservative ham_learned \
        --tasks recall_first count lag --seeds 0 1 2 3 4 --eval-lengths 100 200 400
    python -m benchmarks.context_dynamics --tasks assoc --models lstm transformer particles_dense \
        --steps 6000 --eval-every 100 --target 0.9 --seeds 0 1 2 3 4
    python -m benchmarks.context_dynamics --steps 3000 --seq-len 60 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from yaum.models.hamiltonian import HamiltonianCharModel
from yaum.models.particles import HamiltonianParticleModel
from yaum.models.rnn import RNNCharModel
from yaum.models.transformer import TransformerCharModel

N_SYMBOLS = 8


# ---- tasks -----------------------------------------------------------------
# Each returns (x, y, mask): token ids (B, T), targets (B, T), and a bool mask
# of which positions are scored.

def task_recall_first(batch, T, gen):
    x = torch.randint(N_SYMBOLS, 2 * N_SYMBOLS, (batch, T), generator=gen)
    first = torch.randint(0, N_SYMBOLS, (batch,), generator=gen)
    x[:, 0] = first
    y = torch.zeros(batch, T, dtype=torch.long)
    y[:, -1] = first
    mask = torch.zeros(batch, T, dtype=torch.bool)
    mask[:, -1] = True
    return x, y, mask


def task_recall_mid(batch, T, gen):
    """Like ``recall_first`` but the item is at a random position."""
    x = torch.randint(N_SYMBOLS, 2 * N_SYMBOLS, (batch, T), generator=gen)
    item = torch.randint(0, N_SYMBOLS, (batch,), generator=gen)
    pos = torch.randint(0, T - 1, (batch,), generator=gen)
    x[torch.arange(batch), pos] = item
    y = torch.zeros(batch, T, dtype=torch.long)
    y[:, -1] = item
    mask = torch.zeros(batch, T, dtype=torch.bool)
    mask[:, -1] = True
    return x, y, mask


def task_recall_same(batch, T, gen):
    """Recall the first token when distractors use the *same* symbols."""
    x = torch.randint(0, N_SYMBOLS, (batch, T), generator=gen)
    y = torch.zeros(batch, T, dtype=torch.long)
    y[:, -1] = x[:, 0]
    mask = torch.zeros(batch, T, dtype=torch.bool)
    mask[:, -1] = True
    return x, y, mask


def task_count(batch, T, gen, max_depth=N_SYMBOLS - 1):
    x = torch.zeros(batch, T, dtype=torch.long)
    y = torch.zeros(batch, T, dtype=torch.long)
    depth = torch.zeros(batch, dtype=torch.long)
    for t in range(T):
        up = torch.rand(batch, generator=gen) < 0.5
        up = torch.where(depth == 0, torch.ones_like(up), up)
        up = torch.where(depth == max_depth, torch.zeros_like(up), up)
        depth = depth + torch.where(up, 1, -1)
        x[:, t] = torch.where(up, 0, 1)
        y[:, t] = depth
    return x, y, torch.ones(batch, T, dtype=torch.bool)


def task_lag(batch, T, gen, lag=5):
    x = torch.randint(0, N_SYMBOLS, (batch, T), generator=gen)
    y = torch.zeros_like(x)
    y[:, lag:] = x[:, :-lag]
    mask = torch.zeros(batch, T, dtype=torch.bool)
    mask[:, lag:] = True
    return x, y, mask


def task_assoc(batch, T, gen, pairs=4):
    """Keys are tokens 0..7, values tokens 8..15; targets are value ids 0..7."""
    del T  # length is fixed by the number of pairs
    keys = torch.argsort(torch.rand(batch, N_SYMBOLS, generator=gen), dim=1)[:, :pairs]
    values = torch.randint(0, N_SYMBOLS, (batch, pairs), generator=gen)
    which = torch.randint(0, pairs, (batch,), generator=gen)
    rows = torch.arange(batch)
    x = torch.empty(batch, 2 * pairs + 1, dtype=torch.long)
    x[:, 0:-1:2] = keys
    x[:, 1:-1:2] = values + N_SYMBOLS
    x[:, -1] = keys[rows, which]
    y = torch.zeros_like(x)
    y[:, -1] = values[rows, which]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, -1] = True
    return x, y, mask


TASKS = {
    "recall_first": task_recall_first,
    "recall_mid": task_recall_mid,
    "recall_same": task_recall_same,
    "count": task_count,
    "lag": task_lag,
    "assoc": task_assoc,
}
VOCAB_IN = 2 * N_SYMBOLS


# ---- models ----------------------------------------------------------------

class Wrapped(nn.Module):
    """nn.Embedding + a YAUM sequence model (which takes embedded input)."""

    def __init__(self, core: nn.Module, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_IN, embedding_dim)
        self.core = core

    def forward(self, x):
        logits, _ = self.core(self.embed(x), self.core.init_hidden(x.shape[0]))
        return logits


def set_forget_bias(lstm: nn.LSTM, value: float | None) -> None:
    """Set the LSTM forget-gate bias (Gers et al. 2000; Jozefowicz et al. 2015).

    PyTorch's default init gives a forget gate ≈ 0.5, so information decays
    like 0.5^T. Over 40 steps that erases the first token (≈1e-12), which made
    the LSTM sit at chance on ``recall_first`` in the first runs. ``1.0`` is
    the standard fair-baseline setting; ``None`` keeps PyTorch's init.
    """
    if value is None:
        return
    h = lstm.hidden_size
    with torch.no_grad():
        for layer in range(lstm.num_layers):
            # Gate order in PyTorch: input, forget, cell, output.
            getattr(lstm, f"bias_ih_l{layer}")[h:2 * h].fill_(value)
            getattr(lstm, f"bias_hh_l{layer}")[h:2 * h].zero_()


def build(name: str, args) -> Wrapped:
    if name == "lstm":
        core = RNNCharModel(N_SYMBOLS, args.embedding_dim, args.hidden_dim, 1)
        set_forget_bias(core.lstm, args.lstm_forget_bias)
    elif name == "transformer":
        # Sinusoidal positions, so it can be evaluated at unseen lengths.
        core = TransformerCharModel(
            N_SYMBOLS, args.embedding_dim, args.hidden_dim, 2,
            n_heads=2, max_context=2048, dropout=0.0,
        )
    elif name == "ham_conservative":
        core = HamiltonianCharModel(
            N_SYMBOLS, args.embedding_dim, args.hidden_dim, args.substeps,
            dt=args.dt, dissipation="none",
        )
    elif name in ("ham_cyclic", "ham_cyclic_learned", "ham_charge", "ham_neutral", "ham_exclusion", "ham_exclusion"):
        core = HamiltonianCharModel(
            N_SYMBOLS, args.embedding_dim, args.hidden_dim, args.substeps,
            dt=args.dt, dissipation="none" if name == "ham_cyclic" else "learned",
            gamma_init=args.gamma_init, n_cyclic=args.n_cyclic,
            cyclic_charge={"ham_charge": True, "ham_neutral": "neutral",
                           "ham_exclusion": "exclusion"}.get(name, False),
        )
    elif name in ("ham_learned", "ham_geometry"):
        core = HamiltonianCharModel(
            N_SYMBOLS, args.embedding_dim, args.hidden_dim, args.substeps,
            dt=args.dt, dissipation="learned", gamma_init=args.gamma_init,
            geometry="dynamic" if name == "ham_geometry" else "static",
        )
    elif name == "particles":
        core = HamiltonianParticleModel(
            N_SYMBOLS, args.embedding_dim, args.particle_dim, args.substeps,
            charge_dim=args.charge_dim, dt=args.dt, dissipation="learned",
            gamma_init=args.gamma_init,
        )
    elif name == "particles_dense":
        # "Big Bang" birth: every particle starts within interaction range.
        core = HamiltonianParticleModel(
            N_SYMBOLS, args.embedding_dim, args.particle_dim, args.substeps,
            charge_dim=args.charge_dim, dt=args.dt, dissipation="learned",
            gamma_init=args.gamma_init, spawn_scale_init=0.1,
        )
    else:
        raise ValueError(name)
    return Wrapped(core, args.embedding_dim)


# ---- training --------------------------------------------------------------

@dataclass
class Result:
    task: str
    model: str
    seed: int
    accuracy: float
    params: int
    wall_s: float
    gamma_median: float = float("nan")
    gamma_q90: float = float("nan")
    geom_gamma_median: float = float("nan")
    stiffness_median: float = float("nan")
    steps_to_target: float = float("nan")  # nan = target never reached (or not measured)
    steps_run: int = 0
    length_acc: dict = field(default_factory=dict)  # eval length -> accuracy


def masked_loss_acc(logits, y, mask):
    flat = mask.reshape(-1)
    lg = logits.reshape(-1, logits.shape[-1])[flat]
    tgt = y.reshape(-1)[flat]
    loss = F.cross_entropy(lg, tgt)
    acc = (lg.argmax(-1) == tgt).float().mean()
    return loss, acc


def chunked_accuracy(model, x, y, mask, max_tokens: int = 40_000) -> float:
    """Masked accuracy, evaluated in chunks so long sequences fit in memory.

    Every model returns logits for all positions, so 1024 sequences of
    10 000 tokens at once needs several GB.
    """
    chunk = max(1, max_tokens // x.shape[1])
    correct, total = 0.0, 0
    for i in range(0, x.shape[0], chunk):
        m = mask[i:i + chunk]
        logits = model(x[i:i + chunk])
        correct += float((logits.argmax(-1)[m] == y[i:i + chunk][m]).sum())
        total += int(m.sum())
    return correct / max(total, 1)


def run(task: str, model_name: str, seed: int, args) -> Result:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    model = build(model_name, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    make = TASKS[task]
    # Validation set for steps-to-target: fixed, and distinct from the
    # final evaluation set, so stopping on it does not bias the report.
    val = make(512, args.seq_len, torch.Generator().manual_seed(20_000 + seed))
    steps_to_target, steps_run = float("nan"), 0
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        length = args.seq_len
        if args.train_len_min:
            top = args.train_len_max or args.seq_len
            length = int(torch.randint(args.train_len_min, top + 1, (1,), generator=gen))
        x, y, mask = make(args.batch_size, length, gen)
        loss, _ = masked_loss_acc(model(x), y, mask)
        write = getattr(model.core, "last_cyclic_write", None)
        if args.cyclic_write_cost and write is not None:
            loss = loss + args.cyclic_write_cost * write
        charge_mean = getattr(model.core, "last_charge_mean", None)
        if args.cyclic_charge_cost and charge_mean is not None:
            loss = loss + args.cyclic_charge_cost * charge_mean
        net = getattr(model.core, "last_cyclic_net_force", None)
        if args.cyclic_net_force and net is not None:
            loss = loss + args.cyclic_net_force * net
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps_run = step
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                _, val_acc = masked_loss_acc(model(val[0]), val[1], val[2])
            model.train()
            if float(val_acc) >= args.target:
                steps_to_target = step
                break
    wall = time.perf_counter() - t0

    model.eval()
    length_acc = {}
    with torch.no_grad():
        x, y, mask = make(1024, args.seq_len, torch.Generator().manual_seed(10_000 + seed))
        _, acc = masked_loss_acc(model(x), y, mask)
        for length in args.eval_lengths:
            xl, yl, ml = make(1024, length, torch.Generator().manual_seed(30_000 + seed + length))
            length_acc[length] = chunked_accuracy(model, xl, yl, ml)
    result = Result(
        task=task,
        model=model_name,
        seed=seed,
        accuracy=float(acc),
        params=sum(p.numel() for p in model.parameters()),
        wall_s=wall,
        steps_to_target=steps_to_target,
        steps_run=steps_run,
        length_acc=length_acc,
    )
    core = model.core
    if isinstance(core, (HamiltonianCharModel, HamiltonianParticleModel)) and core.dissipation == "learned":
        # Matter friction only; geometry friction is reported separately.
        g = core.gamma.detach()
        result.gamma_median = float(g.median())
        result.gamma_q90 = float(torch.quantile(g, 0.9))
    if isinstance(core, HamiltonianCharModel) and core.geometry == "dynamic":
        result.geom_gamma_median = float(core.gamma_geometry.detach().median())
        result.stiffness_median = float(core.stiffness.detach().median())
    return result


def _ms(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def report(results: list[Result], args) -> str:
    lines = [
        (
            f"\n## Run steps={args.steps} T={args.seq_len} hidden={args.hidden_dim} "
            f"substeps={args.substeps} dt={args.dt} seeds={list(args.seeds)} "
            f"lstm_forget_bias={args.lstm_forget_bias}"
            + (f" train_len={args.train_len_min}-{args.train_len_max or args.seq_len}" if args.train_len_min else "")
            + (f" cyclic_write_cost={args.cyclic_write_cost}" if args.cyclic_write_cost else "")
            + (f" cyclic_net_force={args.cyclic_net_force}" if args.cyclic_net_force else "")
            + (f" cyclic_charge_cost={args.cyclic_charge_cost}" if args.cyclic_charge_cost else "") + "\n"
        ),
        ("| task | model | accuracy | params | learned γ median (90th pct) "
        "| geometry γ / stiffness λ | wall (s) |"),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for model in MODELS:
            rs = [r for r in results if r.task == task and r.model == model]
            if not rs:
                continue
            acc_m, acc_s = _ms([r.accuracy for r in rs])
            gm, _ = _ms([r.gamma_median for r in rs])
            gq, _ = _ms([r.gamma_q90 for r in rs])
            gamma = f"{gm:.3f} ({gq:.3f})" if math.isfinite(gm) else "—"
            geo_g, _ = _ms([r.geom_gamma_median for r in rs])
            stiff, _ = _ms([r.stiffness_median for r in rs])
            geo = f"{geo_g:.3f} / {stiff:.3f}" if math.isfinite(geo_g) else "—"
            lines.append(
                f"| {task} | `{model}` | {acc_m:.3f} ± {acc_s:.3f} | {rs[0].params:,} "
                f"| {gamma} | {geo} | {_ms([r.wall_s for r in rs])[0]:.0f} |"
            )
    if args.eval_every:
        lines += [
            "",
            (f"Steps to reach {args.target:.2f} on a fixed validation set "
            f"(checked every {args.eval_every} steps, cap {args.steps}; training stops there):"),
            "",
            "| task | model | reached | steps to target (those that reached) | per seed |",
            "|---|---|---:|---:|---|",
        ]
        for task in TASKS:
            for model in MODELS:
                rs = [r for r in results if r.task == task and r.model == model]
                if not rs:
                    continue
                hit = [r.steps_to_target for r in rs if math.isfinite(r.steps_to_target)]
                mean = f"{statistics.mean(hit):.0f}" if hit else "—"
                per = ", ".join(
                    f"{r.steps_to_target:.0f}" if math.isfinite(r.steps_to_target)
                    else f">{args.steps} ({r.accuracy:.2f})" for r in rs
                )
                lines.append(f"| {task} | `{model}` | {len(hit)}/{len(rs)} | {mean} | {per} |")
    if args.eval_lengths:
        lengths = list(args.eval_lengths)
        lines += [
            "",
            (f"Length generalisation: trained at T={args.seq_len}, evaluated at other lengths "
            "(mean ± std over seeds):"),
            "",
            "| task | model | T=" + f"{args.seq_len} | " + " | ".join(f"T={n}" for n in lengths) + " |",
            "|---|---|" + "---:|" * (len(lengths) + 1),
        ]
        for task in TASKS:
            for model in MODELS:
                rs = [r for r in results if r.task == task and r.model == model]
                if not rs:
                    continue
                cells = ["{:.3f} ± {:.3f}".format(*_ms([r.accuracy for r in rs]))]
                cells += ["{:.3f} ± {:.3f}".format(*_ms([r.length_acc[n] for r in rs])) for n in lengths]
                lines.append(f"| {task} | `{model}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Chance accuracy: recall_first, lag and assoc 0.125; count ≈ 0.2. "
        f"Learned γ starts at {args.gamma_init}."
    )
    lines.append("")
    return "\n".join(lines)


def _float_or_none(text: str):
    return None if text.lower() == "none" else float(text)


MODELS = (
    "lstm", "transformer", "ham_conservative", "ham_learned", "ham_geometry",
    "ham_cyclic", "ham_cyclic_learned", "ham_charge", "ham_neutral", "ham_exclusion",
    "particles", "particles_dense",
)
# The particle models cost O(T³) per sequence (tens of minutes per run at
# T = 40 on CPU), so they are opt-in via --models.
DEFAULT_MODELS = ("lstm", "transformer", "ham_conservative", "ham_learned", "ham_geometry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--particle-dim", type=int, default=32,
                        help="position/momentum dimension per particle")
    parser.add_argument("--charge-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--lstm-forget-bias", type=_float_or_none, default=3.0,
        help="LSTM forget-gate bias init (runs before 2026-11-08 evening used "
             "PyTorch's default, equivalent to --lstm-forget-bias none).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tasks", nargs="+", default=["recall_first", "count", "lag", "assoc"],
                        choices=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(MODELS))
    parser.add_argument("--n-cyclic", type=int, default=16,
                        help="cyclic (symmetry-protected) coordinates in ham_cyclic*")
    parser.add_argument("--train-len-min", type=int, default=0,
                        help="if > 0, draw each training batch's length from [this, --seq-len]")
    parser.add_argument("--cyclic-write-cost", type=float, default=0.0,
                        help="weight of the mean squared impulse on conserved coordinates in the loss")
    parser.add_argument("--cyclic-charge-cost", type=float, default=0.0,
                        help="weight of the mean charge (L1) in the loss; with ham_neutral it makes tokens neutral")
    parser.add_argument("--cyclic-net-force", type=float, default=0.0,
                        help="weight of |mean impulse on conserved coordinates|^2 in the loss")
    parser.add_argument("--train-len-max", type=int, default=0,
                        help="upper end of the training-length range (default: --seq-len)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="check a fixed validation set every N steps and stop at --target (0 = off)")
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--eval-lengths", type=int, nargs="*", default=[],
                        help="also evaluate at these sequence lengths after training")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "context_dynamics_results.md"),
        help="Markdown file to append the report to ('' to skip).",
    )
    args = parser.parse_args()

    results = []
    for task in args.tasks:
        for model in args.models:
            for seed in args.seeds:
                r = run(task, model, seed, args)
                print(
                    f"{task:13s} {model:17s} seed={seed} acc={r.accuracy:.3f} "
                    f"γ_med={r.gamma_median:.3f} ({r.wall_s:.0f}s)",
                    flush=True,
                )
                results.append(r)

    text = report(results, args)
    print(text)
    if args.output:
        new = not os.path.exists(args.output)
        with open(args.output, "a", encoding="utf-8") as fh:
            if new:
                fh.write("# Context-dynamics results\n\nAppended by "
                         "`python -m benchmarks.context_dynamics`.\n")
            fh.write(text)
        print(f"Appended report to {args.output}")


if __name__ == "__main__":
    main()
