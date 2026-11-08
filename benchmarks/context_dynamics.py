"""Does a sequence model want to conserve energy in context time?

Compares, on three synthetic tasks, an LSTM against two variants of
``HamiltonianCharModel`` (hidden state = phase point moving under a
learned energy as tokens arrive):

    ham_conservative   friction γ = 0: a purely Hamiltonian model
    ham_learned        per-mode friction γ_i learned with the weights
    ham_geometry       ham_learned + dynamic geometry: the landscape is
                       bent by the content as it is read (joint
                       matter/geometry Hamiltonian; resets every sequence)

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

Run:

    python -m benchmarks.context_dynamics
    python -m benchmarks.context_dynamics --steps 3000 --seq-len 60 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from yaum.models.hamiltonian import HamiltonianCharModel
from yaum.models.rnn import RNNCharModel

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
    elif name == "ham_conservative":
        core = HamiltonianCharModel(
            N_SYMBOLS, args.embedding_dim, args.hidden_dim, args.substeps,
            dt=args.dt, dissipation="none",
        )
    elif name in ("ham_learned", "ham_geometry"):
        core = HamiltonianCharModel(
            N_SYMBOLS, args.embedding_dim, args.hidden_dim, args.substeps,
            dt=args.dt, dissipation="learned", gamma_init=args.gamma_init,
            geometry="dynamic" if name == "ham_geometry" else "static",
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


def masked_loss_acc(logits, y, mask):
    flat = mask.reshape(-1)
    lg = logits.reshape(-1, logits.shape[-1])[flat]
    tgt = y.reshape(-1)[flat]
    loss = F.cross_entropy(lg, tgt)
    acc = (lg.argmax(-1) == tgt).float().mean()
    return loss, acc


def run(task: str, model_name: str, seed: int, args) -> Result:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    model = build(model_name, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    make = TASKS[task]
    t0 = time.perf_counter()
    for _ in range(args.steps):
        x, y, mask = make(args.batch_size, args.seq_len, gen)
        loss, _ = masked_loss_acc(model(x), y, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    wall = time.perf_counter() - t0

    eval_gen = torch.Generator().manual_seed(10_000 + seed)
    model.eval()
    with torch.no_grad():
        x, y, mask = make(1024, args.seq_len, eval_gen)
        _, acc = masked_loss_acc(model(x), y, mask)
    result = Result(
        task=task,
        model=model_name,
        seed=seed,
        accuracy=float(acc),
        params=sum(p.numel() for p in model.parameters()),
        wall_s=wall,
    )
    core = model.core
    if isinstance(core, HamiltonianCharModel) and core.dissipation == "learned":
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
            f"lstm_forget_bias={args.lstm_forget_bias}\n"
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
    lines.append("")
    lines.append(
        "Chance accuracy: recall_first, lag and assoc 0.125; count ≈ 0.2. "
        f"Learned γ starts at {args.gamma_init}."
    )
    lines.append("")
    return "\n".join(lines)


def _float_or_none(text: str):
    return None if text.lower() == "none" else float(text)


MODELS = ("lstm", "ham_conservative", "ham_learned", "ham_geometry")


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
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--lstm-forget-bias", type=_float_or_none, default=1.0,
        help="LSTM forget-gate bias init (runs before 2026-11-08 evening used "
             "PyTorch's default, equivalent to --lstm-forget-bias none).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
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
