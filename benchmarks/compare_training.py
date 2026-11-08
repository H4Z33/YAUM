"""Head-to-head comparison of embedding update rules.

Everything except how the embedding matrix ``E`` is updated is held
fixed: the synthetic corpus, the model (``RNNCharModel``), the weight
optimiser (Adam on ``W``), the number of steps, and the held-out test
set (every non-overlapping window of the test split, identical for all
variants). Each variant runs over several seeds; we report mean ± std.

Variants:

    frozen_E          E stays at its random initialisation (control:
                      how much of the learning is just W?)
    adam_E            E is an ordinary parameter, Adam lr=1e-3 (same lr as W)
    adam_E_tuned      Adam on E with lr=1e-2, the best of a sweep over
                      {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} on this setup. Any
                      claim that YAUM "beats Adam" must be made against
                      this row, not the untuned one.
    yaum_static       leapfrog, uniform mass 1, no friction
                      (the original YAUM configuration)
    yaum_fisher       leapfrog, RMS-Fisher mass, no friction
    yaum_damped       trainer defaults: RMS-Fisher mass, leapfrog core
                      wrapped in friction (Langevin at T=0), adaptive dt

``max |dH/H|`` is the worst *one-step* relative energy error of the
conservative core (same batch, same weights, before vs after the step).
It is a property of the integrator. Earlier versions of this benchmark
reported drift against the first step, which mostly measured how much
the loss had fallen.

Run:

    python -m benchmarks.compare_training
    python -m benchmarks.compare_training --num-steps 1000 --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

from yaum.core.trainer import Trainer
from yaum.core.utils import get_batch
from yaum.models.rnn import RNNCharModel

VOCAB = 32


def _make_corpus(n_tokens: int = 4096, vocab_size: int = 16) -> torch.Tensor:
    """Period-11 cycle with a stutter: predictable but non-trivial."""
    base = [(i * 7 + (i // 11)) % vocab_size for i in range(n_tokens)]
    return torch.tensor(base, dtype=torch.long)


def _split(data: torch.Tensor, test_frac: float = 0.1):
    cut = int(len(data) * (1.0 - test_frac))
    return data[:cut], data[cut:]


def _fixed_test_windows(test_data: torch.Tensor, context_window: int):
    """All non-overlapping (x, y) windows of the test split."""
    n = (len(test_data) - 1) // context_window
    idx = torch.arange(n).unsqueeze(1) * context_window + torch.arange(context_window)
    return test_data[idx], test_data[idx + 1]


@torch.no_grad()
def held_out_loss(model, E, windows) -> float:
    x, y = windows
    was_training = model.training
    model.eval()
    logits, _ = model(F.embedding(x, E.detach()), model.init_hidden(x.shape[0]))
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
    if was_training:
        model.train()
    return float(loss)


@dataclass
class RunResult:
    name: str
    seed: int
    wall_s: float
    test_loss: float
    max_step_drift: float = float("nan")
    E_moved: float = float("nan")  # ||E - E0|| / ||E0||


def _cfg(args) -> dict:
    return {
        "model_type": "rnn",
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": 1,
        "context_window": args.context_window,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "learning_rate_W": args.lr_w,
        "gradient_clip_norm_W": 1.0,
    }


def run_plain(name, train_data, windows, cfg, seed, lr_e) -> RunResult:
    """frozen_E (lr_e=None) or adam_E."""
    torch.manual_seed(seed)
    model = RNNCharModel(VOCAB, cfg["embedding_dim"], cfg["hidden_dim"], cfg["n_layers"])
    E0 = torch.randn(VOCAB, cfg["embedding_dim"])
    E = E0.clone().requires_grad_(lr_e is not None)
    opt_w = optim.Adam(model.parameters(), lr=cfg["learning_rate_W"])
    opt_e = optim.Adam([E], lr=lr_e) if lr_e is not None else None
    model.train()
    t0 = time.perf_counter()
    for _ in range(cfg["num_steps"]):
        x, y = get_batch(train_data, cfg["context_window"], cfg["batch_size"], "cpu")
        logits, _ = model(F.embedding(x, E), model.init_hidden(cfg["batch_size"]))
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt_w.zero_grad(set_to_none=True)
        if opt_e is not None:
            opt_e.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm_W"])
        opt_w.step()
        if opt_e is not None:
            opt_e.step()
    wall = time.perf_counter() - t0
    return RunResult(
        name=name,
        seed=seed,
        wall_s=wall,
        test_loss=held_out_loss(model, E, windows),
        E_moved=float((E.detach() - E0).norm() / E0.norm()),
    )


def run_yaum(name, train_data, test_data, windows, cfg, seed, overrides) -> RunResult:
    torch.manual_seed(seed)
    config = dict(cfg)
    config.update(
        eval_interval=cfg["num_steps"],  # telemetry once at the end; we eval below
        eval_iters=1,
        results_dir=tempfile.mkdtemp(prefix="yaum_bench_"),
        snapshot_interval=0,
        snapshot_on_shift=False,
        reversibility_check_interval=0,
        safe_speed=False,
        step_delay=0.0,
    )
    config.update(overrides)
    trainer = Trainer(config)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.setup(train_data, test_data, {}, {}, VOCAB, torch.ones(VOCAB, 1))
        E0 = trainer.E.detach().clone()
        t0 = time.perf_counter()
        for _ in trainer.train():
            pass
        wall = time.perf_counter() - t0
    drift = trainer.get_history()["H_drift"]
    return RunResult(
        name=name,
        seed=seed,
        wall_s=wall,
        test_loss=held_out_loss(trainer.model, trainer.E, windows),
        max_step_drift=max(drift) if drift else float("nan"),
        E_moved=float((trainer.E.detach() - E0).norm() / E0.norm()),
    )


YAUM_VARIANTS = {
    "yaum_static": dict(
        integrator="leapfrog", mass_mode="static", adaptive_dt=False, dt=0.05
    ),
    "yaum_fisher": dict(integrator="leapfrog", adaptive_dt=False, dt=0.01),
    "yaum_damped": {},  # trainer defaults
}


def _mean_std(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def format_report(results: list[RunResult], args) -> str:
    names = list(dict.fromkeys(r.name for r in results))
    lines = [
        f"\n## Run steps={args.num_steps} seeds={list(args.seeds)}\n",
        "| variant | held-out loss | ‖E−E₀‖/‖E₀‖ | max one-step \\|ΔH/H\\| | wall (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    summary = {}
    for name in names:
        rs = [r for r in results if r.name == name]
        loss_m, loss_s = _mean_std([r.test_loss for r in rs])
        moved_m, _ = _mean_std([r.E_moved for r in rs])
        drift_m, _ = _mean_std([r.max_step_drift for r in rs])
        wall_m, _ = _mean_std([r.wall_s for r in rs])
        summary[name] = loss_m
        drift = f"{drift_m:.1e}" if math.isfinite(drift_m) else "—"
        lines.append(
            f"| `{name}` | {loss_m:.4f} ± {loss_s:.4f} | {moved_m:.2f} | {drift} | {wall_m:.1f} |"
        )
    best = min(summary, key=summary.get)
    lines.append("")
    lines.append(f"Lowest held-out loss: `{best}`.")
    if "frozen_E" in summary and "yaum_static" in summary:
        lines.append(
            f"`yaum_static` vs `frozen_E`: {summary['yaum_static'] - summary['frozen_E']:+.4f} "
            "(negative = the dynamics on E help at all)."
        )
    stds = {
        name: _mean_std([r.test_loss for r in results if r.name == name])[1]
        for name in names
    }
    if "adam_E_tuned" in summary and "yaum_damped" in summary:
        diff = summary["yaum_damped"] - summary["adam_E_tuned"]
        noise = max(stds["yaum_damped"], stds["adam_E_tuned"])
        verdict = (
            "within seed noise — a tie" if abs(diff) <= noise
            else ("YAUM better" if diff < 0 else "Adam better")
        )
        lines.append(
            f"`yaum_damped` vs `adam_E_tuned`: {diff:+.4f} (seed std ≈ {noise:.4f}; {verdict})."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--context-window", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-w", type=float, default=1e-3)
    parser.add_argument("--lr-e", type=float, default=1e-3)
    parser.add_argument("--lr-e-tuned", type=float, default=1e-2)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "results.md"),
        help="Markdown file to append the report to ('' to skip).",
    )
    args = parser.parse_args()

    cfg = _cfg(args)
    train_data, test_data = _split(_make_corpus(8192, VOCAB), test_frac=0.1)
    windows = _fixed_test_windows(test_data, cfg["context_window"])

    results: list[RunResult] = []
    for seed in args.seeds:
        print(f"seed {seed}...", flush=True)
        results.append(run_plain("frozen_E", train_data, windows, cfg, seed, None))
        results.append(run_plain("adam_E", train_data, windows, cfg, seed, args.lr_e))
        results.append(
            run_plain("adam_E_tuned", train_data, windows, cfg, seed, args.lr_e_tuned)
        )
        for name, overrides in YAUM_VARIANTS.items():
            results.append(
                run_yaum(name, train_data, test_data, windows, cfg, seed, overrides)
            )

    report = format_report(results, args)
    print(report)
    if args.output:
        with open(args.output, "a", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Appended report to {args.output}")


if __name__ == "__main__":
    main()
