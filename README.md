# YAUM — Yet Another Universal Model

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-lightgrey.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

**Language-model training and inference through the lens of classical mechanics.**

YAUM is a research codebase built around one hypothesis: neural networks need
not just data and mechanics but *rules*, and Hamiltonian mechanics — energy,
momentum, conservation, controlled dissipation — is a good candidate for the
rules of a model's substrate.

It explores that idea in two places:

1. **Physics in training time** (`yaum/core/`). Token embeddings `E` are
   particles with momentum `P` and mass `m`, moved by a symplectic integrator
   with the loss as potential energy, while the network weights `W` are
   trained with Adam.
2. **Physics in context time** (`yaum/models/hamiltonian.py`). The model's
   hidden state is a phase point `(q, p)` that evolves under a *learned*
   energy function as tokens arrive: each token is an impulse, and between
   tokens the state follows Hamiltonian flow with a learned, per-mode friction.
   This is the "a model is a small universe that moves with context as time"
   version of the idea.

See [`docs/ARTICLE.md`](docs/ARTICLE.md) for the reasoning and the measured
results — including the ones that did not go the project's way.

## What the measurements say so far

* **Training time.** With a well-scaled mass (RMS of the force, see
  `yaum/core/geometry.py`) and friction, symplectic updates of `E` tie a
  *tuned* Adam on the same embeddings (held-out loss 0.170 vs 0.171, seed std
  0.007) and clearly beat leaving `E` frozen (0.221). They do not beat Adam.
  See `benchmarks/results.md`.
* **Context time.** A Hamiltonian hidden state with learned per-mode
  friction solves long recall, bounded counting and lag copying on
  synthetic tasks (1.00 / 1.00 / 0.996, 3 seeds), and the friction it
  learns orders the tasks by how much forgetting they need
  (0.035 < 0.085 < 0.15 < 0.24). A fairly initialised LSTM matches or
  beats it everywhere, though, and clearly on in-context key→value binding
  (0.69 vs 0.40). A first "dynamic geometry" version failed to train
  (exploding gradients). A many-body version, where every token is a
  particle with content-dependent attraction and repulsion
  (`yaum/models/particles.py`), can solve binding when particles are born
  close together, but learns it far more slowly than a small transformer
  and has not yet learned the 40-token tasks. See `docs/ARTICLE.md` §4.

* **Length generalisation** (trained on 40 tokens, tested up to 1000).
  With every coordinate oscillating, the Hamiltonian memory is a rotating
  phase and its readout fails off the training length. Making some
  coordinates symmetric (cyclic) gives exactly conserved momenta
  (Noether): the first token is then perfectly recoverable from the
  state at 1000 tokens (fresh linear readout 1.000 on 10/10 runs; LSTM
  0.90), but the model's own readout still fails by 400 tokens, where
  the LSTM scores 0.93. See `docs/ARTICLE.md` §4.2.

## Key features

* **Symplectic integrators:** leapfrog, Yoshida 4th order, Langevin (OBABO,
  with `temperature=0` giving pure damping), rRESPA multi-time-step.
* **Damped dynamics by default:** a conservative leapfrog core wrapped in
  friction, so embeddings can settle instead of oscillating forever.
* **RMS-Fisher mass:** `m = sqrt(EMA[F²]) + ε`, giving O(1) acceleration at
  any gradient scale.
* **Honest energy diagnostic:** `ΔH/H` is the one-step energy error of the
  integrator's conservative core (same batch, same weights), which drives
  the adaptive `dt`.
* **Model-agnostic force field:** RNN, Transformer, and the Hamiltonian
  sequence model all plug into the same trainer.
* **Gradio UI** for configuration, live telemetry, checkpoints and sampling.
* **Safe checkpoints:** loaded with `weights_only=True`, so a malicious file
  cannot run code.

## Installation

```bash
git clone https://github.com/H4Z33/YAUM.git
cd YAUM
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # NVIDIA GPU (CUDA 12.4 PyTorch)
# or
pip install -r requirements-cpu.txt   # CPU-only Linux, macOS (incl. Apple Silicon)
```

Other CUDA versions: install PyTorch from <https://pytorch.org/get-started/locally/>
first, then `pip install -r requirements-common.txt`.

## How to run

### One-liner via uv (recommended)

The scripts install [`uv`](https://github.com/astral-sh/uv) on first run,
create a Python 3.12 environment, install dependencies (CUDA wheels if
`nvidia-smi` is found, CPU wheels otherwise on Linux/macOS), and launch the UI.

```bash
./run.sh                # Linux / macOS
.\run.ps1               # Windows PowerShell
```

### Manual

```bash
python -m yaum.ui.app                          # http://127.0.0.1:7860
python -m yaum.ui.app --server-name 0.0.0.0    # reachable on your LAN
python -m yaum.ui.app --share                  # public gradio.live link
```

The UI is local-only by default. Only use `--share` on a machine you are
happy to expose: the UI reads files by server-side path.

### Benchmarks

```bash
python -m benchmarks.compare_training      # embedding update rules (training time)
python -m benchmarks.context_dynamics      # LSTM vs Hamiltonian state (context time)
python -m pytest -q                        # test suite
```

## Basic usage

1. **Setup & Config tab:** pick a `.txt` training file; choose the model
   family (`rnn`, `transformer`, `hamiltonian`, `particles`), sizes, integrator, friction,
   `dt` and mass mode; then click **Prepare Data & Model / Apply Config**, or
   load a checkpoint.
2. **Training tab:** start/stop training; watch losses, `|ΔH/H|`, `dt`,
   momentum and the other observables.
3. **Inference tab:** generate text with a prompt, length and temperature.

## License

YAUM is licensed under the **PolyForm Noncommercial License 1.0.0**: you may
use, modify and share it for non-commercial purposes. Commercial use requires
a separate license — contact [H4Z33 on GitHub](https://github.com/H4Z33/YAUM).
See [LICENSE](LICENSE).

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Before changing code, read [CLAUDE.md](CLAUDE.md) (layout, contracts,
research hygiene) and [docs/LESSONS.md](docs/LESSONS.md) (past mistakes and
the open backlog) — they are written for human and AI contributors alike.
Contributions are accepted under the same PolyForm Noncommercial license.

### Contributor License Agreement (CLA)

By contributing to this project, you agree that:

1. All contributions you make are licensed under the
   [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
2. You **do not retain rights** to the code after contributing; the rights
   remain with the project owner.
3. Commercial use requires separate licensing.
4. You grant the project owner the right to make commercial use of the code
   in any way, including licensing and distribution.

If you do not agree with the above, please do not contribute.

## Future directions

* Scale the context-time model (parallel scan or chunked integration instead
  of a Python loop over tokens) and test it on real text.
* Symmetry-derived potentials: choose what the energy should be invariant to
  and let conservation laws follow (Noether), rather than adding them by hand.
* Composition/binding experiments: can stable "composite particles" (bound
  states of the learned potential) be found and probed?
* Wire replica exchange (`yaum/core/replica.py`, implemented and tested but
  not yet used by the trainer) into training.
* Subword tokenization (BPE/SentencePiece).
