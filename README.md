# YAUM
Yet Another Universal Model

# Hamiltonian LLM Trainer

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-lightgrey.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

**Explore Language Model Training Through the Lens of Classical Mechanics**

This project provides a framework and application for training character-level language models using an approach inspired by Hamiltonian mechanics and the principle of least action. Instead of standard gradient descent on embeddings, we treat token embeddings as particles in a "Semantic Phase Space" evolving over training time according to Hamiltonian dynamics.

**Key Features:**

*   **Hamiltonian Dynamics:** Implements Leapfrog integration for updating token embeddings (`E`) and their corresponding momenta (`P`).
*   **Configurable Physics:** Easily adjust timestep (`dt`), adaptive `dt` bounds, and token mass. New runs default to Fisher mass with frequency/uniform scalar mass still available.
*   **Standard + Hamiltonian:** Combines Hamiltonian updates for embeddings with standard optimizers (Adam) for the main model weights.
*   **Model-Agnostic Force Field:** Runs the same Hamiltonian substrate under RNN or Transformer sequence models.
*   **Interactive UI:** A Gradio-based interface for easy configuration, data loading, training monitoring (progress, logs, plots), and text generation (inference).
*   **Modular Structure:** Code organized into data handling, models, core dynamics, trainer logic, and UI for extensibility.
*   **Checkpointing:** Save and load training state (model weights, embeddings, momenta, optimizer state), including periodic snapshots and phase-shift snapshots.
*   **Dual Licensed:** Offered under both Apache 2.0 and PolyForm Noncommercial 1.0.0 licenses (see below).

**(Placeholder for a Screenshot/GIF of the Gradio UI)**
![UI Screenshot Placeholder](placeholder.png)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/hamiltonian-llm-trainer.git
    cd hamiltonian-llm-trainer
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You might need to install PyTorch separately first, following instructions on the official PyTorch website: https://pytorch.org/)*

## How to Run

### One-liner via uv (recommended)

Both scripts auto-install [`uv`](https://github.com/astral-sh/uv) on first run,
resolve Python 3.10+, install the dependencies in an isolated environment,
and launch the Gradio UI.

```bash
# Linux / macOS
./run.sh

# Windows PowerShell
.\run.ps1
```

Any extra arguments are forwarded to `python -m yaum.ui.app`.

### Manual

1.  **Ensure dependencies are installed.**
2.  **Run the Gradio application:**
    ```bash
    python -m yaum.ui.app
    ```
3.  **Open your web browser** to the local URL provided (usually `http://127.0.0.1:7860`).

## Basic Usage

1.  **Setup & Config Tab:**
    *   Enter the path to your training text file (`.txt`).
    *   Adjust model architecture (model family, embedding dim, hidden dim, layers), Hamiltonian parameters (`dt`, mass mode/type), snapshot cadence, and training settings (steps, batch size, learning rate).
    *   Optionally, provide a path to a checkpoint file to load.
    *   Click "Prepare Data & Model / Apply Config" to load data and initialize the trainer based on the current settings OR click "Load Checkpoint" to load a previous run. Check the status bar for confirmation or errors.
2.  **Training Tab:**
    *   Click "Start Training". The training will begin in the background.
    *   Monitor progress via the progress bar, log output, and dynamically updating plots.
    *   Click "Stop Training" to gracefully halt the process (saves a final checkpoint).
3.  **Inference Tab:**
    *   Ensure a model is ready (either trained or loaded from a checkpoint).
    *   Enter a starting prompt.
    *   Adjust generation length and temperature.
    *   Click "Generate Text".

## License

This project is dual-licensed:

*   **PolyForm Noncommercial License 1.0.0:** You may use this software for non-commercial purposes. See [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
*   **Commercial use requires separate licensing** 

You may choose which license applies to your specific use case. Contributions are accepted under the Apache License 2.0.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and the [CALL_FOR_COLLABORATORS.md](CALL_FOR_COLLABORATORS.md) if you're interested in joining the research effort.

## Future Directions

*   Implement subword tokenization (BPE/SentencePiece).
*   Integrate richer model architectures and substrate couplings.
*   Explore different potential energy (`V`) formulations incorporating grammatical rules.
*   Add more sophisticated UI features and visualizations.
*   Rigorous comparison against standard training baselines.
*   Theoretical analysis of the dynamics in semantic space.

## Contributor License Agreement (CLA)

By contributing to this project, you agree to the following:

1. All contributions you make to this repository are licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
2. You acknowledge that **you will not retain rights** to the code after contributing, and the rights to the code will remain with the project owner.
3. You agree that **commercial use requires separate licensing**.
4. You grant the project owner the ability to make commercial use of the code in any way, including licensing and distributing.

If you do not agree with the above, please do not contribute.

For commercial use inquiries, please contact me on GitHub
