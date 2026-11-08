"""Model registry. Pick an architecture via ``config['model_type']``."""
from __future__ import annotations

from .hamiltonian import HamiltonianCharModel
from .rnn import RNNCharModel
from .transformer import TransformerCharModel

MODELS: dict[str, type] = {
    "rnn": RNNCharModel,
    "transformer": TransformerCharModel,
    "hamiltonian": HamiltonianCharModel,
}


def make_model(vocab_size: int, config: dict):
    """Instantiate a model from the config.

    The model class takes ``vocab_size``, ``embedding_dim``, ``hidden_dim``
    and ``n_layers``. Transformer-specific keys (``n_heads``,
    ``max_context``, ``dropout``) are forwarded when present, as are the
    Hamiltonian model's ``ham_dt``, ``ham_dissipation`` and
    ``ham_n_potential`` (there ``n_layers`` means substeps per token).
    """
    model_type = config.get("model_type", "rnn")
    try:
        cls = MODELS[model_type]
    except KeyError as err:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available: {sorted(MODELS)}"
        ) from err

    kwargs = {
        "vocab_size": vocab_size,
        "embedding_dim": config["embedding_dim"],
        "hidden_dim": config["hidden_dim"],
        "n_layers": config["n_layers"],
    }
    if model_type == "transformer":
        if "n_heads" in config:
            kwargs["n_heads"] = int(config["n_heads"])
        kwargs["max_context"] = int(
            config.get("max_context", config.get("context_window", 1024))
        )
        if "dropout" in config:
            kwargs["dropout"] = float(config["dropout"])
    if model_type == "hamiltonian":
        if "ham_dt" in config:
            kwargs["dt"] = float(config["ham_dt"])
        if "ham_dissipation" in config:
            kwargs["dissipation"] = config["ham_dissipation"]
        if config.get("ham_n_potential"):
            kwargs["n_potential"] = int(config["ham_n_potential"])
    return cls(**kwargs)
