"""Model registry. Pick an architecture via ``config['model_type']``."""
from __future__ import annotations

from .rnn import RNNCharModel
from .transformer import TransformerCharModel

MODELS: dict[str, type] = {
    "rnn": RNNCharModel,
    "transformer": TransformerCharModel,
}


def make_model(vocab_size: int, config: dict):
    """Instantiate a model from the config.

    The model class takes ``vocab_size``, ``embedding_dim``, ``hidden_dim``
    and ``n_layers``. Transformer-specific keys (``n_heads``,
    ``max_context``, ``dropout``) are forwarded when present.
    """
    model_type = config.get("model_type", "rnn")
    try:
        cls = MODELS[model_type]
    except KeyError as err:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available: {sorted(MODELS)}"
        ) from err

    kwargs = dict(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    )
    if model_type == "transformer":
        if "n_heads" in config:
            kwargs["n_heads"] = int(config["n_heads"])
        kwargs["max_context"] = int(
            config.get("max_context", config.get("context_window", 1024))
        )
        if "dropout" in config:
            kwargs["dropout"] = float(config["dropout"])
    return cls(**kwargs)
