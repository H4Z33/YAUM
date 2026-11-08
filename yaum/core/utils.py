"""Device selection and batch sampling."""
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = f"GPU: {torch.cuda.get_device_name(0)}"
elif torch.backends.mps.is_available():  # Apple Silicon
    device = torch.device("mps")
    device_name = "Apple Metal Performance Shaders (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"


def describe_device() -> str:
    """Human-readable description of the selected device (no import-time prints)."""
    if device.type == "cpu":
        return "CPU (CUDA/MPS not available)"
    return f"{device} ({device_name})"


def get_batch(data, context_window, batch_size, target_device):
    """Sample a batch of (context, next-token) windows uniformly at random.

    Vectorised gather: build a (batch_size, context_window) index matrix and
    index ``data`` once, instead of stacking ``batch_size`` Python slices.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)

    max_start = len(data) - context_window - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset too small: need at least context_window + 2 = "
            f"{context_window + 2} tokens, got {len(data)}."
        )

    starts = torch.randint(0, max_start + 1, (batch_size,), device=data.device)
    offsets = torch.arange(context_window, device=data.device)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    x = data[idx]
    y = data[idx + 1]
    return x.to(target_device, non_blocking=True), y.to(target_device, non_blocking=True)