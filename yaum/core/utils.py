# yaum/core/utils.py
import torch
import numpy as np
import os # Import os

print("-" * 50)
print("--- Initializing Device ---")

_device_name = "cpu" # Default

if torch.cuda.is_available():
    _device = torch.device("cuda")
    _device_name = f"GPU: {torch.cuda.get_device_name(0)}"
    print(f"CUDA available. Using device: {_device} ({_device_name})")
    # Optional: Check CUDA version PyTorch was built with
    # print(f"PyTorch CUDA version: {torch.version.cuda}")
    # Optional: Check available CUDA devices
    # print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

elif torch.backends.mps.is_available(): # Support for Apple Silicon
     _device = torch.device("mps")
     _device_name = "Apple Metal Performance Shaders (MPS)"
     print(f"MPS available. Using device: {_device} ({_device_name})")
     # Note: MPS support might still be less complete than CUDA
else:
    _device = torch.device("cpu")
    _device_name = "CPU"
    print("Warning: CUDA/MPS not available. Using CPU.")
    # Optional: Add a stronger warning or exit if GPU was expected
    # print("ERROR: GPU acceleration expected but not found!")
    # sys.exit(1) # Uncomment to force exit if no GPU

device = _device # Assign to the global 'device' variable for import
device_name = _device_name # Assign name for potential UI display

print(f"--- Device set to: {device} ---")
print("-" * 50)


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