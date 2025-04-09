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


# --- Batching Function (Adapt from Colab's get_batch) ---
def get_batch(data, context_window, batch_size, target_device):
    """Gets a random batch of context (x) and targets (y)."""
    # Ensure data is a torch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)

    # Generate random starting indices for sequences in the batch
    ix = torch.randint(len(data) - context_window, (batch_size,))
    # Stack the sequences starting from these indices
    x = torch.stack([data[i:i+context_window] for i in ix])
    # The target for each position in the context is the next character
    y = torch.stack([data[i+1:i+context_window+1] for i in ix])
    # Use the passed target_device explicitly
    return x.to(target_device), y.to(target_device)

# --- Add other utility functions if needed ---