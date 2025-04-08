import torch
import numpy as np

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available(): # Support for Apple Silicon
     device = torch.device("mps")
     print("Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Batching Function (Adapt from Colab's get_batch) ---
def get_batch(data, context_window, batch_size, device):
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
    return x.to(device), y.to(device)

# --- Add other utility functions if needed ---