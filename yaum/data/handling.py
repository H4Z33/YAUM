import torch
import pickle
import os
from collections import Counter

# --- Tokenization ---
def tokenize_text(text):
    """Performs character-level tokenization."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"Character vocabulary size: {vocab_size}")
    return char_to_idx, idx_to_char, vocab_size

def text_to_indices(text, char_to_idx):
    """Converts text string to list of indices."""
    return [char_to_idx.get(ch, 0) for ch in text] # Use index 0 for unknown? Or handle differently?

# --- Data Loading and Splitting ---
def load_and_split_data(filepath, test_split_ratio=0.1):
    """Loads text, tokenizes, and splits into train/test tensors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Read {len(text)} characters from {filepath}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None, None, None, None

    char_to_idx, idx_to_char, vocab_size = tokenize_text(text)
    data_indices = text_to_indices(text, char_to_idx)
    data_tensor = torch.tensor(data_indices, dtype=torch.long)

    split_idx = int(len(data_tensor) * (1 - test_split_ratio))
    train_data = data_tensor[:split_idx]
    test_data = data_tensor[split_idx:]

    print(f"Training data size: {len(train_data)} tokens")
    print(f"Test data size: {len(test_data)} tokens")

    return train_data, test_data, char_to_idx, idx_to_char, vocab_size

# --- Mass Vector Calculation ---
def calculate_mass_vector(train_data, vocab_size, mass_type="frequency", device='cpu'):
    """Calculates the mass vector based on training data frequency."""
    print(f"Calculating mass vector (type: {mass_type})...")
    if mass_type == "uniform":
        mass_vector = torch.ones(vocab_size, device=device)
    else:
        # Ensure train_data is a tensor for bincount
        if not isinstance(train_data, torch.Tensor):
             train_data = torch.tensor(train_data, dtype=torch.long)

        freqs = torch.bincount(train_data, minlength=vocab_size).float().to(device)
        freqs += 1e-7 # Epsilon for stability, increased slightly

        if mass_type == "frequency":
            # Normalize mass (e.g., by mean frequency)
            mean_freq = freqs.mean()
            if mean_freq == 0: mean_freq = 1.0 # Avoid division by zero if vocab > train data chars
            mass_vector = freqs / mean_freq
            mass_vector[mass_vector == 0] = 1e-7 # Ensure no zero mass

        elif mass_type == "inverse_frequency":
            # Using inverse frequency: rarer tokens are heavier
            mass_vector = 1.0 / freqs
            # Normalize mass (e.g., by mean inverse frequency)
            mean_inv_freq = mass_vector.mean()
            if mean_inv_freq == 0: mean_inv_freq = 1.0
            mass_vector = mass_vector / mean_inv_freq
            mass_vector[torch.isinf(mass_vector)] = 1.0 # Handle potential inf if freq was 0
            mass_vector[mass_vector == 0] = 1e-7 # Ensure no zero mass

        else: # Fallback to uniform
             print(f"Warning: Unknown mass_type '{mass_type}'. Falling back to uniform.")
             mass_vector = torch.ones(vocab_size, device=device)

    print(f"Mass vector: Min={mass_vector.min():.4f}, Max={mass_vector.max():.4f}, Mean={mass_vector.mean():.4f}")
    # Add a minimum clamp to avoid extremely small masses if needed
    mass_vector = torch.clamp(mass_vector, min=1e-6)
    return mass_vector.unsqueeze(1) # Return as (vocab_size, 1) for broadcasting

# --- Vocabulary Saving/Loading ---
def save_vocab(char_to_idx, idx_to_char, vocab_size, filepath):
    vocab_data = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    except Exception as e:
        print(f"Error saving vocabulary: {e}")

def load_vocab(filepath):
    try:
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        print(f"Vocabulary loaded from {filepath}")
        return vocab_data['char_to_idx'], vocab_data['idx_to_char'], vocab_data['vocab_size']
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {filepath}")
        return None, None, None
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return None, None, None