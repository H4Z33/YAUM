import torch
import torch.nn.functional as F

# --- Define device within this module or import from utils ---
try:
    from .utils import device
except ImportError:
     if torch.cuda.is_available(): device = torch.device("cuda")
     elif torch.backends.mps.is_available(): device = torch.device("mps")
     else: device = torch.device("cpu")
     print(f"(core/dynamics.py) Using device: {device}")


# --- Loss and Grad Calculation (Adapted from Colab) ---
def calculate_loss_and_grads_rnn(context_indices, target_indices, E_matrix, model, criterion, retain_graph=False):
    """ Calculates loss and gradients w.r.t. input embeddings for RNN. """
    # Assume model is already on the correct device
    batch_size, seq_len = context_indices.shape
    hidden = model.init_hidden(batch_size) # Initializes hidden state on correct device

    # Ensure E_matrix requires grad if needed by caller
    # E_matrix should be on the same device as the model
    batch_embeddings = F.embedding(context_indices, E_matrix)

    logits, _ = model(batch_embeddings, hidden)
    loss = criterion(logits.view(-1, model.vocab_size), target_indices.view(-1))

    # Caller might need to zero E_matrix.grad before calling this if accumulating
    # if E_matrix.grad is not None:
    #     E_matrix.grad.zero_()

    try:
        embedding_grads = torch.autograd.grad(
            outputs=loss,
            inputs=batch_embeddings,
            grad_outputs=torch.ones_like(loss),
            retain_graph=retain_graph
        )[0]
    except RuntimeError as e:
        print(f"RuntimeError during autograd.grad: {e}")
        print("Check if retain_graph=True was needed or if tensors were detached unexpectedly.")
        # Return zero grads or raise? For now, return None to indicate failure.
        return loss, None, batch_embeddings


    return loss, embedding_grads, batch_embeddings


# --- Hamiltonian Step (Adapted from Colab) ---
def hamiltonian_step_rnn(context_indices, target_indices, E, P, model, criterion, mass_vector, dt):
    """ Performs one step of Leapfrog integration for embeddings E and momenta P (RNN version). """
    # Assume E, P, model, mass_vector are on the correct device

    # Ensure E requires grad for the step
    E_requires_grad_orig = E.requires_grad
    E.requires_grad_(True)

    active_indices_flat = context_indices.flatten().unique() # Find unique tokens in batch

    # Use E directly if requires_grad=True, otherwise clone
    # We need grad enabled for loss calculation, but updates happen out-of-place
    E_current_for_grad = E # if E.requires_grad else E.clone().requires_grad_(True)

    # --- Step 1: Calculate Force at time t ---
    # Ensure E_current_for_grad has its grad zeroed if necessary (safer: do it here)
    if E_current_for_grad.grad is not None:
        E_current_for_grad.grad.zero_()

    loss1, grad_V1, _ = calculate_loss_and_grads_rnn(
        context_indices, target_indices, E_current_for_grad, model, criterion, retain_graph=False
    )
    if grad_V1 is None: # Handle potential autograd failure
        print("Warning: grad_V1 calculation failed in Hamiltonian step. Skipping update.")
        E.requires_grad_(E_requires_grad_orig) # Restore original grad setting
        # Return state unchanged, perhaps with NaN loss to signal error?
        return E, P, torch.tensor(float('nan')), torch.tensor(float('nan'))

    # Use P.clone() to avoid modifying P in-place during calculation
    P_current = P # Use reference, update will create P_new

    # Accumulate force using index_add_
    force_t_sparse = torch.zeros_like(P_current)
    # Use reshape for safety with non-contiguous tensors
    force_t_sparse.index_add_(0, context_indices.view(-1), -grad_V1.reshape(-1, E.shape[1]))

    # --- Step 2: Update momenta half step ---
    P_half = P_current + force_t_sparse * (dt / 2.0)

    # --- Step 3: Update positions full step ---
    # Calculate velocity only for active embeddings to potentially save computation
    # However, indexing mass_vector might be slower than broadcast division if vocab is huge? Test needed.
    # Simpler: Use broadcast division (assuming mass_vector is (vocab_size, 1))
    # velocity_half = P_half / mass_vector # Requires mass_vector on correct device & shape

    # Update using active indices for potentially better performance if batch << vocab_size
    p_half_active = P_half[active_indices_flat]
    m_active = mass_vector[active_indices_flat] # mass_vector should be (vocab_size, 1)
    velocity_active = p_half_active / m_active

    # Create E_new out-of-place
    E_new_temp = E.clone().detach() # Start with current E values, detached
    E_new_temp.index_add_(0, active_indices_flat, velocity_active * dt)
    E_new_temp.requires_grad_(True) # Enable grad for the next loss calculation

    # --- Step 4: Calculate Force at time t+dt ---
    # Ensure grad is zeroed before calculation
    if E_new_temp.grad is not None: E_new_temp.grad.zero_()

    # *** CRITICAL: retain_graph=True for loss2 used in W update ***
    loss2, grad_V2, _ = calculate_loss_and_grads_rnn(
        context_indices, target_indices, E_new_temp, model, criterion, retain_graph=True
    )

    if grad_V2 is None:
        print("Warning: grad_V2 calculation failed in Hamiltonian step. Using previous P.")
        E.requires_grad_(E_requires_grad_orig)
        # Return E_new but the old P, signal error in loss2?
        return E_new_temp.detach().requires_grad_(E_requires_grad_orig), P, loss1.detach(), torch.tensor(float('nan'))


    force_t_plus_dt_sparse = torch.zeros_like(P_current)
    force_t_plus_dt_sparse.index_add_(0, context_indices.view(-1), -grad_V2.reshape(-1, E.shape[1]))

    # --- Step 5: Update momenta full step ---
    P_new = P_half + force_t_plus_dt_sparse * (dt / 2.0)

    # Restore original requires_grad state for E before returning E_new
    E_new_final = E_new_temp.detach().requires_grad_(E_requires_grad_orig)

    return E_new_final, P_new.detach(), loss1.detach(), loss2 # loss2 still has graph attached