# yaum/core/trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F # <--- ADD THIS LINE
import os
import time
import numpy as np
import pickle
from .utils import device, get_batch
from .dynamics import hamiltonian_step_rnn
from ..models.rnn import RNNCharModel # Adjust import path if needed
from ..data.handling import save_vocab # Adjust import path

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.E = None # Embedding matrix
        self.P = None # Momentum matrix
        self.optimizer_W = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mass_vector = None

        self.train_data = None
        self.test_data = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None

        self.current_step = 0
        self.train_losses_l1 = []
        self.train_losses_l2 = []
        self.test_losses = []
        self.debug_stats = {'grad_W': [], 'force_E': [], 'P_norm': []}
        self.run_id = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        self.save_dir = os.path.join(config.get('results_dir', 'results'), self.run_id)
        os.makedirs(self.save_dir, exist_ok=True)

        self._stop_training_flag = False

    def setup(self, train_data, test_data, char_to_idx, idx_to_char, vocab_size, mass_vector):
        """Initialize model, optimizers, E, P based on loaded data and config."""
        print("Setting up trainer...")
        self.train_data = train_data
        self.test_data = test_data
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        self.mass_vector = mass_vector.to(device) # Ensure mass vector is on correct device

        # Instantiate model
        self.model = RNNCharModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers']
        ).to(device)
        print(f"Model instantiated with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} params.")

        # Initialize E and P
        self.E = torch.randn(self.vocab_size, self.config['embedding_dim'], device=device, requires_grad=True)
        self.P = torch.zeros(self.vocab_size, self.config['embedding_dim'], device=device)
        print(f"E: {self.E.shape}, P: {self.P.shape}")

        # Initialize optimizer for model weights W
        self.optimizer_W = optim.Adam(self.model.parameters(), lr=self.config['learning_rate_W'])
        print(f"Optimizer W: Adam lr={self.config['learning_rate_W']}")

        self.current_step = 0
        self._stop_training_flag = False
        print("Trainer setup complete.")


    def train(self):
        """Runs the main training loop."""
        if not self.model or self.E is None:
            print("Error: Trainer not set up. Call setup() first.")
            # Consider yielding an error message back to UI
            yield {"status": "error", "message": "Trainer not set up"}
            return

        self._stop_training_flag = False
        start_time = time.time()
        self.model.train()
        self.E.requires_grad_(True) # Ensure E requires grad for training

        print(f"Starting training from step {self.current_step}...")

        for step in range(self.current_step, self.config['num_steps']):
            if self._stop_training_flag:
                print(f"Training stopped externally at step {step}.")
                yield {"status": "stopped", "step": step}
                break

            # --- Get Batch ---
            x_batch, y_batch = get_batch(
                self.train_data,
                self.config['context_window'],
                self.config['batch_size'],
                device # Pass the imported device variable
            )

            # --- Hamiltonian Step for E, P ---
            try:
                 E_new, P_new, loss_l1, loss_l2 = hamiltonian_step_rnn(
                      x_batch, y_batch, self.E, self.P, self.model, self.criterion,
                      self.mass_vector, self.config['dt']
                 )
                 # Check for NaN losses which might indicate instability
                 if torch.isnan(loss_l1) or torch.isnan(loss_l2):
                      print(f"ERROR: NaN loss detected at step {step}. Stopping training.")
                      yield {"status": "error", "message": f"NaN loss at step {step}"}
                      break

            except Exception as e:
                 print(f"ERROR during Hamiltonian step {step}: {e}")
                 # Potentially log traceback
                 yield {"status": "error", "message": f"Error in Hamiltonian step: {e}"}
                 break # Stop training on error


            # --- Store debug info (approximate force and P norm) ---
            with torch.no_grad():
                 delta_P = P_new - self.P
                 force_eff_norm = torch.linalg.norm(delta_P / self.config['dt']).item() if self.config['dt'] != 0 else 0
                 p_norm = torch.linalg.norm(P_new).item()

            # Update E and P
            self.E = E_new
            self.P = P_new

            # --- Standard Update for W (using loss_l2) ---
            self.optimizer_W.zero_grad()
            # loss_l2 has graph attached from hamiltonian_step
            try:
                loss_l2.backward() # Backpropagate through loss_l2 for W gradients
            except RuntimeError as e:
                print(f"ERROR during W backward pass at step {step}: {e}")
                yield {"status": "error", "message": f"Error in W backward pass: {e}"}
                break


            # Calculate W grad norm pre-clip for debugging
            with torch.no_grad():
                 total_norm_W_before = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5

            # Clip W gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clip_norm_W'])

            # Update W
            self.optimizer_W.step()

            self.current_step = step + 1 # Increment step count

            # --- Logging and Evaluation ---
            if self.current_step % self.config['eval_interval'] == 0 or self.current_step == self.config['num_steps']:
                eval_results = self.evaluate() # Get test loss
                test_loss = eval_results['test_loss']

                # Store history
                self.train_losses_l1.append(loss_l1.item())
                self.train_losses_l2.append(loss_l2.item())
                self.test_losses.append(test_loss)
                self.debug_stats['grad_W'].append(total_norm_W_before)
                self.debug_stats['force_E'].append(force_eff_norm)
                self.debug_stats['P_norm'].append(p_norm)

                time_elapsed = time.time() - start_time
                log_message = (
                    f"Step {self.current_step}/{self.config['num_steps']} | T: {time_elapsed:.1f}s "
                    f"| L1: {loss_l1.item():.4f} | L2->W: {loss_l2.item():.4f} | Test L: {test_loss:.4f} "
                    f"| ||∇W||: {total_norm_W_before:.3f} | ||F_E||: {force_eff_norm:.3f} | ||P||: {p_norm:.3f}"
                )
                print(log_message)

                # Yield progress update for UI
                yield {
                    "status": "running",
                    "step": self.current_step,
                    "max_steps": self.config['num_steps'],
                    "l1": loss_l1.item(),
                    "l2": loss_l2.item(),
                    "test_loss": test_loss,
                    "grad_W_norm": total_norm_W_before,
                    "force_E_norm": force_eff_norm,
                    "P_norm": p_norm,
                    "log_message": log_message,
                    # Include loss history for plotting
                    "history": self.get_history()
                }

                # --- Save Checkpoint ---
                self.save_checkpoint()


        if not self._stop_training_flag:
            print("\nTraining finished.")
            yield {"status": "finished", "step": self.current_step, "history": self.get_history()}


    def evaluate(self):
        """Calculates loss on the test set."""
        # Explicitly check if model and test_data are initialized and test_data is not empty
        if self.model is None or self.test_data is None or len(self.test_data) == 0:
            print("Warning: Skipping evaluation. Model or test data not ready or test data empty.")
            return {"test_loss": float('nan')}

        self.model.eval() # Set model to evaluation mode
        total_loss = 0
        eval_iters = 100 # Number of batches to average test loss over
        actual_eval_iters = 0
        nan_detected = False # Flag to track if NaN occurred

        print(f"Starting evaluation for {eval_iters} batches...") # Add start message

        with torch.no_grad():
            eval_batch_size = self.config.get('eval_batch_size', self.config['batch_size'])
            # Initialize hidden state ONCE before the loop if RNN structure allows
            try:
                hidden_test = self.model.init_hidden(eval_batch_size)
            except Exception as e:
                print(f"Error initializing hidden state for eval: {e}")
                self.model.train()
                return {"test_loss": float('nan')}


            for k in range(eval_iters):
                try:
                    x_test, y_test = get_batch(
                        self.test_data,
                        self.config['context_window'],
                        eval_batch_size,
                        device
                    )

                    # Check batch validity
                    if x_test is None or y_test is None or x_test.numel() == 0 or y_test.numel() == 0:
                         print(f"Warning: Evaluation batch {k} is empty or invalid. Skipping.")
                         continue

                    test_embeddings = F.embedding(x_test, self.E.detach())

                    # Check for NaNs in embeddings (unlikely but possible)
                    if torch.isnan(test_embeddings).any():
                         print(f"WARNING: NaN detected in test_embeddings at eval batch {k}. Skipping.")
                         nan_detected = True
                         continue

                    test_logits, hidden_test = self.model(test_embeddings, hidden_test)
                    hidden_test = tuple([h.detach() for h in hidden_test])

                    # Check for NaNs in logits
                    if torch.isnan(test_logits).any():
                         print(f"WARNING: NaN detected in model logits at eval batch {k}. Skipping.")
                         nan_detected = True
                         continue

                    loss = self.criterion(test_logits.view(-1, self.vocab_size), y_test.view(-1))

                    # Check for NaN in loss
                    if torch.isnan(loss):
                        print(f"WARNING: NaN calculated for loss at eval batch {k}. Logits min/max: {test_logits.min():.2f}/{test_logits.max():.2f}. Skipping batch.")
                        nan_detected = True
                        continue # Skip adding NaN to total_loss

                    total_loss += loss.item()
                    actual_eval_iters += 1

                except Exception as e:
                    print(f"Warning: Error during evaluation batch {k}: {e}")
                    # Optionally add more detailed error logging/traceback here
                    continue # Try next batch

        self.model.train() # Set model back to training mode

        if actual_eval_iters == 0:
             print("Error: No evaluation batches were successfully processed.")
             avg_loss = float('nan')
        else:
             avg_loss = total_loss / actual_eval_iters

        # Add final printout
        print(f"Evaluation finished. Avg Loss: {avg_loss:.4f} ({actual_eval_iters}/{eval_iters} batches successful). NaN detected: {nan_detected}")

        return {"test_loss": avg_loss}


    def save_checkpoint(self, filename="checkpoint_latest.pt"):
        """Saves the current state (model, E, P, optimizer, step)."""
        if not self.model or self.E is None or self.optimizer_W is None:
            print("Warning: Cannot save checkpoint, trainer not fully initialized.")
            return

        state = {
            'step': self.current_step,
            'config': self.config,
            'vocab_size': self.vocab_size,
            'model_state_dict': self.model.state_dict(),
            'E': self.E.detach().cpu(), # Save on CPU
            'P': self.P.detach().cpu(), # Save on CPU
            'optimizer_W_state_dict': self.optimizer_W.state_dict(),
            'train_losses_l1': self.train_losses_l1,
            'train_losses_l2': self.train_losses_l2,
            'test_losses': self.test_losses,
            'debug_stats': self.debug_stats,
            # Save vocab mappings too for self-contained checkpoint
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }
        filepath = os.path.join(self.save_dir, filename)
        try:
            torch.save(state, filepath)
            print(f"Checkpoint saved to {filepath} at step {self.current_step}")
            # Save vocab separately as well? Maybe redundant if in checkpoint
            # vocab_path = os.path.join(self.save_dir, "vocab.pkl")
            # save_vocab(self.char_to_idx, self.idx_to_char, self.vocab_size, vocab_path)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, filepath):
        """Loads state from a checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=device) # Load directly to target device
            print(f"Loading checkpoint from {filepath}...")

            # Load config first to recreate model structure
            self.config = checkpoint['config']
            self.vocab_size = checkpoint['vocab_size']
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']

            # Re-create model and optimizer with loaded config
            self.model = RNNCharModel(
                vocab_size=self.vocab_size,
                embedding_dim=self.config['embedding_dim'],
                hidden_dim=self.config['hidden_dim'],
                n_layers=self.config['n_layers']
            ).to(device)
            self.optimizer_W = optim.Adam(self.model.parameters(), lr=self.config['learning_rate_W']) # Recreate Adam

            # Load states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.E = checkpoint['E'].to(device).requires_grad_(True) # Ensure grad=True after loading
            self.P = checkpoint['P'].to(device)
            self.optimizer_W.load_state_dict(checkpoint['optimizer_W_state_dict'])
            self.current_step = checkpoint['step']
            self.train_losses_l1 = checkpoint['train_losses_l1']
            self.train_losses_l2 = checkpoint['train_losses_l2']
            self.test_losses = checkpoint['test_losses']
            self.debug_stats = checkpoint['debug_stats']
            self.save_dir = os.path.dirname(filepath) # Update save dir based on loaded checkpoint

            print(f"Checkpoint loaded successfully. Resuming from step {self.current_step}")
            # Need to ensure data and mass vector are also loaded/recalculated consistent with checkpoint
            return True

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Log traceback here potentially
            return False


    def stop_training(self):
        """Sets the flag to stop the training loop gracefully."""
        print("Stop training requested.")
        self._stop_training_flag = True

    def get_history(self):
        """Returns the training history for plotting."""
        eval_interval = self.config.get('eval_interval', 1) # Handle case where interval might be missing
        steps = list(np.arange(len(self.test_losses)) * eval_interval)
        return {
            'steps': steps,
            'train_l1': self.train_losses_l1,
            'train_l2': self.train_losses_l2,
            'test_l': self.test_losses,
            'grad_W': self.debug_stats['grad_W'],
            'force_E': self.debug_stats['force_E'],
            'P_norm': self.debug_stats['P_norm']
        }

    # --- Inference Method (Conceptual) ---
    def generate(self, start_prompt, length, temperature):
        """Generates text using the trained model."""
        if not self.model or self.E is None or self.idx_to_char is None:
            return "Error: Model not loaded or trainer not set up."

        print(f"Generating text (temp={temperature}): '{start_prompt}'")
        self.model.eval()
        generated_indices = [self.char_to_idx.get(char, 0) for char in start_prompt]
        context_indices = torch.tensor([generated_indices], dtype=torch.long, device=device)
        hidden = self.model.init_hidden(batch_size=1)

        # Prime hidden state
        if len(generated_indices) > 0:
            with torch.no_grad():
                prompt_embeddings = F.embedding(context_indices, self.E.detach())
                _, hidden = self.model(prompt_embeddings, hidden)

        current_input_idx = torch.tensor([[generated_indices[-1]]], dtype=torch.long, device=device) if generated_indices else \
                          torch.randint(0, self.vocab_size, (1,1), device=device) # Start random if no prompt


        with torch.no_grad():
            for _ in range(length):
                current_input_embedding = F.embedding(current_input_idx, self.E.detach())
                logits, hidden = self.model(current_input_embedding, hidden)
                last_logits = logits.squeeze(1)
                scaled_logits = last_logits / temperature
                probabilities = F.softmax(scaled_logits, dim=-1)
                next_token_index = torch.multinomial(probabilities, num_samples=1)
                generated_indices.append(next_token_index.item())
                current_input_idx = next_token_index

        self.model.train() # Set back to train mode? Or keep eval? Depends.
        generated_text = "".join([self.idx_to_char.get(idx, '?') for idx in generated_indices])
        return generated_text