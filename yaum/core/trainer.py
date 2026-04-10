"""Trainer that evolves embeddings via symplectic dynamics.

The trainer owns three state objects:

* a neural network whose weights ``W`` are updated by a standard optimiser;
* an embedding matrix ``E`` that moves through semantic phase space;
* a momentum matrix ``P`` conjugate to ``E``.

Each training step asks a :class:`SymplecticIntegrator` to advance ``(E, P)``
by ``dt``. The potential energy is the loss itself, so the model's error
landscape literally shapes the trajectory of its own embeddings.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..data.handling import save_vocab  # noqa: F401 - re-exported for callers
from ..models.rnn import RNNCharModel
from .adaptive import AdaptiveStepController
from .dynamics import make_rnn_force_fn, total_hamiltonian
from .integrators import make_integrator
from .utils import device, get_batch


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.E = None
        self.P = None
        self.optimizer_W = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mass_vector = None

        self.train_data = None
        self.test_data = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None

        self.integrator = make_integrator(config.get("integrator", "leapfrog"))
        self.adaptive = self._build_adaptive_controller(config)

        self.current_step = 0
        self.train_losses_l1: list[float] = []
        self.train_losses_l2: list[float] = []
        self.test_losses: list[float] = []
        self.debug_stats: dict[str, list[float]] = {
            "grad_W": [],
            "force_E": [],
            "P_norm": [],
            "H_drift": [],
            "dt": [],
        }
        self._energy_reference = None
        self._current_dt: float = float(config["dt"]) if "dt" in config else 0.01

        self.run_id = f"run_{time.strftime('%Y%m%d-%H%M%S')}"

        self.save_dir = os.path.join(config.get("results_dir", "results"), self.run_id)
        os.makedirs(self.save_dir, exist_ok=True)

        self._stop_training_flag = False

    @staticmethod
    def _build_adaptive_controller(config) -> AdaptiveStepController | None:
        if not config.get("adaptive_dt"):
            return None
        dt = float(config.get("dt", 0.01))
        return AdaptiveStepController(
            dt_init=dt,
            dt_min=float(config.get("dt_min", dt * 1e-2)),
            dt_max=float(config.get("dt_max", dt * 1e2)),
            drift_high=float(config.get("drift_high", 1e-2)),
            drift_low=float(config.get("drift_low", 1e-5)),
            shrink=float(config.get("dt_shrink", 0.5)),
            grow=float(config.get("dt_grow", 1.2)),
            grow_after=int(config.get("dt_grow_after", 10)),
        )

    def setup(
        self,
        train_data,
        test_data,
        char_to_idx,
        idx_to_char,
        vocab_size,
        mass_vector,
    ):
        print("Setting up trainer...")
        self.train_data = train_data
        self.test_data = test_data
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        self.mass_vector = mass_vector.to(device)

        self.model = RNNCharModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            n_layers=self.config["n_layers"],
        ).to(device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model instantiated with {n_params:,} params.")

        self.E = torch.randn(
            self.vocab_size,
            self.config["embedding_dim"],
            device=device,
            requires_grad=True,
        )
        self.P = torch.zeros(
            self.vocab_size, self.config["embedding_dim"], device=device
        )
        print(f"E: {self.E.shape}, P: {self.P.shape}")

        self.optimizer_W = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate_W"]
        )
        print(
            f"Optimizer W: Adam lr={self.config['learning_rate_W']} "
            f"| integrator: {self.integrator.name} (order {self.integrator.order})"
        )

        self.current_step = 0
        self._stop_training_flag = False
        self._energy_reference = None
        self._current_dt = float(self.config["dt"])
        if self.adaptive is not None:
            self.adaptive.dt = self._current_dt
        print("Trainer setup complete.")

    def _step_embeddings(self, x_batch, y_batch):
        """Run one integrator step and return (new_E, new_P, loss1, loss2)."""
        force_fn = make_rnn_force_fn(x_batch, y_batch, self.model, self.criterion)
        step = self.integrator.step(
            self.E,
            self.P,
            self._current_dt,
            force_fn,
            self.mass_vector,
            retain_final=True,
        )
        return step.E, step.P, step.loss_initial, step.loss_final

    def train(self):
        if not self.model or self.E is None:
            yield {"status": "error", "message": "Trainer not set up"}
            return

        self._stop_training_flag = False
        start_time = time.time()
        self.model.train()
        self.E.requires_grad_(True)

        print(f"Starting training from step {self.current_step}...")

        for step in range(self.current_step, self.config["num_steps"]):
            if self._stop_training_flag:
                print(f"Training stopped externally at step {step}.")
                yield {"status": "stopped", "step": step}
                break

            x_batch, y_batch = get_batch(
                self.train_data,
                self.config["context_window"],
                self.config["batch_size"],
                device,
            )

            try:
                E_new, P_new, loss_l1, loss_l2 = self._step_embeddings(x_batch, y_batch)
            except Exception as e:
                print(f"ERROR during integrator step {step}: {e}")
                yield {"status": "error", "message": f"Integrator step failed: {e}"}
                break

            if torch.isnan(loss_l1) or torch.isnan(loss_l2):
                print(f"ERROR: NaN loss detected at step {step}. Stopping training.")
                yield {"status": "error", "message": f"NaN loss at step {step}"}
                break

            with torch.no_grad():
                dt = self._current_dt
                force_eff_norm = (
                    torch.linalg.norm((P_new - self.P) / dt).item() if dt else 0.0
                )
                p_norm = torch.linalg.norm(P_new).item()

            self.E = E_new.requires_grad_(True)
            self.P = P_new

            self.optimizer_W.zero_grad()
            try:
                loss_l2.backward()
            except RuntimeError as e:
                print(f"ERROR during W backward pass at step {step}: {e}")
                yield {"status": "error", "message": f"W backward failed: {e}"}
                break

            with torch.no_grad():
                total_norm_W_before = (
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config["gradient_clip_norm_W"],
            )
            self.optimizer_W.step()
            self.current_step = step + 1

            energy = total_hamiltonian(self.P, self.mass_vector, loss_l2)
            if self._energy_reference is None:
                self._energy_reference = energy
            h_drift = energy.drift(self._energy_reference)

            if self.adaptive is not None:
                self._current_dt = self.adaptive.observe(h_drift)

            eval_every = self.config["eval_interval"]
            is_eval_step = (
                self.current_step % eval_every == 0
                or self.current_step == self.config["num_steps"]
            )
            if not is_eval_step:
                continue

            eval_results = self.evaluate()
            test_loss = eval_results["test_loss"]

            self.train_losses_l1.append(loss_l1.item())
            self.train_losses_l2.append(loss_l2.item())
            self.test_losses.append(test_loss)
            self.debug_stats["grad_W"].append(total_norm_W_before)
            self.debug_stats["force_E"].append(force_eff_norm)
            self.debug_stats["P_norm"].append(p_norm)
            self.debug_stats["H_drift"].append(h_drift)
            self.debug_stats["dt"].append(self._current_dt)

            elapsed = time.time() - start_time
            log_message = (
                f"Step {self.current_step}/{self.config['num_steps']} | T: {elapsed:.1f}s "
                f"| L1: {loss_l1.item():.4f} | L2->W: {loss_l2.item():.4f} | Test L: {test_loss:.4f} "
                f"| ||∇W||: {total_norm_W_before:.3f} | ||F_E||: {force_eff_norm:.3f} "
                f"| ||P||: {p_norm:.3f} | ΔH/H: {h_drift:.3e} | dt: {self._current_dt:.3e}"
            )
            print(log_message)

            yield {
                "status": "running",
                "step": self.current_step,
                "max_steps": self.config["num_steps"],
                "l1": loss_l1.item(),
                "l2": loss_l2.item(),
                "test_loss": test_loss,
                "grad_W_norm": total_norm_W_before,
                "force_E_norm": force_eff_norm,
                "P_norm": p_norm,
                "H_drift": h_drift,
                "dt": self._current_dt,
                "log_message": log_message,
                "history": self.get_history(),
            }
            self.save_checkpoint()

        if not self._stop_training_flag:
            print("\nTraining finished.")
            yield {
                "status": "finished",
                "step": self.current_step,
                "history": self.get_history(),
            }

    def evaluate(self):
        if self.model is None or self.test_data is None or len(self.test_data) == 0:
            print("Warning: Skipping evaluation. Model or test data not ready.")
            return {"test_loss": float("nan")}

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        eval_iters = self.config.get("eval_iters", 100)
        actual_iters = 0
        nan_detected = False

        with torch.no_grad():
            eval_batch_size = self.config.get("eval_batch_size", self.config["batch_size"])
            for k in range(eval_iters):
                try:
                    x_test, y_test = get_batch(
                        self.test_data,
                        self.config["context_window"],
                        eval_batch_size,
                        device,
                    )
                except ValueError as e:
                    print(f"Evaluation cannot sample a batch: {e}")
                    break

                # Each batch is sampled from random positions, so the hidden
                # state must be reset per batch — persisting it across
                # unrelated windows was silently corrupting the eval signal.
                hidden = self.model.init_hidden(eval_batch_size)

                test_embeddings = F.embedding(x_test, self.E.detach())
                if torch.isnan(test_embeddings).any():
                    nan_detected = True
                    continue

                test_logits, _ = self.model(test_embeddings, hidden)
                if torch.isnan(test_logits).any():
                    nan_detected = True
                    continue

                loss = self.criterion(
                    test_logits.view(-1, self.vocab_size), y_test.view(-1)
                )
                if torch.isnan(loss):
                    nan_detected = True
                    continue

                total_loss += loss.item()
                actual_iters += 1

        if was_training:
            self.model.train()

        avg_loss = total_loss / actual_iters if actual_iters > 0 else float("nan")
        print(
            f"Evaluation finished. Avg Loss: {avg_loss:.4f} "
            f"({actual_iters}/{eval_iters} batches). NaN: {nan_detected}"
        )
        return {"test_loss": avg_loss}

    def save_checkpoint(self, filename="checkpoint_latest.pt"):
        if not self.model or self.E is None or self.optimizer_W is None:
            print("Warning: Cannot save checkpoint, trainer not fully initialized.")
            return

        state = {
            "step": self.current_step,
            "config": self.config,
            "vocab_size": self.vocab_size,
            "model_state_dict": self.model.state_dict(),
            "E": self.E.detach().cpu(),
            "P": self.P.detach().cpu(),
            "optimizer_W_state_dict": self.optimizer_W.state_dict(),
            "train_losses_l1": self.train_losses_l1,
            "train_losses_l2": self.train_losses_l2,
            "test_losses": self.test_losses,
            "debug_stats": self.debug_stats,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
        }
        filepath = os.path.join(self.save_dir, filename)
        try:
            torch.save(state, filepath)
            print(f"Checkpoint saved to {filepath} at step {self.current_step}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=device)
            print(f"Loading checkpoint from {filepath}...")

            self.config = checkpoint["config"]
            self.vocab_size = checkpoint["vocab_size"]
            self.char_to_idx = checkpoint["char_to_idx"]
            self.idx_to_char = checkpoint["idx_to_char"]
            self.integrator = make_integrator(self.config.get("integrator", "leapfrog"))

            self.model = RNNCharModel(
                vocab_size=self.vocab_size,
                embedding_dim=self.config["embedding_dim"],
                hidden_dim=self.config["hidden_dim"],
                n_layers=self.config["n_layers"],
            ).to(device)
            self.optimizer_W = optim.Adam(
                self.model.parameters(), lr=self.config["learning_rate_W"]
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.E = checkpoint["E"].to(device).requires_grad_(True)
            self.P = checkpoint["P"].to(device)
            self.optimizer_W.load_state_dict(checkpoint["optimizer_W_state_dict"])
            self.current_step = checkpoint["step"]
            self.train_losses_l1 = checkpoint["train_losses_l1"]
            self.train_losses_l2 = checkpoint["train_losses_l2"]
            self.test_losses = checkpoint["test_losses"]
            loaded_debug = checkpoint["debug_stats"]
            # Back-compat: older checkpoints had no H_drift or dt columns.
            loaded_debug.setdefault("H_drift", [0.0] * len(self.test_losses))
            loaded_debug.setdefault(
                "dt", [float(self.config.get("dt", 0.01))] * len(self.test_losses)
            )
            self.debug_stats = loaded_debug
            self.adaptive = self._build_adaptive_controller(self.config)
            self._current_dt = float(self.config["dt"])
            if self.adaptive is not None:
                self.adaptive.dt = self._current_dt
            self.save_dir = os.path.dirname(filepath)
            self._energy_reference = None

            print(f"Checkpoint loaded. Resuming from step {self.current_step}")
            return True

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def stop_training(self):
        print("Stop training requested.")
        self._stop_training_flag = True

    def get_history(self):
        eval_interval = self.config.get("eval_interval", 1)
        steps = list(np.arange(len(self.test_losses)) * eval_interval)
        return {
            "steps": steps,
            "train_l1": self.train_losses_l1,
            "train_l2": self.train_losses_l2,
            "test_l": self.test_losses,
            "grad_W": self.debug_stats["grad_W"],
            "force_E": self.debug_stats["force_E"],
            "P_norm": self.debug_stats["P_norm"],
            "H_drift": self.debug_stats.get("H_drift", []),
            "dt": self.debug_stats.get("dt", []),
        }

    def generate(self, start_prompt, length, temperature):
        if not self.model or self.E is None or self.idx_to_char is None:
            return "Error: Model not loaded or trainer not set up."

        print(f"Generating text (temp={temperature}): '{start_prompt}'")
        was_training = self.model.training
        self.model.eval()
        try:
            generated_indices = [
                self.char_to_idx.get(char, 0) for char in start_prompt
            ]
            hidden = self.model.init_hidden(batch_size=1)

            with torch.no_grad():
                if generated_indices:
                    context = torch.tensor(
                        [generated_indices], dtype=torch.long, device=device
                    )
                    prompt_embeddings = F.embedding(context, self.E.detach())
                    _, hidden = self.model(prompt_embeddings, hidden)
                    current_input_idx = torch.tensor(
                        [[generated_indices[-1]]], dtype=torch.long, device=device
                    )
                else:
                    current_input_idx = torch.randint(
                        0, self.vocab_size, (1, 1), device=device
                    )

                for _ in range(length):
                    emb = F.embedding(current_input_idx, self.E.detach())
                    logits, hidden = self.model(emb, hidden)
                    scaled = logits.squeeze(1) / max(temperature, 1e-6)
                    probs = F.softmax(scaled, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    generated_indices.append(int(next_idx.item()))
                    current_input_idx = next_idx
        finally:
            if was_training:
                self.model.train()

        return "".join(self.idx_to_char.get(i, "?") for i in generated_indices)
