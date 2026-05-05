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
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..data.handling import save_vocab  # noqa: F401 - re-exported for callers
from ..models import make_model
from .action import ActionAccumulator
from .adaptive import AdaptiveStepController
from .diagnostics import measure_reversibility
from .dynamics import make_force_fn, total_hamiltonian
from .integrity import CheckpointIntegrity, compute_integrity, verify_integrity
from .integrators import make_integrator
from .geometry import FisherMassConfig, FisherMassEstimator, fisher_diagonal_sample
from .observables import ObservableWindow, snapshot
from .substrate import HamiltonianSubstrate
from .utils import device, get_batch


def _to_cpu_state(obj):
    """Return a CPU-only copy of tensors inside checkpoint state."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, Mapping):
        return {key: _to_cpu_state(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_cpu_state(value) for value in obj)
    if isinstance(obj, list):
        return [_to_cpu_state(value) for value in obj]
    return obj


def _fmt_config_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


TRAINER_DEFAULTS = {
    "mass_mode": "fisher",
    "adaptive_dt": True,
    "dt_min": 0.00001,
    "dt_max": 0.01,
    "drift_high": 0.05,
    "drift_low": 0.001,
    "snapshot_interval": 5000,
    "snapshot_keep_last": 100,
    "snapshot_on_shift": True,
    "snapshot_shift_cooldown": 5000,
    "snapshot_shift_force_min": 5.0,
    "snapshot_shift_force_ratio": 2.0,
    "snapshot_shift_momentum_delta": 0.5,
    "snapshot_shift_drift_delta": 0.25,
    "snapshot_shift_susceptibility_min": 0.25,
    "snapshot_shift_susceptibility_ratio": 2.0,
    "reversibility_check_interval": 10,
    "reversibility_check_steps": 5,
}


def _with_trainer_defaults(config: Mapping) -> dict:
    merged = dict(TRAINER_DEFAULTS)
    merged.update(config)
    return merged


class Trainer:
    def __init__(self, config):
        self.config = _with_trainer_defaults(config)
        config = self.config
        self.model = None
        self.E = None
        self.P = None
        self.substrate: HamiltonianSubstrate | None = None
        self.optimizer_W = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mass_vector = None

        self.train_data = None
        self.test_data = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None

        self.integrator = make_integrator(
            config.get("integrator", "leapfrog"),
            **config.get("integrator_params", {}),
        )
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
            "reversibility": [],
            "specific_heat": [],
            "susceptibility": [],
            "corr_time": [],
            "entropy_rate": [],
            "action": [],
            "lagrangian": [],
        }
        self._eval_counter = 0
        self._energy_reference = None
        self._last_snapshot_step = 0
        self._last_shift_snapshot_step = 0
        initial_dt = float(config["dt"]) if "dt" in config else 0.01
        self._current_dt: float = (
            float(self.adaptive.dt) if self.adaptive is not None else initial_dt
        )
        self._observables = ObservableWindow(
            size=int(config.get("observable_window", 32))
        )
        self._action = ActionAccumulator()
        self._fisher: FisherMassEstimator | None = None
        self._fisher_cfg: FisherMassConfig | None = None

        self.run_id = f"run_{time.strftime('%Y%m%d-%H%M%S')}"

        self.save_dir = os.path.join(config.get("results_dir", "results"), self.run_id)
        os.makedirs(self.save_dir, exist_ok=True)

        self._stop_training_flag = False
        self._step_delay = float(config.get("step_delay", 0.0))
        self._safe_speed = config.get("safe_speed", True)  # Default on for stability
        default_sync = 1 if self._safe_speed else 20
        self._cuda_sync_interval = max(
            1, int(config.get("cuda_sync_interval", default_sync))
        )

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

    def _sync_substrate(self) -> None:
        """Keep the explicit phase substrate aligned with trainer state."""
        if self.E is None or self.P is None or self.mass_vector is None:
            self.substrate = None
            return
        if self.substrate is None:
            self.substrate = HamiltonianSubstrate(
                E=self.E,
                P=self.P,
                mass=self.mass_vector,
                integrator=self.integrator,
                dt=self._current_dt,
            )
            return
        self.substrate.E = self.E
        self.substrate.P = self.P
        self.substrate.configure(
            mass=self.mass_vector,
            integrator=self.integrator,
            dt=self._current_dt,
        )

    def effective_config_lines(self) -> list[str]:
        """Return the runtime settings that materially affect a train run."""
        lines = ["Effective trainer config:"]

        data_bits = [f"device={device}", f"vocab_size={self.vocab_size}"]
        if self.train_data is not None:
            data_bits.append(f"train_tokens={len(self.train_data)}")
        if self.test_data is not None:
            data_bits.append(f"test_tokens={len(self.test_data)}")
        data_file = self.config.get("data_file")
        if data_file:
            data_bits.append(f"data_file={data_file}")
        lines.append("  data: " + ", ".join(data_bits))

        groups = (
            ("model", ("model_type", "embedding_dim", "hidden_dim", "n_layers")),
            (
                "training",
                (
                    "context_window",
                    "batch_size",
                    "num_steps",
                    "eval_interval",
                    "eval_iters",
                    "eval_batch_size",
                    "snapshot_interval",
                    "snapshot_keep_last",
                    "snapshot_on_shift",
                    "snapshot_shift_cooldown",
                    "learning_rate_W",
                    "gradient_clip_norm_W",
                    "reversibility_check_interval",
                    "reversibility_check_steps",
                ),
            ),
            (
                "dynamics",
                (
                    "integrator",
                    "mass_type",
                    "mass_mode",
                    "dt",
                    "adaptive_dt",
                    "dt_min",
                    "dt_max",
                    "drift_high",
                    "drift_low",
                ),
            ),
            (
                "safety",
                (
                    "safe_speed",
                    "step_delay",
                    "cuda_sync_interval",
                    "cudnn_benchmark",
                    "disable_cudnn",
                ),
            ),
        )
        for label, keys in groups:
            parts = [
                f"{key}={_fmt_config_value(self.config[key])}"
                for key in keys
                if key in self.config
            ]
            if parts:
                lines.append(f"  {label}: " + ", ".join(parts))

        if self.E is not None and self.P is not None:
            lines.append(f"  state: E={tuple(self.E.shape)}, P={tuple(self.P.shape)}")
        if self.mass_vector is not None:
            mass = self.mass_vector.detach()
            lines.append(
                "  mass: "
                f"shape={tuple(mass.shape)}, "
                f"min={mass.min().item():.6g}, "
                f"max={mass.max().item():.6g}, "
                f"mean={mass.mean().item():.6g}"
            )
        return lines

    def print_effective_config(self) -> None:
        for line in self.effective_config_lines():
            print(line)

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
        # Keep corpus tensors on CPU. Large text files can consume enough VRAM
        # to starve the model and trigger Windows TDR resets on display GPUs.
        self.train_data = torch.as_tensor(train_data, dtype=torch.long).detach().cpu()
        self.test_data = torch.as_tensor(test_data, dtype=torch.long).detach().cpu()
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        self.mass_vector = mass_vector.to(device)

        # cuDNN benchmarking can launch long autotune kernels on first use. Keep
        # it opt-in so the default path is safer on Windows display adapters.
        if device.type == "cuda":
            torch.backends.cudnn.enabled = not bool(
                self.config.get("disable_cudnn", False)
            )
            torch.backends.cudnn.benchmark = bool(
                self.config.get("cudnn_benchmark", False)
                and torch.backends.cudnn.enabled
            )
            cudnn_mode = "enabled" if torch.backends.cudnn.enabled else "disabled"
            mode = "enabled" if torch.backends.cudnn.benchmark else "disabled"
            print(f"cuDNN {cudnn_mode}; benchmarking {mode}.")

        self.model = make_model(self.vocab_size, self.config).to(device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model instantiated with {n_params:,} params.")

        self.substrate = HamiltonianSubstrate.random(
            self.vocab_size,
            self.config["embedding_dim"],
            mass=self.mass_vector,
            integrator=self.integrator,
            dt=float(self.config["dt"]),
            device=device,
        )
        self.E = self.substrate.E
        self.P = self.substrate.P
        print(f"E: {self.E.shape}, P: {self.P.shape}")

        self.optimizer_W = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate_W"]
        )
        print(
            f"Optimizer W: Adam lr={self.config['learning_rate_W']} "
            f"| integrator: {self.integrator.name} (order {self.integrator.order})"
        )

        if self.config.get("mass_mode") == "fisher":
            self._fisher_cfg = FisherMassConfig(
                beta=float(self.config.get("fisher_beta", 0.9)),
                eps=float(self.config.get("fisher_eps", 1e-3)),
                refresh_every=int(self.config.get("fisher_refresh_every", 50)),
                batches_per_refresh=int(
                    self.config.get("fisher_batches_per_refresh", 4)
                ),
            )
            self._fisher = FisherMassEstimator(
                shape=(self.vocab_size, self.config["embedding_dim"]),
                config=self._fisher_cfg,
            )
            self._fisher.initialise(device, self.E.dtype)
            self.mass_vector = self._fisher.current()
            print(
                f"Fisher mass: eps={self._fisher_cfg.eps} "
                f"beta={self._fisher_cfg.beta} "
                f"refresh_every={self._fisher_cfg.refresh_every}"
            )
        else:
            self._fisher = None
            self._fisher_cfg = None

        self.current_step = 0
        self._stop_training_flag = False
        self._energy_reference = None
        self.adaptive = self._build_adaptive_controller(self.config)
        self._current_dt = (
            float(self.adaptive.dt)
            if self.adaptive is not None
            else float(self.config["dt"])
        )
        if self.adaptive is not None:
            self.adaptive.dt = self._current_dt
        default_sync = 1 if self._safe_speed else 20
        self._cuda_sync_interval = max(
            1, int(self.config.get("cuda_sync_interval", default_sync))
        )
        self._observables.clear()
        self._action.reset()
        self._last_snapshot_step = 0
        self._last_shift_snapshot_step = 0
        self._sync_substrate()
        self.print_effective_config()
        print("Trainer setup complete.")

    def _breathe(self, step: int) -> None:
        if not self._safe_speed:
            if self._step_delay > 0:
                time.sleep(self._step_delay)
            return
        if device.type == "cuda" and step % self._cuda_sync_interval == 0:
            torch.cuda.synchronize()
        if self._step_delay > 0:
            time.sleep(self._step_delay)
        else:
            time.sleep(0.001)

    def _refresh_fisher_mass(self) -> None:
        """Blend a fresh diagonal empirical Fisher into the mass estimator."""
        if self._fisher is None or self._fisher_cfg is None:
            return
        cfg = self._fisher_cfg
        accum = torch.zeros_like(self.E)
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            for _ in range(cfg.batches_per_refresh):
                try:
                    x, y = get_batch(
                        self.train_data,
                        self.config["context_window"],
                        self.config["batch_size"],
                        device,
                    )
                except ValueError:
                    break
                accum += fisher_diagonal_sample(
                    self.model, self.criterion, self.E, x, y
                )
            accum /= max(cfg.batches_per_refresh, 1)
            self._fisher.update(accum)
            self.mass_vector = self._fisher.current()
        finally:
            if was_training:
                self.model.train()

    def _run_reversibility_probe(self, x_batch, y_batch) -> float:
        """Integrate forward/backward on the current batch and return residual."""
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            frozen_params = [p.requires_grad for p in self.model.parameters()]
            for p in self.model.parameters():
                p.requires_grad_(False)
            try:
                force_fn = make_force_fn(
                    x_batch, y_batch, self.model, self.criterion
                )
                residual = measure_reversibility(
                    self.integrator,
                    self.E.detach(),
                    self.P.detach(),
                    self.mass_vector,
                    force_fn,
                    dt=self._current_dt,
                    n_steps=int(self.config.get("reversibility_check_steps", 5)),
                )
                return residual.total
            finally:
                for p, was in zip(self.model.parameters(), frozen_params):
                    p.requires_grad_(was)
        finally:
            if was_training:
                self.model.train()

    def _step_embeddings(self, x_batch, y_batch):
        """Run one integrator step and return (new_E, new_P, loss1, loss2)."""
        force_fn = make_force_fn(x_batch, y_batch, self.model, self.criterion)
        self._sync_substrate()
        if self.substrate is None:
            raise RuntimeError("Hamiltonian substrate is not initialized.")
        step = self.substrate.propose(force_fn, retain_final=True)
        return step.E, step.P, step.loss_initial, step.loss_final

    def train(self):
        if not self.model or self.E is None:
            yield {"status": "error", "message": "Trainer not set up"}
            return

        self._stop_training_flag = False
        start_time = time.time()
        self.model.train()
        self.E.requires_grad_(True)

        if self._fisher is not None and self._fisher.update_count == 0:
            self._refresh_fisher_mass()

        print(f"Starting training from step {self.current_step}...")
        self.print_effective_config()

        for step in range(self.current_step, self.config["num_steps"]):
            if self._stop_training_flag:
                print(f"Training stopped externally at step {step}.")
                yield {"status": "stopped", "step": step}
                break

            # Fisher mass is now updated amortized inside _step_embeddings.
            # We no longer need periodic lump refreshes.

            x_batch, y_batch = get_batch(
                self.train_data,
                self.config["context_window"],
                self.config["batch_size"],
                device,
            )

            try:
                P_prev = self.P
                E_new, P_new, loss_l1, loss_l2 = self._step_embeddings(x_batch, y_batch)

                # Update Fisher mass amortized once per step (fast, in-GPU EMA)
                if self._fisher is not None:
                    with torch.no_grad():
                        force = (P_new - P_prev) / (-self._current_dt)
                        self._fisher.update(force * force)
                        self.mass_vector = self._fisher.current()
            except Exception as e:
                print(f"ERROR during integrator step {step}: {e}")
                yield {"status": "error", "message": f"Integrator step failed: {e}"}
                break

            if torch.isnan(loss_l1) or torch.isnan(loss_l2):
                print(f"ERROR: NaN loss detected at step {step}. Stopping training.")
                yield {"status": "error", "message": f"NaN loss at step {step}"}
                break

            self.optimizer_W.zero_grad()
            try:
                loss_l2.backward()
            except RuntimeError as e:
                print(f"ERROR during W backward pass at step {step}: {e}")
                yield {"status": "error", "message": f"W backward failed: {e}"}
                break

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config["gradient_clip_norm_W"],
            )
            self.optimizer_W.step()
            self.E = E_new.detach().requires_grad_(True)
            self.P = P_new.detach()
            if self.substrate is not None:
                self.substrate.E = self.E
                self.substrate.P = self.P
            self.current_step = step + 1
            self._breathe(step)

            # --- Telemetry & Evaluation (Only on eval steps) ---
            eval_every = self.config["eval_interval"]
            is_eval_step = (
                (step + 1) % eval_every == 0
                or (step + 1) == self.config["num_steps"]
            )


            if not is_eval_step:
                continue

            # Heavy telemetry and evaluation only on eval steps
            with torch.no_grad():
                dt = self._current_dt
                force_eff_norm = (torch.linalg.norm((P_new - P_prev) / dt).item() if dt else 0.0)
                p_norm = torch.linalg.norm(self.P).item()
                total_norm_W_before = (
                    sum(p.grad.data.norm(2).item() ** 2
                        for p in self.model.parameters() if p.grad is not None) ** 0.5
                )
                energy = total_hamiltonian(self.P, self.mass_vector, loss_l2)
                if self._energy_reference is None:
                    self._energy_reference = energy
                h_drift = energy.drift(self._energy_reference)

                self._observables.push(snapshot(self.E, H=energy.total))
                action_report = self._action.update(
                    kinetic=energy.kinetic, potential=energy.potential, dt=self._current_dt
                )
                if self.adaptive is not None:
                    self._current_dt = self.adaptive.observe(h_drift)

            eval_results = self.evaluate()
            test_loss = eval_results["test_loss"]

            self._eval_counter += 1
            rev_interval = int(self.config.get("reversibility_check_interval", 0))
            if rev_interval and self._eval_counter % rev_interval == 0:
                rev_residual = self._run_reversibility_probe(x_batch, y_batch)
            else:
                rev_residual = float("nan")

            obs_report = self._observables.report()
            if obs_report is not None:
                c_v, chi, tau, s_rate = obs_report.specific_heat, obs_report.susceptibility, obs_report.corr_time, obs_report.entropy_rate
            else:
                c_v = chi = tau = s_rate = float("nan")

            self.train_losses_l1.append(loss_l1.item())
            self.train_losses_l2.append(loss_l2.item())
            self.test_losses.append(test_loss)
            self.debug_stats["grad_W"].append(total_norm_W_before)
            self.debug_stats["force_E"].append(force_eff_norm)
            self.debug_stats["P_norm"].append(p_norm)
            self.debug_stats["H_drift"].append(h_drift)
            self.debug_stats["dt"].append(self._current_dt)
            self.debug_stats["reversibility"].append(rev_residual)
            self.debug_stats["specific_heat"].append(c_v)
            self.debug_stats["susceptibility"].append(chi)
            self.debug_stats["corr_time"].append(tau)
            self.debug_stats["entropy_rate"].append(s_rate)
            self.debug_stats["action"].append(action_report.S)
            self.debug_stats["lagrangian"].append(action_report.L)

            elapsed = time.time() - start_time
            log_message = (
                f"Step {self.current_step}/{self.config['num_steps']} | T: {elapsed:.1f}s "
                f"| L1: {loss_l1.item():.4f} | L2->W: {loss_l2.item():.4f} | Test L: {test_loss:.4f} "
                f"| grad_W: {total_norm_W_before:.3f} | F_E: {force_eff_norm:.3f} "
                f"| P_norm: {p_norm:.3f} | dH/H: {h_drift:.3e} | dt: {self._current_dt:.3e}"
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
            self._maybe_save_snapshot()
            self._maybe_save_shift_snapshot()


        if not self._stop_training_flag:
            print("\nTraining finished.")
            yield {
                "status": "finished",
                "step": self.current_step,
                "history": self.get_history(),
            }

        # Final cleanup
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

                # Tiny breath for the GPU/OS every 25 eval batches
                if k % 25 == 0:
                    time.sleep(0.001)

                # Each batch is sampled from random positions, so the hidden
                # state must be reset per batch â€” persisting it across
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

    def _snapshot_filenames(self) -> list[str]:
        if not os.path.isdir(self.save_dir):
            return []
        names = []
        for name in os.listdir(self.save_dir):
            if name.startswith("checkpoint_step_") and name.endswith(".pt"):
                names.append(name)
        return sorted(names)

    def _prune_snapshots(self) -> None:
        keep_last = int(self.config.get("snapshot_keep_last", 0) or 0)
        if keep_last <= 0:
            return
        names = self._snapshot_filenames()
        stale = names[:-keep_last]
        for name in stale:
            path = os.path.join(self.save_dir, name)
            try:
                os.remove(path)
            except OSError as e:
                print(f"Warning: Could not prune snapshot {path}: {e}")

    @staticmethod
    def _finite(value: float | None) -> bool:
        return value is not None and np.isfinite(float(value))

    def _phase_shift_reason(self) -> str | None:
        if not bool(self.config.get("snapshot_on_shift", False)):
            return None
        if len(self.debug_stats["H_drift"]) < 2:
            return None

        cur = {key: self.debug_stats[key][-1] for key in self.debug_stats}
        prev = {key: self.debug_stats[key][-2] for key in self.debug_stats}
        reasons = []

        force_min = float(self.config.get("snapshot_shift_force_min", 5.0))
        force_ratio = float(self.config.get("snapshot_shift_force_ratio", 2.0))
        cur_f = cur.get("force_E")
        prev_f = prev.get("force_E")
        if (
            self._finite(cur_f)
            and self._finite(prev_f)
            and float(prev_f) > 1e-12
            and float(cur_f) >= force_min
            and float(cur_f) / float(prev_f) >= force_ratio
        ):
            reasons.append("force_jump")

        p_delta = float(self.config.get("snapshot_shift_momentum_delta", 0.5))
        cur_p = cur.get("P_norm")
        prev_p = prev.get("P_norm")
        if (
            self._finite(cur_p)
            and self._finite(prev_p)
            and abs(float(cur_p) - float(prev_p)) >= p_delta
        ):
            reasons.append("momentum_jump")

        drift_delta = float(self.config.get("snapshot_shift_drift_delta", 0.25))
        cur_h = cur.get("H_drift")
        prev_h = prev.get("H_drift")
        if (
            self._finite(cur_h)
            and self._finite(prev_h)
            and abs(float(cur_h) - float(prev_h)) >= drift_delta
        ):
            reasons.append("drift_jump")

        chi_ratio = float(self.config.get("snapshot_shift_susceptibility_ratio", 2.0))
        chi_min = float(self.config.get("snapshot_shift_susceptibility_min", 0.25))
        cur_chi = cur.get("susceptibility")
        prev_chi = prev.get("susceptibility")
        if (
            self._finite(cur_chi)
            and self._finite(prev_chi)
            and float(prev_chi) > 1e-12
            and float(cur_chi) >= chi_min
            and float(cur_chi) / float(prev_chi) >= chi_ratio
        ):
            reasons.append("susceptibility_jump")

        if not reasons:
            return None
        return "_".join(reasons[:2])

    def _maybe_save_shift_snapshot(self) -> None:
        reason = self._phase_shift_reason()
        if reason is None:
            return
        cooldown = int(self.config.get("snapshot_shift_cooldown", 0) or 0)
        if (
            cooldown > 0
            and self.current_step - self._last_shift_snapshot_step < cooldown
        ):
            return
        filename = f"checkpoint_shift_{self.current_step:08d}_{reason}.pt"
        self.save_checkpoint(filename, kind="phase_shift")
        self._last_shift_snapshot_step = self.current_step

    def _maybe_save_snapshot(self) -> None:
        interval = int(self.config.get("snapshot_interval", 0) or 0)
        if interval <= 0 or self.current_step <= 0:
            return
        is_final = self.current_step >= int(self.config.get("num_steps", 0))
        if (
            not is_final
            and self.current_step - self._last_snapshot_step < interval
        ):
            return
        filename = f"checkpoint_step_{self.current_step:08d}.pt"
        self.save_checkpoint(filename, kind="snapshot")
        self._last_snapshot_step = self.current_step
        self._prune_snapshots()

    def save_checkpoint(self, filename="checkpoint_latest.pt", *, kind="latest"):
        if not self.model or self.E is None or self.optimizer_W is None:
            print("Warning: Cannot save checkpoint, trainer not fully initialized.")
            return

        E_cpu = self.E.detach().cpu()
        P_cpu = self.P.detach().cpu()
        mass_cpu = self.mass_vector.detach().cpu()
        integrity = compute_integrity(E_cpu, P_cpu, mass_cpu)
        state = {
            "step": self.current_step,
            "checkpoint_kind": kind,
            "config": self.config,
            "vocab_size": self.vocab_size,
            "model_state_dict": _to_cpu_state(self.model.state_dict()),
            "E": E_cpu,
            "P": P_cpu,
            "mass_vector": mass_cpu,
            "integrity": integrity.to_dict(),
            "optimizer_W_state_dict": _to_cpu_state(
                self.optimizer_W.state_dict()
            ),
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
            checkpoint = torch.load(
                filepath, map_location="cpu", weights_only=False
            )
            print(f"Loading checkpoint from {filepath}...")

            saved_fp = checkpoint.get("integrity")
            if saved_fp is not None:
                mass_cpu = checkpoint.get("mass_vector")
                if mass_cpu is None:
                    print("Checkpoint has integrity fingerprint but no mass_vector; refusing.")
                    return False
                fresh = compute_integrity(
                    checkpoint["E"], checkpoint["P"], mass_cpu
                )
                report = verify_integrity(
                    CheckpointIntegrity.from_dict(saved_fp), fresh
                )
                if not report.ok:
                    print(report.format())
                    return False
                print(report.format())

            loaded_config = dict(checkpoint["config"])
            self.config = _with_trainer_defaults(loaded_config)
            if (
                "mass_mode" not in loaded_config
                and "mass_vector" in checkpoint
                and checkpoint["mass_vector"] is not None
                and tuple(checkpoint["mass_vector"].shape) != tuple(checkpoint["E"].shape)
            ):
                self.config["mass_mode"] = "static"
            self.vocab_size = checkpoint["vocab_size"]
            self.char_to_idx = checkpoint["char_to_idx"]
            self.idx_to_char = checkpoint["idx_to_char"]
            self.integrator = make_integrator(
                self.config.get("integrator", "leapfrog"),
                **self.config.get("integrator_params", {}),
            )

            self.model = make_model(self.vocab_size, self.config).to(device)
            self.optimizer_W = optim.Adam(
                self.model.parameters(), lr=self.config["learning_rate_W"]
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.E = checkpoint["E"].to(device).requires_grad_(True)
            self.P = checkpoint["P"].to(device)
            if "mass_vector" in checkpoint and checkpoint["mass_vector"] is not None:
                self.mass_vector = checkpoint["mass_vector"].to(device)
            if self.config.get("mass_mode") == "fisher":
                self._fisher_cfg = FisherMassConfig(
                    beta=float(self.config.get("fisher_beta", 0.9)),
                    eps=float(self.config.get("fisher_eps", 1e-3)),
                    refresh_every=int(self.config.get("fisher_refresh_every", 50)),
                    batches_per_refresh=int(
                        self.config.get("fisher_batches_per_refresh", 4)
                    ),
                )
                self._fisher = FisherMassEstimator(
                    shape=tuple(self.mass_vector.shape),
                    config=self._fisher_cfg,
                )
                self._fisher.mass = self.mass_vector
            else:
                self._fisher = None
                self._fisher_cfg = None
            self.optimizer_W.load_state_dict(checkpoint["optimizer_W_state_dict"])
            self.current_step = checkpoint["step"]
            self.train_losses_l1 = checkpoint["train_losses_l1"]
            self.train_losses_l2 = checkpoint["train_losses_l2"]
            self.test_losses = checkpoint["test_losses"]
            loaded_debug = checkpoint["debug_stats"]
            # Back-compat: older checkpoints may be missing any of these columns.
            n = len(self.test_losses)
            loaded_debug.setdefault("H_drift", [0.0] * n)
            loaded_debug.setdefault(
                "dt", [float(self.config.get("dt", 0.01))] * n
            )
            loaded_debug.setdefault("reversibility", [float("nan")] * n)
            for key in (
                "specific_heat",
                "susceptibility",
                "corr_time",
                "entropy_rate",
                "action",
                "lagrangian",
            ):
                loaded_debug.setdefault(key, [float("nan")] * n)
            self.debug_stats = loaded_debug
            self.adaptive = self._build_adaptive_controller(self.config)
            self._current_dt = (
                float(self.adaptive.dt)
                if self.adaptive is not None
                else float(self.config["dt"])
            )
            self.save_dir = os.path.dirname(filepath)
            self._energy_reference = None
            self._observables.clear()
            self._action.reset()
            self._last_snapshot_step = self.current_step
            self._last_shift_snapshot_step = self.current_step
            self._sync_substrate()

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
            "reversibility": self.debug_stats.get("reversibility", []),
            "specific_heat": self.debug_stats.get("specific_heat", []),
            "susceptibility": self.debug_stats.get("susceptibility", []),
            "corr_time": self.debug_stats.get("corr_time", []),
            "entropy_rate": self.debug_stats.get("entropy_rate", []),
            "action": self.debug_stats.get("action", []),
            "lagrangian": self.debug_stats.get("lagrangian", []),
        }

    def generate(self, start_prompt, length, temperature):
        """Sample ``length`` new characters after ``start_prompt``.

        Each decoding step feeds the full tail of the generated sequence
        to the model and takes the last-position logits. This works
        uniformly for the RNN (which could use incremental hidden state
        but doesn't need to for correctness) and the transformer (which
        requires the full context window every step).
        """
        if not self.model or self.E is None or self.idx_to_char is None:
            return "Error: Model not loaded or trainer not set up."

        print(f"Generating text (temp={temperature}): '{start_prompt}'")
        was_training = self.model.training
        self.model.eval()

        max_ctx = getattr(
            self.model, "max_context", self.config.get("context_window", 1024)
        )

        try:
            generated = [self.char_to_idx.get(ch, 0) for ch in start_prompt]
            if not generated:
                generated = [int(torch.randint(0, self.vocab_size, (1,)).item())]

            with torch.no_grad():
                for _ in range(length):
                    window = generated[-max_ctx:]
                    context = torch.tensor(
                        [window], dtype=torch.long, device=device
                    )
                    emb = F.embedding(context, self.E.detach())
                    hidden = self.model.init_hidden(batch_size=1)
                    logits, _ = self.model(emb, hidden)
                    last_logits = logits[:, -1, :]
                    scaled = last_logits / max(temperature, 1e-6)
                    probs = F.softmax(scaled, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    generated.append(int(next_idx.item()))
        finally:
            if was_training:
                self.model.train()

        return "".join(self.idx_to_char.get(i, "?") for i in generated)
