import gradio as gr
import sys
import os
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjust path to import from parent directory modules if running as script
# This allows importing yaum.* modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Imports from your project ---
try:
    from yaum.data.handling import load_and_split_data, calculate_mass_vector, save_vocab, load_vocab
    from yaum.core.trainer import Trainer
    from yaum.core.utils import device # Import device info
except ImportError as e:
     print(f"Error importing project modules: {e}")
     print("Please ensure the script is run from the project root directory (e.g., using 'python -m yaum.ui.app') or the PYTHONPATH is set correctly.")
     sys.exit(1)

# --- Global State ---
# Using global state for simplicity. For more complex apps, consider
# dependency injection or Gradio's State management more extensively.
APP_STATE = {
    "trainer": None,
    "config": {},
    "data_loaded": False,
    "model_ready": False,
    "training_thread": None,
    "latest_log": "Welcome! Configure settings and prepare data.",
    "log_history": ["Welcome! Configure settings and prepare data."],  # Add this line to store log history
    "latest_history": None, # Store history fetched by generator
    "thread_active": False,  # Flag to track if thread is active
}

# --- Default Configuration ---
DEFAULT_CONFIG = {
    'data_file': '',
    'test_split_ratio': 0.1,
    'embedding_dim': 64,
    'context_window': 200,
    'hidden_dim': 512,
    'n_layers': 3,
    'dt': 0.01,
    'mass_type': 'frequency',
    'num_steps': 15000,
    'batch_size': 64,
    'learning_rate_W': 0.001,
    'eval_interval': 500,
    'gradient_clip_norm_W': 1.0,
    'results_dir': 'results', # Base directory for saving runs
    'checkpoint_to_load': None, # Path to a specific checkpoint
}
APP_STATE["config"] = DEFAULT_CONFIG.copy() # Initialize state with defaults

def append_to_log(message):
    """Adds a message to the log history and updates the latest log."""
    if message:
        APP_STATE["log_history"].append(str(message))
        # Keep log history at a reasonable size (e.g., last 100 messages)
        if len(APP_STATE["log_history"]) > 100:
            APP_STATE["log_history"] = APP_STATE["log_history"][-100:]
        APP_STATE["latest_log"] = str(message)  # Also update latest_log for status display

# --- File Browser Functions ---
def browse_data_file():
    """Opens a file browser and returns the selected file path"""
    # Check if a training thread is active first
    if APP_STATE.get("thread_active"):
        print("Cannot open file browser while training is active")
        return APP_STATE["config"].get("data_file", "")
        
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring the dialog to the front
        
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        # Update config if a file was selected
        if file_path:
            APP_STATE["config"]["data_file"] = file_path
            return file_path
        return APP_STATE["config"].get("data_file", "")
    except Exception as e:
        print(f"Error in file browser: {e}")
        return APP_STATE["config"].get("data_file", "")

def browse_checkpoint_file():
    """Opens a file browser and returns the selected checkpoint file path"""
    # Check if a training thread is active first
    if APP_STATE.get("thread_active"):
        print("Cannot open file browser while training is active")
        return ""
        
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring the dialog to the front
        
        file_path = filedialog.askopenfilename(
            title="Select Checkpoint File",
            filetypes=[("PyTorch Files", "*.pt"), ("All Files", "*.*")]
        )
        
        # Just return the path, don't update config here
        return file_path if file_path else ""
    except Exception as e:
        print(f"Error in file browser: {e}")
        return ""

# --- UI Helper Functions ---
def update_config_value(key, value):
    """Updates a single value in the global config state."""
    if value is not None:
         # Basic type casting for numerical inputs from Gradio
         if key in ['embedding_dim', 'context_window', 'hidden_dim', 'n_layers', 'num_steps', 'batch_size', 'eval_interval']:
             try: value = int(value)
             except (ValueError, TypeError): value = DEFAULT_CONFIG.get(key) # Reset on error
         elif key in ['test_split_ratio', 'dt', 'learning_rate_W', 'gradient_clip_norm_W']:
             try: value = float(value)
             except (ValueError, TypeError): value = DEFAULT_CONFIG.get(key) # Reset on error
         APP_STATE["config"][key] = value
    # Return the possibly updated value to the UI component
    return APP_STATE["config"].get(key, DEFAULT_CONFIG.get(key))

def get_config_values(*keys):
     """Gets multiple config values for updating UI components."""
     return [APP_STATE["config"].get(k, DEFAULT_CONFIG.get(k)) for k in keys]

def update_status(text):
    """Updates the global status message and log history."""
    append_to_log(str(text)) # Add to log history
    # This function is called by button clicks, return value updates status textbox
    return str(text)

def plot_training_history(history):
    """Generates Matplotlib plots from training history dictionary."""
    # Check if history is valid and has steps
    if not history or not isinstance(history, dict) or not history.get('steps'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, 'No training data yet', ha='center', va='center', fontsize=9)
        ax.axis('off')
        # Close the figure explicitly BEFORE returning it
        plt.close(fig)
        return fig # Return the closed figure object (Gradio might handle this)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True) # Use constrained_layout
    fig.suptitle("Training Metrics", fontsize=12)
    steps = history.get('steps', [])

    # --- Helper function to filter NaN/Inf and check if data exists ---
    def filter_valid(data):
        if not data: return []
        return [x for x in data if x is not None and np.isfinite(x)]

    # --- Plotting Data ---
    train_l1 = filter_valid(history.get('train_l1', []))
    train_l2 = filter_valid(history.get('train_l2', []))
    test_l = filter_valid(history.get('test_l', []))
    grad_w = filter_valid(history.get('grad_W', []))
    force_e = filter_valid(history.get('force_E', []))
    p_norm = filter_valid(history.get('P_norm', []))

    # Align steps with filtered data if lengths mismatch (e.g., if NaN occurred mid-list)
    steps_l = steps[:len(test_l)] if test_l else steps
    steps_w = steps[:len(grad_w)] if grad_w else steps
    steps_e = steps[:len(force_e)] if force_e else steps
    steps_p = steps[:len(p_norm)] if p_norm else steps


    # Loss Plot
    ax = axes[0, 0]
    if steps_l and test_l:
        if train_l1: ax.plot(steps_l, train_l1[:len(steps_l)], label='L1', alpha=0.7, linewidth=1) # Align length
        if train_l2: ax.plot(steps_l, train_l2[:len(steps_l)], label='L2->W', alpha=0.7, linewidth=1) # Align length
        ax.plot(steps_l, test_l, label='Test Loss', linewidth=1.5, color='red')
        ax.set_title('Losses', fontsize=10)
        ax.set_ylabel('Loss', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)
        min_loss_plot = min(test_l) * 0.9
        max_loss_plot = max(test_l) * 1.1
        if max_loss_plot > min_loss_plot * 20: max_loss_plot = min_loss_plot * 20
        ax.set_ylim(bottom=min_loss_plot, top=max_loss_plot)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    else:
        ax.text(0.5, 0.5, 'Loss data pending', ha='center', va='center', fontsize=9); ax.axis('off')

    # Grad W Plot
    ax = axes[0, 1]
    if steps_w and grad_w:
        ax.plot(steps_w, grad_w, label='||∇W|| (pre)', linewidth=1)
        ax.set_title('W Gradient Norm', fontsize=10)
        ax.set_ylabel('Norm', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=max(grad_w) * 1.1 if grad_w else 1.0)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    else:
        ax.text(0.5, 0.5, '∇W data pending', ha='center', va='center', fontsize=9); ax.axis('off')

    # Force E Plot
    ax = axes[1, 0]
    if steps_e and force_e:
        ax.plot(steps_e, force_e, label='||F_E||', linewidth=1)
        ax.set_title('Effective Force Norm on E', fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Norm', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=max(force_e) * 1.1 if force_e else 1.0)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    else:
        ax.text(0.5, 0.5, 'F_E data pending', ha='center', va='center', fontsize=9); ax.axis('off')

    # Momentum P Plot
    ax = axes[1, 1]
    if steps_p and p_norm:
        ax.plot(steps_p, p_norm, label='||P||', linewidth=1)
        ax.set_title('Momentum Matrix Norm', fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Norm', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=max(p_norm) * 1.1 if p_norm else 1.0)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    else:
        ax.text(0.5, 0.5, '||P|| data pending', ha='center', va='center', fontsize=9); ax.axis('off')

    # Close the figure BEFORE returning it - this might help with the memory warning
    plt.close(fig)
    return fig


# --- UI Update Fetch Functions ---
def fetch_log_plot_history_update():
    """Fetches updates for Log, Plot, and History State."""
    trainer = APP_STATE.get("trainer")
    log_history = "\n".join(APP_STATE.get("log_history", ["Status unavailable."]))
    history = APP_STATE.get("latest_history") # Use last known history
    fig = None

    # Try to get fresh history if trainer exists
    if trainer:
        try:
            current_history = trainer.get_history()
            if current_history: # Check if history is valid
                history = current_history
                APP_STATE["latest_history"] = history # Update stored history
        except Exception as e:
            print(f"Error getting history from trainer: {e}")
            # Keep using APP_STATE["latest_history"]

    # Plotting logic (using potentially stale history if trainer error)
    if not history:
         history = {'steps': [], 'train_l1': [], 'train_l2': [], 'test_l': [], 'grad_W': [], 'force_E': [], 'P_norm': []}
    try:
         fig = plot_training_history(history)
    except Exception as e:
         print(f"Error plotting history: {e}")
         fig = None # Avoid crashing UI

    # Order: log, fig, history (for state)
    return log_history, fig, history

# We'll update progress through regular UI refresh instead
def update_ui_periodically():
    """Updates UI elements periodically"""
    # This function will be called by interval
    log_history, fig, history = fetch_log_plot_history_update()
    # Calculate progress value if training is active
    progress = None
    trainer = APP_STATE.get("trainer")
    if trainer and APP_STATE.get("thread_active"):
        try:
            current_step = trainer.current_step
            max_steps = trainer.config.get('num_steps', 1)
            if max_steps > 0:
                progress = f"Training: {current_step}/{max_steps} steps ({(current_step/max_steps*100):.1f}%)"
            else:
                progress = f"Training: {current_step} steps"
        except:
            progress = "Training in progress..."
    else:
        if APP_STATE.get("model_ready"):
            progress = "Model ready"
        else:
            progress = "Model not prepared"
    
    return log_history, fig, history, progress

# --- UI Update Loop Generators ---
def refresh_ui():
    """Generator that periodically refreshes the UI."""
    poll_interval = 1
    last_update_time = 0
    while True:
        current_time = time.time()
        if current_time - last_update_time >= poll_interval:
            try:
                log, fig, history, progress = update_ui_periodically()
                last_update_time = current_time
                yield log, fig, history, progress
            except Exception as e:
                print(f"Error in UI refresh: {e}")
                yield "Error refreshing UI", None, {}, "Status unknown"
        else:
            yield gr.update(), gr.update(), gr.update(), gr.update()
        time.sleep(0.2)  # Short sleep for responsive UI

# --------------------------------------------------------------------------
# Backend Logic Functions (Called by UI Callbacks)
# --------------------------------------------------------------------------

def prepare_data_and_model(config_dict_state):
    """Loads data, calculates mass, sets up Trainer instance."""
    # Check if training is active first
    if APP_STATE.get("thread_active"):
        return update_status("Please stop training before preparing new data/model.")
        
    # Make a copy of the config to avoid modifying the original
    config = config_dict_state.copy() if config_dict_state else APP_STATE["config"].copy()
    
    # Rest of function remains the same...
    # Ensure we're using the most up-to-date file path from APP_STATE
    config['data_file'] = APP_STATE["config"].get('data_file', '')
    
    filepath = config.get('data_file')
    print(f"Preparing with file path: {filepath}")
    
    if not filepath:
        return update_status("Error: Data file path is required.")
    if not os.path.exists(filepath):
        return update_status(f"Error: Data file not found: {filepath}")

    try:
        # Stop existing training if running
        if APP_STATE.get("training_thread") and APP_STATE["thread_active"]:
             stop_training_ui()
             time.sleep(0.5) # Give thread a moment to potentially stop

        print("Preparing data and model...")
        train_data, test_data, char_to_idx, idx_to_char, vocab_size = load_and_split_data(
            filepath, config['test_split_ratio']
        )
        if train_data is None:
             return update_status("Error: Failed to load or split data.")

        mass_vector = calculate_mass_vector(
            train_data, vocab_size, config['mass_type'], device
        )

        # Create or reuse Trainer instance, pass current config
        APP_STATE["trainer"] = Trainer(config)
        APP_STATE["trainer"].setup(
            train_data, test_data, char_to_idx, idx_to_char, vocab_size, mass_vector
        )

        APP_STATE["data_loaded"] = True
        APP_STATE["model_ready"] = True
        return update_status(f"Data loaded & Trainer ready. Vocab size: {vocab_size}. Device: {device}.")

    except Exception as e:
        APP_STATE["data_loaded"] = False
        APP_STATE["model_ready"] = False
        print(f"Error during data/model preparation: {e}")
        import traceback
        traceback.print_exc() # Print detailed error
        return update_status(f"Error preparing data/model: {e}")


def run_training_background():
    """Target function for the training thread."""
    if not APP_STATE.get("model_ready") or not APP_STATE.get("trainer"):
        print("Training thread: Trainer not ready.")
        append_to_log("Error: Trainer not ready. Prepare data/model first.")
        APP_STATE["thread_active"] = False
        return

    trainer = APP_STATE["trainer"]
    print("Training thread started...")
    append_to_log("Training thread started...")
    APP_STATE["thread_active"] = True
    
    try:
         # Consume the generator from trainer.train()
         for update in trainer.train():
             # Store the latest log message from the trainer
             if update and isinstance(update, dict) and update.get("log_message"):
                  append_to_log(update["log_message"])
             # Check stop flag frequently inside the trainer's loop
             if trainer._stop_training_flag:
                 print("Training thread: Stop flag detected.")
                 break
                 
         # Update status after loop finishes/breaks
         if trainer._stop_training_flag:
             append_to_log(f"Training stopped at step {trainer.current_step}.")
             print("Training thread finished (stopped).")
         else:
             append_to_log(f"Training finished at step {trainer.current_step}.")
             print("Training thread finished (completed).")
             
    except Exception as e:
         print(f"Exception in training thread: {e}")
         import traceback
         traceback.print_exc()
         append_to_log(f"Training Error: {e}")
         APP_STATE["model_ready"] = False  # Mark as not ready after error
         
    finally:
        # This section ALWAYS runs, even if there's an exception or return above
        APP_STATE["thread_active"] = False  # Mark thread as inactive
        print("Training thread terminated and resources cleaned up.")
        # Make sure any resources are properly released here

def start_training_ui():
    """Starts the training process in a background thread."""
    if not APP_STATE.get("model_ready"):
        return update_status("Error: Prepare Data/Model first.")
    if APP_STATE.get("thread_active"):
        return update_status("Training is already running.")

    # Ensure config in trainer is up-to-date with UI potentially? Or assume prepare was clicked?
    # For safety, let's update trainer's config if it exists
    if APP_STATE.get("trainer"):
         APP_STATE["trainer"].config = APP_STATE["config"].copy()
         print("Updated trainer config before starting.")

    status_msg = "Starting training..."
    print(status_msg)
    append_to_log(status_msg)  # Use append_to_log
    trainer = APP_STATE["trainer"]
    trainer._stop_training_flag = False # Reset flag

    thread = threading.Thread(target=run_training_background, daemon=True)
    APP_STATE["training_thread"] = thread
    thread.start()

    return update_status(status_msg + " (in background)")


def stop_training_ui():
    """Signals the training thread to stop and waits for clean termination."""
    trainer = APP_STATE.get("trainer")
    thread = APP_STATE.get("training_thread")
    
    if trainer and APP_STATE.get("thread_active") and thread:
        status_msg = "Requesting training stop..."
        print(status_msg)
        trainer.stop_training()  # Set the flag
        
        # Give a reasonable timeout for the thread to gracefully terminate
        timeout_seconds = 5.0
        start_time = time.time()
        
        # Use a non-blocking approach to wait for thread termination
        while thread.is_alive() and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)  # Short sleep to avoid busy waiting
        
        # If thread is still running after timeout, it may be stuck
        if thread.is_alive():
            append_to_log("Warning: Training thread taking longer than expected to stop.")
            # Don't force terminate as it could lead to resource leaks,
            # but mark as inactive so UI knows it's in an abnormal state
            APP_STATE["thread_active"] = False
        else:
            append_to_log(f"Training stopped at step {trainer.current_step if hasattr(trainer, 'current_step') else 'unknown'}.")
        
        return update_status(f"Training stopped. You can now use other functions.")
    else:
        return update_status("Training is not currently running.")


# --- Inference Function ---
def run_inference(prompt, length, temperature):
    """Calls the trainer's generate method."""
    trainer = APP_STATE.get("trainer")
    if not trainer or not APP_STATE.get("model_ready"):
        return "Error: Model not ready. Train or load a checkpoint first."
    if not hasattr(trainer, 'generate'):
         return "Error: Trainer object does not have a 'generate' method."

    try:
        # Ensure model components are on the correct device before inference
        trainer.model.to(device)
        if hasattr(trainer, 'E') and trainer.E is not None: 
            trainer.E = trainer.E.to(device)

        length = int(length)
        temperature = float(temperature)
        print(f"Running inference: temp={temperature}, len={length}, prompt='{prompt[:50]}...'")
        result = trainer.generate(prompt, length, temperature)
        print("Inference complete.")
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during inference: {e}"

def clear_checkpoint_path():
    """Clears the checkpoint path textbox."""
    return ""

def restore_default_config():
    """Restores the default configuration values."""
    # Copy the default config to the app state
    APP_STATE["config"] = DEFAULT_CONFIG.copy()
    
    # Get all the default values to update UI
    output_component_keys = [
        'data_file', 'test_split_ratio', 'embedding_dim', 'context_window', 'hidden_dim', 'n_layers',
        'dt', 'mass_type', 'num_steps', 'batch_size', 'learning_rate_W',
        'eval_interval', 'gradient_clip_norm_W'
    ]
    
    default_values = [DEFAULT_CONFIG.get(key, "") for key in output_component_keys]
    
    # Add status message at the beginning
    status_msg = "Configuration restored to defaults."
    append_to_log(status_msg)
    
    return [status_msg] + default_values

# --- Checkpoint Loading ---
def load_checkpoint_ui(filepath):
    """Loads a checkpoint and sets up the trainer."""
    # Define the expected output components structure
    output_component_keys = [
        'data_file', 'embedding_dim', 'context_window', 'hidden_dim', 'n_layers',
        'dt', 'mass_type', 'num_steps', 'batch_size', 'learning_rate_W',
        'eval_interval', 'gradient_clip_norm_W'
    ]
    
    # Check if training is active first
    if APP_STATE.get("thread_active"):
        status_msg = "Please stop training before loading a checkpoint."
        config_vals = get_config_values(*output_component_keys)
        return [update_status(status_msg)] + config_vals
        
    if not filepath or not os.path.exists(filepath):
        status_msg = f"Error: Checkpoint file not found: {filepath}"
        # Return status + current config values
        config_vals = get_config_values(*output_component_keys)
        return [update_status(status_msg)] + config_vals

    try:
        # Stop existing training if running
        if APP_STATE.get("thread_active"):
             stop_training_ui()
             time.sleep(0.5) # Give thread a moment
        
        append_to_log(f"Loading checkpoint from {filepath}...")
        print(f"Attempting to load checkpoint: {filepath}")
        
        # First, try to load the checkpoint to extract the configuration
        import torch
        checkpoint = torch.load(filepath, map_location=device)
        
        # Check if the checkpoint contains configuration
        if 'config' in checkpoint:
            # Get the configuration from the checkpoint
            stored_config = checkpoint['config']
            append_to_log("Found configuration in checkpoint, restoring settings...")
            
            # Update the global config with values from the checkpoint
            for key in stored_config:
                if key in APP_STATE["config"]:
                    APP_STATE["config"][key] = stored_config[key]
            
            # Make sure critical architecture parameters are definitely set
            if 'embedding_dim' in stored_config:
                APP_STATE["config"]['embedding_dim'] = stored_config['embedding_dim']
            if 'hidden_dim' in stored_config:
                APP_STATE["config"]['hidden_dim'] = stored_config['hidden_dim']
            if 'n_layers' in stored_config:
                APP_STATE["config"]['n_layers'] = stored_config['n_layers']
            
            append_to_log(f"Restored configuration from checkpoint: hidden_dim={APP_STATE['config']['hidden_dim']}, n_layers={APP_STATE['config']['n_layers']}")
        else:
            append_to_log("Warning: Checkpoint does not contain configuration. Using current settings which may cause mismatch errors.")
        
        # Now create a new Trainer with the updated config
        APP_STATE["trainer"] = Trainer(APP_STATE["config"].copy())
        
        # Load the checkpoint into the trainer
        loaded_ok = APP_STATE["trainer"].load_checkpoint(filepath)

        if loaded_ok:
            # Ensure the APP_STATE config is fully updated from the trainer
            APP_STATE["config"] = APP_STATE["trainer"].config.copy()
            APP_STATE["data_loaded"] = True
            APP_STATE["model_ready"] = True
            
            status_msg = f"Checkpoint loaded successfully. Model ready at step: {APP_STATE['trainer'].current_step}"
            append_to_log(status_msg)
            print(status_msg)
            
            # Fetch the newly loaded config values to update UI
            config_vals = get_config_values(*output_component_keys)
            # Update the history state as well
            APP_STATE["latest_history"] = APP_STATE["trainer"].get_history()

            return [update_status(status_msg)] + config_vals
        else:
            # Keep model marked as not ready
            APP_STATE["model_ready"] = False
            status_msg = f"Failed to load checkpoint: {filepath}"
            append_to_log(status_msg)
            config_vals = get_config_values(*output_component_keys)
            return [update_status(status_msg)] + config_vals

    except Exception as e:
        print(f"Error processing checkpoint load: {e}")
        import traceback
        traceback.print_exc()
        APP_STATE["model_ready"] = False
        status_msg = f"Error loading checkpoint: {e}"
        append_to_log(status_msg)
        config_vals = get_config_values(*output_component_keys)
        return [update_status(status_msg)] + config_vals


# --------------------------------------------------------------------------
# Gradio UI Definition
# --------------------------------------------------------------------------
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: #007bff; background: #007bff; min-width: 100px;}
.gr-button-stop { border-color: #dc3545; background: #dc3545; }
.gr-input { min-width: 50px; } /* Adjust input width */
.gr-output { min-width: 50px; } /* Adjust output width */
footer { display: none !important; }
"""

def create_ui():
    demo = gr.Blocks(css=css, title="YAUM - Hamiltonian LLM Trainer")
    
    with demo:
        gr.Markdown("# YAUM - Hamiltonian LLM Trainer")
        gr.Markdown("Configure, train, and test LLMs using Hamiltonian-inspired dynamics.")

        status_textbox = gr.Textbox(label="Status", value=APP_STATE["latest_log"], interactive=False, lines=1)

        with gr.Tabs():
            # Setup & Config Tab
            with gr.TabItem("Setup & Config"):
                with gr.Row():
                    # Left Column - Data & Architecture
                    with gr.Column(scale=1):
                        gr.Markdown("### Data & Checkpoint")
                        with gr.Row():
                            data_file_input = gr.Textbox(label="Data File Path (.txt)", value=DEFAULT_CONFIG['data_file'], elem_classes="gr-input", scale=3)
                            data_file_button = gr.Button("Browse", elem_classes="gr-button", scale=1)
                        
                        with gr.Row():
                            load_checkpoint_input = gr.Textbox(label="Load Checkpoint Path (.pt)", placeholder="Optional: Path to resume/load", elem_classes="gr-input", scale=3)
                            checkpoint_file_button = gr.Button("Browse", elem_classes="gr-button", scale=1)

                        with gr.Row():
                            clear_checkpoint_button = gr.Button("Clear Checkpoint", elem_classes="gr-button", scale=1)
                            restore_defaults_button = gr.Button("Restore Defaults", elem_classes="gr-button", scale=1)

                        test_split_ratio_input = gr.Slider(label="Test Split Ratio", minimum=0.01, maximum=0.5, step=0.01, value=DEFAULT_CONFIG['test_split_ratio'])

                        gr.Markdown("### Model Architecture")
                        embedding_dim_input = gr.Number(label="Embedding Dim", value=DEFAULT_CONFIG['embedding_dim'], precision=0, elem_classes="gr-input")
                        context_window_input = gr.Number(label="Context Window", value=DEFAULT_CONFIG['context_window'], precision=0, elem_classes="gr-input")
                        hidden_dim_input = gr.Number(label="RNN Hidden Dim", value=DEFAULT_CONFIG['hidden_dim'], precision=0, elem_classes="gr-input")
                        n_layers_input = gr.Number(label="RNN Layers", value=DEFAULT_CONFIG['n_layers'], precision=0, elem_classes="gr-input")

                    # Right Column - Dynamics & Training
                    with gr.Column(scale=1):
                        gr.Markdown("### Hamiltonian Dynamics")
                        dt_input = gr.Number(label="dt (Time Step)", value=DEFAULT_CONFIG['dt'], minimum=0.0001, step=0.001, elem_classes="gr-input")
                        mass_type_input = gr.Dropdown(label="Mass Type", choices=["uniform", "frequency", "inverse_frequency"], value=DEFAULT_CONFIG['mass_type'], elem_classes="gr-input")

                        gr.Markdown("### Training Parameters")
                        num_steps_input = gr.Number(label="Num Steps", value=DEFAULT_CONFIG['num_steps'], precision=0, step=1000, elem_classes="gr-input")
                        batch_size_input = gr.Number(label="Batch Size", value=DEFAULT_CONFIG['batch_size'], precision=0, step=8, elem_classes="gr-input")
                        learning_rate_W_input = gr.Number(label="Adam LR (W)", value=DEFAULT_CONFIG['learning_rate_W'], minimum=1e-6, step=1e-4, elem_classes="gr-input")
                        eval_interval_input = gr.Number(label="Eval Interval (steps)", value=DEFAULT_CONFIG['eval_interval'], precision=0, step=100, elem_classes="gr-input")
                        gradient_clip_norm_W_input = gr.Number(label="Grad Clip Norm (W)", value=DEFAULT_CONFIG['gradient_clip_norm_W'], minimum=0.1, step=0.1, elem_classes="gr-input")

                # Button Row
                with gr.Row():
                    prepare_button = gr.Button("Prepare Data & Model / Apply Config", variant="primary", elem_classes="gr-button")
                    load_checkpoint_button = gr.Button("Load Checkpoint", elem_classes="gr-button")

                # --- Define outputs list for load_checkpoint_ui ---
                # Must match the order returned by the function AND the variable names used above
                config_output_components_for_load = [
                    status_textbox,
                    data_file_input, embedding_dim_input, context_window_input, hidden_dim_input, n_layers_input,
                    dt_input, mass_type_input, num_steps_input, batch_size_input, learning_rate_W_input,
                    eval_interval_input, gradient_clip_norm_W_input
                ]

                # --- Attach callbacks ---
                # Use gr.State to pass the whole config dict correctly
                config_state = gr.State(value=APP_STATE["config"])
                
                # Make the prepare button always use the latest APP_STATE["config"]
                prepare_button.click(
                    lambda: prepare_data_and_model(APP_STATE["config"]), 
                    inputs=None, 
                    outputs=[status_textbox]
                )
                
                load_checkpoint_button.click(
                    load_checkpoint_ui, 
                    inputs=[load_checkpoint_input], 
                    outputs=config_output_components_for_load
                )

                # Clear checkpoint button click handler
                clear_checkpoint_button.click(
                    clear_checkpoint_path,
                    inputs=[],
                    outputs=[load_checkpoint_input]
                )

                # Restore defaults button click handler
                # Define all components that should be updated with defaults
                restore_defaults_components = [
                    status_textbox,
                    data_file_input, test_split_ratio_input, embedding_dim_input, context_window_input, 
                    hidden_dim_input, n_layers_input, dt_input, mass_type_input, num_steps_input, 
                    batch_size_input, learning_rate_W_input, eval_interval_input, gradient_clip_norm_W_input
                ]

                restore_defaults_button.click(
                    restore_default_config,
                    inputs=[],
                    outputs=restore_defaults_components
                )
                
                # File browser button callbacks
                data_file_button.click(browse_data_file, inputs=[], outputs=[data_file_input])
                checkpoint_file_button.click(browse_checkpoint_file, inputs=[], outputs=[load_checkpoint_input])

                # Attach .change listeners AFTER components are defined to update APP_STATE config
                data_file_input.change(lambda x: update_config_value('data_file', x), inputs=data_file_input, outputs=data_file_input) 
                test_split_ratio_input.change(lambda x: update_config_value('test_split_ratio', x), inputs=test_split_ratio_input, outputs=test_split_ratio_input)
                embedding_dim_input.change(lambda x: update_config_value('embedding_dim', x), inputs=embedding_dim_input, outputs=embedding_dim_input)
                context_window_input.change(lambda x: update_config_value('context_window', x), inputs=context_window_input, outputs=context_window_input)
                hidden_dim_input.change(lambda x: update_config_value('hidden_dim', x), inputs=hidden_dim_input, outputs=hidden_dim_input)
                n_layers_input.change(lambda x: update_config_value('n_layers', x), inputs=n_layers_input, outputs=n_layers_input)
                dt_input.change(lambda x: update_config_value('dt', x), inputs=dt_input, outputs=dt_input)
                mass_type_input.change(lambda x: update_config_value('mass_type', x), inputs=mass_type_input, outputs=mass_type_input)
                num_steps_input.change(lambda x: update_config_value('num_steps', x), inputs=num_steps_input, outputs=num_steps_input)
                batch_size_input.change(lambda x: update_config_value('batch_size', x), inputs=batch_size_input, outputs=batch_size_input)
                learning_rate_W_input.change(lambda x: update_config_value('learning_rate_W', x), inputs=learning_rate_W_input, outputs=learning_rate_W_input)
                eval_interval_input.change(lambda x: update_config_value('eval_interval', x), inputs=eval_interval_input, outputs=eval_interval_input)
                gradient_clip_norm_W_input.change(lambda x: update_config_value('gradient_clip_norm_W', x), inputs=gradient_clip_norm_W_input, outputs=gradient_clip_norm_W_input)

            # Training Tab
            with gr.TabItem("Training"):
                with gr.Row():
                    start_button = gr.Button("Start Training", variant="primary", elem_classes="gr-button")
                    stop_button = gr.Button("Stop Training", elem_classes="gr-button gr-button-stop")
                
                # Progress as a text display instead of progress bar component
                progress_text = gr.Textbox(label="Progress", value="Model not prepared", interactive=False)
                training_log_output = gr.Textbox(label="Training Log", lines=10, interactive=False, max_lines=20, elem_classes="gr-output")
                plot_output = gr.Plot(label="Training Curves", elem_classes="gr-output")
                history_state = gr.State({}) # Holds history data returned by generator

                # --- Define outputs for the refresh UI function ---
                ui_refresh_outputs = [training_log_output, plot_output, history_state, progress_text]
                
                # --- Setup UI refresh loop ---
                demo.load(
                    refresh_ui,
                    inputs=None,
                    outputs=ui_refresh_outputs
                )

                # --- Button click handlers ---
                start_button.click(start_training_ui, inputs=None, outputs=[status_textbox])
                stop_button.click(stop_training_ui, inputs=None, outputs=[status_textbox])

            # Inference Tab
            with gr.TabItem("Inference"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(label="Start Prompt", lines=3, placeholder="Enter starting text...", elem_classes="gr-input")
                        generated_output = gr.Textbox(label="Generated Text", lines=10, interactive=False, elem_classes="gr-output")
                    with gr.Column(scale=1):
                        gen_length_input = gr.Slider(label="Generation Length", minimum=10, maximum=1000, step=10, value=300)
                        temperature_input = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.8)
                        generate_button = gr.Button("Generate Text", variant="primary", elem_classes="gr-button")
                # --- Button click handler ---
                generate_button.click(run_inference, inputs=[prompt_input, gen_length_input, temperature_input], outputs=[generated_output])

    return demo

# --- Run the App ---
if __name__ == "__main__":
    print(f"Launching Gradio UI... Default config: {APP_STATE['config']}")
    demo = create_ui()
    # Enable queue for smoother handling of multiple requests/updates
    demo.queue()
    demo.launch(share=False) # set share=True for public link if needed