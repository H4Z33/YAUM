import gradio as gr
import sys
import os
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjust path to import from parent directory modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hamiltonian_llm_trainer.data.handling import load_and_split_data, calculate_mass_vector, save_vocab, load_vocab
from hamiltonian_llm_trainer.core.trainer import Trainer
from hamiltonian_llm_trainer.core.utils import device # Import device info

# --- Global State (Use Gradio's State or manage carefully) ---
# Store the trainer instance and configuration globally or via gr.State
# Using global here for simplicity, but gr.State is better for complex apps
APP_STATE = {
    "trainer": None,
    "config": {},
    "data_loaded": False,
    "model_ready": False,
    "training_thread": None,
    "latest_log": "Welcome! Configure settings and prepare data.",
    "latest_history": None,
}

# Default Config
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
APP_STATE["config"] = DEFAULT_CONFIG.copy()

# --- UI Helper Functions ---

def update_config_value(key, value):
    """Updates a single value in the global config state."""
    if value is not None: # Ensure we don't store None from empty inputs accidentally
         APP_STATE["config"][key] = value
    # Return the possibly updated value to the UI component
    return APP_STATE["config"].get(key, DEFAULT_CONFIG.get(key))


def get_config_values(*keys):
     """Gets multiple config values for updating UI components."""
     return [APP_STATE["config"].get(k, DEFAULT_CONFIG.get(k)) for k in keys]

def update_status(text):
    APP_STATE["latest_log"] = text
    return text

# --- Backend Logic Functions Called by UI ---

def prepare_data_and_model(config_dict):
    """Loads data, calculates mass, sets up Trainer instance."""
    filepath = config_dict.get('data_file')
    if not filepath or not os.path.exists(filepath):
        return update_status(f"Error: Data file not found or not specified: {filepath}")

    APP_STATE["config"].update(config_dict) # Update state with current UI values
    config = APP_STATE["config"]

    try:
        train_data, test_data, char_to_idx, idx_to_char, vocab_size = load_and_split_data(
            filepath, config['test_split_ratio']
        )
        if train_data is None: # Check if loading failed
             return update_status("Error: Failed to load or split data.")

        mass_vector = calculate_mass_vector(
            train_data, vocab_size, config['mass_type'], device
        )

        # Create or reuse Trainer instance
        APP_STATE["trainer"] = Trainer(config) # Pass the updated config
        APP_STATE["trainer"].setup(
            train_data, test_data, char_to_idx, idx_to_char, vocab_size, mass_vector
        )

        APP_STATE["data_loaded"] = True
        APP_STATE["model_ready"] = True
        return update_status(f"Data loaded & Trainer ready. Vocab size: {vocab_size}. Device: {device}.")

    except Exception as e:
        APP_STATE["data_loaded"] = False
        APP_STATE["model_ready"] = False
        # Log traceback here?
        print(f"Error during data/model preparation: {e}")
        return update_status(f"Error preparing data/model: {e}")


def run_training_background():
    """Target function for the training thread."""
    if not APP_STATE.get("model_ready") or not APP_STATE.get("trainer"):
        print("Training start requested, but trainer not ready.")
        # Need a way to signal this back to UI thread?
        return

    trainer = APP_STATE["trainer"]
    # The train method is now a generator yielding progress
    # We don't directly interact with its output here, the UI callback will
    try:
         # Consume the generator to run training
         # The actual updates to UI will happen via the polling mechanism below
         # or direct Gradio updates if the generator is handled correctly in the UI callback
         for _ in trainer.train():
             # Check stop flag frequently if needed within the generator itself
             if trainer._stop_training_flag:
                 break
    except Exception as e:
         print(f"Exception in training thread: {e}")
         # Update status somehow? Maybe set a flag checked by UI poll
         APP_STATE["latest_log"] = f"Training Error: {e}"


def start_training_ui():
    """Starts the training process in a background thread."""
    if not APP_STATE.get("model_ready"):
        return update_status("Error: Prepare Data/Model first.")
    if APP_STATE.get("training_thread") and APP_STATE["training_thread"].is_alive():
        return update_status("Training is already running.")

    update_status("Starting training...")
    trainer = APP_STATE["trainer"]
    trainer._stop_training_flag = False # Ensure flag is reset

    # Run trainer.train in a background thread
    thread = threading.Thread(target=run_training_background, daemon=True)
    APP_STATE["training_thread"] = thread
    thread.start()

    # UI updates will happen via the `update_training_progress` function below
    # Return initial status message
    return update_status("Training started in background...")


def stop_training_ui():
    """Signals the training thread to stop."""
    trainer = APP_STATE.get("trainer")
    thread = APP_STATE.get("training_thread")

    if trainer and thread and thread.is_alive():
        update_status("Requesting training stop...")
        trainer.stop_training()
        # Wait briefly for thread to potentially finish current step? Optional.
        # thread.join(timeout=5) # Careful with join on daemon threads
        return update_status("Stop requested. Training will halt after current step/check.")
    elif trainer and not (thread and thread.is_alive()):
         return update_status("Training is not currently running.")
    else:
        return update_status("Trainer not initialized.")


def update_training_progress():
    """Called periodically by Gradio UI to get training updates."""
    trainer = APP_STATE.get("trainer")
    thread = APP_STATE.get("training_thread")
    is_running = thread and thread.is_alive()

    if not trainer:
        # Return default empty/initial states for plots and logs
         empty_history = {'steps': [], 'train_l1': [], 'train_l2': [], 'test_l': [], 'grad_W': [], 'force_E': [], 'P_norm': []}
         return "Trainer not ready.", None, None, empty_history # Log, Progress, Plot, History

    current_step = trainer.current_step
    max_steps = trainer.config.get('num_steps', 1)
    progress = gr.Progress(progress=current_step / max_steps if max_steps > 0 else 0) if is_running else None

    log = APP_STATE.get("latest_log", "No status yet.") # Get latest log msg

    # Generate plots from history
    history = trainer.get_history()
    fig = plot_training_history(history)

    # Return updates for Gradio components
    # Need to match the outputs defined in the Gradio interface
    return log, progress, fig, history # Match outputs of training_progress_box update

def plot_training_history(history):
    """Generates Matplotlib plots from training history."""
    if not history or not history['steps']:
        # Return an empty figure or placeholder
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, 'No training data yet', ha='center', va='center')
        ax.axis('off')
        plt.close(fig) # Prevent showing inline if not needed
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    steps = history['steps']

    # Loss Plot
    axes[0, 0].plot(steps, history['train_l1'], label='L1 (Pre-E)', alpha=0.7)
    axes[0, 0].plot(steps, history['train_l2'], label='L2->W (Post-E)', alpha=0.7)
    axes[0, 0].plot(steps, history['test_l'], label='Test Loss', linewidth=2)
    axes[0, 0].set_title('Losses')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    min_loss_plot = min(history['test_l']) * 0.95 if history['test_l'] else 0
    max_loss_plot = max(history['test_l']) * 1.05 if history['test_l'] else 5.0
    if max_loss_plot > min_loss_plot * 10: max_loss_plot = min_loss_plot * 10 # Cap ylim range
    axes[0, 0].set_ylim(bottom=min_loss_plot, top=max_loss_plot)


    # Grad W Plot
    axes[0, 1].plot(steps, history['grad_W'], label='||∇W|| (pre-clip)')
    axes[0, 1].set_title('W Gradient Norm')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Norm')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    if history['grad_W']: axes[0, 1].set_ylim(bottom=0, top=max(history['grad_W']) * 1.1)


    # Force E Plot
    axes[1, 0].plot(steps, history['force_E'], label='||F_E|| (approx)')
    axes[1, 0].set_title('Effective Force Norm on E')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Norm')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    if history['force_E']: axes[1, 0].set_ylim(bottom=0, top=max(history['force_E']) * 1.1)


    # Momentum P Plot
    axes[1, 1].plot(steps, history['P_norm'], label='||P||')
    axes[1, 1].set_title('Momentum Matrix Norm')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Norm')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    if history['P_norm']: axes[1, 1].set_ylim(bottom=0, top=max(history['P_norm']) * 1.1)


    plt.tight_layout()
    plt.close(fig) # Prevent showing inline
    return fig


def run_inference(prompt, length, temperature):
    """Calls the trainer's generate method."""
    trainer = APP_STATE.get("trainer")
    if not trainer or not APP_STATE.get("model_ready"):
         # Try loading latest checkpoint if available? Or require manual load?
        return "Error: Model not ready. Train or load a checkpoint first."

    try:
        # Ensure model is on correct device (loading might change it?)
        trainer.model.to(device)
        trainer.E = trainer.E.to(device)

        length = int(length)
        temperature = float(temperature)
        result = trainer.generate(prompt, length, temperature)
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        return f"Error during inference: {e}"

def load_checkpoint_ui(filepath):
    """Loads a checkpoint and sets up the trainer."""
    if not filepath or not os.path.exists(filepath):
        return update_status(f"Error: Checkpoint file not found: {filepath}")

    try:
        # Create a new trainer instance based on the loaded config later
        # Or, have one global trainer and load into it? Let's reload into global.
        if APP_STATE["trainer"] is None:
             # Need a dummy config to initialize trainer before loading? Or handle in load_checkpoint
             # Let's assume load_checkpoint creates/replaces if needed.
             APP_STATE["trainer"] = Trainer({}) # Dummy init

        loaded_ok = APP_STATE["trainer"].load_checkpoint(filepath)

        if loaded_ok:
            APP_STATE["config"] = APP_STATE["trainer"].config # Update UI config state
            APP_STATE["data_loaded"] = True # Assume checkpoint implies data setup
            APP_STATE["model_ready"] = True
            # Update UI elements with loaded config values
            # This requires returning multiple values from this function or separate updates
            status_msg = f"Checkpoint loaded from {filepath}. Step: {APP_STATE['trainer'].current_step}"
            print(status_msg)
            # Need to return config values to update UI fields
            config_vals = get_config_values(
                 'data_file', 'embedding_dim', 'context_window', 'hidden_dim', 'n_layers',
                 'dt', 'mass_type', 'num_steps', 'batch_size', 'learning_rate_W',
                 'eval_interval', 'gradient_clip_norm_W'
            )
            return [status_msg] + config_vals
        else:
            APP_STATE["model_ready"] = False
            return [update_status(f"Failed to load checkpoint: {filepath}")] + get_config_values(
                 'data_file', 'embedding_dim', 'context_window', 'hidden_dim', 'n_layers',
                 'dt', 'mass_type', 'num_steps', 'batch_size', 'learning_rate_W',
                 'eval_interval', 'gradient_clip_norm_W'
            ) # Return current config if load fails

    except Exception as e:
        print(f"Error processing checkpoint load: {e}")
        APP_STATE["model_ready"] = False
        return [update_status(f"Error loading checkpoint: {e}")] + get_config_values(
             'data_file', 'embedding_dim', 'context_window', 'hidden_dim', 'n_layers',
             'dt', 'mass_type', 'num_steps', 'batch_size', 'learning_rate_W',
             'eval_interval', 'gradient_clip_norm_W'
        )


# --- Gradio UI Definition ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: #007bff; background: #007bff; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="Hamiltonian LLM Trainer") as demo:
    gr.Markdown("# Hamiltonian LLM Trainer")
    gr.Markdown("Configure, train, and test LLMs using Hamiltonian-inspired dynamics.")

    status_textbox = gr.Textbox(label="Status", value=APP_STATE["latest_log"], interactive=False, lines=1)

    with gr.Tabs():
        with gr.TabItem("Setup & Config"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Data")
                    data_file_input = gr.Textbox(label="Data File Path (.txt)", value=DEFAULT_CONFIG['data_file'])
                    load_checkpoint_input = gr.Textbox(label="Load Checkpoint Path (.pt)", placeholder="Optional: Path to checkpoint file")
                    test_split_ratio_input = gr.Slider(label="Test Split Ratio", minimum=0.01, maximum=0.5, step=0.01, value=DEFAULT_CONFIG['test_split_ratio'])

                    gr.Markdown("### Model Architecture")
                    embedding_dim_input = gr.Number(label="Embedding Dim", value=DEFAULT_CONFIG['embedding_dim'], precision=0)
                    context_window_input = gr.Number(label="Context Window", value=DEFAULT_CONFIG['context_window'], precision=0)
                    hidden_dim_input = gr.Number(label="RNN Hidden Dim", value=DEFAULT_CONFIG['hidden_dim'], precision=0)
                    n_layers_input = gr.Number(label="RNN Layers", value=DEFAULT_CONFIG['n_layers'], precision=0)

                with gr.Column(scale=1):
                    gr.Markdown("### Hamiltonian Dynamics")
                    dt_input = gr.Number(label="dt (Time Step)", value=DEFAULT_CONFIG['dt'], minimum=0.0001)
                    mass_type_input = gr.Dropdown(label="Mass Type", choices=["uniform", "frequency", "inverse_frequency"], value=DEFAULT_CONFIG['mass_type'])

                    gr.Markdown("### Training Parameters")
                    num_steps_input = gr.Number(label="Num Steps", value=DEFAULT_CONFIG['num_steps'], precision=0)
                    batch_size_input = gr.Number(label="Batch Size", value=DEFAULT_CONFIG['batch_size'], precision=0)
                    learning_rate_W_input = gr.Number(label="Adam LR (for RNN Weights)", value=DEFAULT_CONFIG['learning_rate_W'])
                    eval_interval_input = gr.Number(label="Eval Interval (steps)", value=DEFAULT_CONFIG['eval_interval'], precision=0)
                    gradient_clip_norm_W_input = gr.Number(label="Gradient Clip Norm (W)", value=DEFAULT_CONFIG['gradient_clip_norm_W'])

            with gr.Row():
                prepare_button = gr.Button("Prepare Data & Model / Apply Config", variant="primary")
                load_checkpoint_button = gr.Button("Load Checkpoint")

            # Link UI components to update the config state (using helper function)
            # This is verbose, could be done more elegantly with loops/classes
            # Important: Use _js parameter to run function on change without submitting full form
            data_file_input.change(lambda x: update_config_value('data_file', x), inputs=data_file_input, outputs=data_file_input)
            test_split_ratio_input.change(lambda x: update_config_value('test_split_ratio', x), inputs=test_split_ratio_input, outputs=test_split_ratio_input)
            embedding_dim_input.change(lambda x: update_config_value('embedding_dim', int(x)), inputs=embedding_dim_input, outputs=embedding_dim_input)
            context_window_input.change(lambda x: update_config_value('context_window', int(x)), inputs=context_window_input, outputs=context_window_input)
            hidden_dim_input.change(lambda x: update_config_value('hidden_dim', int(x)), inputs=hidden_dim_input, outputs=hidden_dim_input)
            n_layers_input.change(lambda x: update_config_value('n_layers', int(x)), inputs=n_layers_input, outputs=n_layers_input)
            dt_input.change(lambda x: update_config_value('dt', x), inputs=dt_input, outputs=dt_input)
            mass_type_input.change(lambda x: update_config_value('mass_type', x), inputs=mass_type_input, outputs=mass_type_input)
            num_steps_input.change(lambda x: update_config_value('num_steps', int(x)), inputs=num_steps_input, outputs=num_steps_input)
            batch_size_input.change(lambda x: update_config_value('batch_size', int(x)), inputs=batch_size_input, outputs=batch_size_input)
            learning_rate_W_input.change(lambda x: update_config_value('learning_rate_W', x), inputs=learning_rate_W_input, outputs=learning_rate_W_input)
            eval_interval_input.change(lambda x: update_config_value('eval_interval', int(x)), inputs=eval_interval_input, outputs=eval_interval_input)
            gradient_clip_norm_W_input.change(lambda x: update_config_value('gradient_clip_norm_W', x), inputs=gradient_clip_norm_W_input, outputs=gradient_clip_norm_W_input)
            # Checkpoint path is handled separately by load button

            # Define outputs for the load checkpoint function to update UI fields
            config_output_components = [
                 status_textbox, # First output is status
                 data_file_input, embedding_dim_input, context_window_input, hidden_dim_input, n_layers_input,
                 dt_input, mass_type_input, num_steps_input, batch_size_input, learning_rate_W_input,
                 eval_interval_input, gradient_clip_norm_W_input
            ]

            prepare_button.click(prepare_data_and_model, inputs=[gr.State(APP_STATE["config"])], outputs=[status_textbox])
            load_checkpoint_button.click(load_checkpoint_ui, inputs=[load_checkpoint_input], outputs=config_output_components)


        with gr.TabItem("Training"):
            with gr.Row():
                start_button = gr.Button("Start Training", variant="primary")
                stop_button = gr.Button("Stop Training")
            training_progress = gr.Progress(label="Training Progress")
            training_log_output = gr.Textbox(label="Training Log", lines=10, interactive=False, max_lines=20)
            plot_output = gr.Plot(label="Training Curves")

            # Hidden state to store history data for plotting
            history_state = gr.State({})

            # Use `every=` parameter to poll for updates periodically
            # The function `update_training_progress` will provide data for log, progress, plot
            # Match outputs: log, progress, plot, history_state (used internally by plot func)
            train_update_outputs = [training_log_output, training_progress, plot_output, history_state]
            demo.load(update_training_progress, inputs=None, outputs=train_update_outputs, every=5) # Poll every 5 seconds

            start_button.click(start_training_ui, inputs=None, outputs=[status_textbox])
            stop_button.click(stop_training_ui, inputs=None, outputs=[status_textbox])


        with gr.TabItem("Inference"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(label="Start Prompt", lines=3, placeholder="Enter starting text...")
                    generated_output = gr.Textbox(label="Generated Text", lines=10, interactive=False)
                with gr.Column(scale=1):
                    gen_length_input = gr.Slider(label="Generation Length", minimum=10, maximum=1000, step=10, value=300)
                    temperature_input = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.8)
                    generate_button = gr.Button("Generate Text", variant="primary")

            generate_button.click(run_inference, inputs=[prompt_input, gen_length_input, temperature_input], outputs=[generated_output])


# --- Add __main__ guard for running the app ---
if __name__ == "__main__":
    print("Launching Gradio UI...")
    # Share=True creates a public link (useful for Colab/remote access, remove for local only)
    demo.launch(share=False)