import os
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime 
import time

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Print config to screen
    print("\n🔹 Loaded Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    #print("\n")

    return config

def setup_trainer(config, trial_mode=False, profiler=None):
    """Creates a Lightning Trainer with trial mode support and prints checkpoint path."""

    print("\n🔹 Setting up Trainer...")
    # Logger
    experiment_name = f"CHESSENGINE_{config['max_epochs']}epochs"
    logger = TensorBoardLogger("lightning_logs", name=experiment_name)

    # Checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Generate timestamp in "hh-mm_dd-mm-yy" format
    timestamp = datetime.now().strftime("%H-%M_%d-%m-%y")

    # Custom checkpoint filename
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss_total", 
        save_top_k=3,  # Keep top 3 models
        mode="min", 
        dirpath=checkpoint_dir, 
        filename=f"CHESSENGINE_{timestamp}_epoch={{epoch:02d}}_val_loss={{val_loss:.4f}}"
        
    )

    early_stop_callback = EarlyStopping(monitor="val/loss_total", patience=3, mode="min")

    # Print where checkpoints will be saved
    print(f"\n📂 Checkpoints will be saved in: {os.path.abspath(checkpoint_dir)}\n")
    print(f"📝 Checkpoint file format: {checkpoint_callback.filename}\n")

    trainer_args = {
        "max_epochs": 1 if trial_mode else config['max_epochs'],
        "accelerator": config.get("device", "mps" if torch.backends.mps.is_available() else "cpu"),
        "precision": config.get("precision", 32),
        "logger": logger,
        "callbacks": [checkpoint_callback, early_stop_callback, EpochTimerCallback()],
        "profiler": profiler,  # Add profiler if available
    }

    if trial_mode:
        trainer_args["limit_train_batches"] = 0.1  # Only use 10% batches
        trainer_args["limit_val_batches"] = 0.1

    return pl.Trainer(**trainer_args), checkpoint_dir

class EpochTimerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - pl_module.start_time
        if hasattr(trainer.logger.experiment, "add_scalar"):
            trainer.logger.experiment.add_scalar("epoch_time", epoch_duration, trainer.current_epoch)
        elif hasattr(trainer.logger, "log_metrics"):
            trainer.logger.log_metrics({"epoch_time": epoch_duration}, step=trainer.current_epoch)
        print(f"Epoch {trainer.current_epoch} time: {epoch_duration:.2f} sec")