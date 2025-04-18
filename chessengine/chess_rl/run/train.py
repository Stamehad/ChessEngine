# chess_rl/run/train.py
import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from chess_rl.training.lightning_module import ChessRLModule
from chess_rl.training.datamodule import ReplayDataModule

def load_config(config_path='chess_rl/rl_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_training_cycle(config):
    print("\n--- Starting Training Cycle ---")

    # 1. Initialize DataModule
    datamodule = ReplayDataModule(config)
    # Important: Call setup to find data files and potentially load subset
    datamodule.setup()

    # Check if data exists before starting trainer
    if datamodule.dataset is None or len(datamodule.dataset) == 0:
        print("Skipping training cycle: No data found in the replay buffer or loaded subset is empty.")
        return # Exit if no data

    # 2. Initialize LightningModule (will load latest weights internally)
    model = ChessRLModule(config)

    # 3. Configure Trainer
    # Optional: Checkpoint callback to save full Lightning checkpoints (for resuming training)
    # pl_checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(config['model']['checkpoint_dir'], "pl_checkpoints"),
    #     filename='chess-rl-{epoch:02d}-{val_loss:.2f}', # Example, if validation is added
    #     save_top_k=1,
    #     monitor='train_loss', # Monitor train loss if no validation
    #     mode='min'
    # )
    # callbacks = [pl_checkpoint_callback]
    callbacks = [] # Add callbacks as needed

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        callbacks=callbacks,
        log_every_n_steps=10, # Adjust logging frequency
        # precision="16-mixed" # Optional: if using mixed precision
        # Add other Trainer args as needed (profiler, logger, etc.)
    )

    # 4. Train the model
    print(f"Starting training on {len(datamodule.dataset)} samples...")
    trainer.fit(model, datamodule)

    # Note: Saving the core model weights now happens *within* the LightningModule's on_train_epoch_end

    print("--- Training Cycle Completed ---")

if __name__ == "__main__":
    config = load_config()
    run_training_cycle(config)