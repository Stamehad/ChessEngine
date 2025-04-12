import pytorch_lightning as pl
import argparse
import os

#from chessengine.dataloader import get_dataloaders
from chessengine.model.datamodule import ChessDataModule
from chessengine.model.engine_pl import ChessLightningModule
from train_utils import load_config, setup_trainer

from dotenv import load_dotenv
load_dotenv(".env")

training_paths = os.getenv("TRAINING_DATASET").split(", ")
test_path = os.getenv("TEST_DATASET")

# setup arg parser to get checkpoint path
def parse_args():
    parser = argparse.ArgumentParser(description="CHESS ENGINE Training")
    parser.add_argument("--config", type=str, default="engine_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    return parser.parse_args()

def main():
    # Load args
    args = parse_args()

    # Load YAML config
    config = load_config(args.config)

    # Set random seed for reproducibility
    pl.seed_everything(config.get("seed", 42))

    # Load data paths
    if args.test:
        data_paths = [test_path]
        print(f"Using test path: {data_paths}")
    else:
        data_paths = training_paths
        print(f"Using training paths.")

    dm = ChessDataModule(config, data_paths, OVERFIT=False)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    config["train"]["T_max"] = len(train_loader) * config["train"]["max_epochs"]
    print(f"Total training steps: {config['train']['T_max']}")
    
    # Initialize Model
    print("Initializing model...")
    model = ChessLightningModule(config)

    # Setup Trainer
    trainer, checkpoint_dir = setup_trainer(config["train"]) #, profiler="simple")

    if args.checkpoint:
        print(f"🔄 Resuming from checkpoint: {args.checkpoint}")
        trainer.test(model, val_loader, ckpt_path=args.checkpoint)
    else:
        trainer.test(model, val_loader)  # Initial test with randomly initialized model

    # Train (from scratch or from checkpoint)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.checkpoint)

    # Final test after training
    trainer.test(model, val_loader)

    # Print final checkpoint path
    print(f"\n✅ Training Complete! Best checkpoint saved in: {checkpoint_dir}\n")

if __name__ == "__main__":
    main()