import pytorch_lightning as pl
import argparse

#from chessengine.dataloader import get_dataloaders
from chessengine.datamodule import ChessDataModule
from chessengine.engine_pl import ChessLightningModule
from train_utils import load_config, setup_trainer

# setup arg parser to get checkpoint path
def parse_args():
    parser = argparse.ArgumentParser(description="CHESS ENGINE Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training")
    return parser.parse_args()

def main():
    # Load args
    args = parse_args()
    #config = load_config(args.config)

    # Load YAML config
    config_path = "engine_config.yaml"
    config = load_config(config_path)

    # Set random seed for reproducibility
    pl.seed_everything(config.get("seed", 42))

    # Get DataLoaders
    data_paths = ["../chess_engine/data/shards300_small/positions0.pt", "../chess_engine/data/shards300_small/positions4.pt"]
    data_paths = ["../chess_engine/data/test_positions.pt"]
    #train_loader, val_loader = get_dataloaders(data_paths, config)

    dm = ChessDataModule(config, data_paths)
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

    # if args.checkpoint:
    #     print(f"🔄 Resuming from checkpoint: {args.checkpoint}")
    #     trainer.test(model, val_loader, ckpt_path=args.checkpoint)
    # else:
    #     trainer.test(model, val_loader)  # Initial test with randomly initialized model

    # Train (from scratch or from checkpoint)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.checkpoint)

    # Final test after training
    trainer.test(model, val_loader)

    # Print final checkpoint path
    print(f"\n✅ Training Complete! Best checkpoint saved in: {checkpoint_dir}\n")

if __name__ == "__main__":
    main()