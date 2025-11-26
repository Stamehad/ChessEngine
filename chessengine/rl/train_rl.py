import argparse
import yaml

import pytorch_lightning as pl
import torch

from chessengine.model.engine_pl import ChessLightningModule
from chessengine.pytorchchess.beam_search.self_play import SelfPlayEngine
from chessengine.pytorchchess.utils import constants_new as const
from chessengine.rl.initial_state_sampler import InitialStateSampler
from pytorchchess import TorchBoard
from rl.data_module import SelfPlayDataModule
from rl.dataset import SelfPlayDataset
from train_utils import setup_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play RL training")
    parser.add_argument("--config", type=str, default="chessengine/rl/rl_config.yml")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def apply_debug_settings(config):
    print("Debug mode enabled: reducing training parameters for quick testing.")
    rl_cfg = config.get("rl", {})
    rl_cfg["games"] = 1
    rl_cfg["expansion_factors"] = [3,2,1]
    rl_cfg["max_steps"] = 450
    rl_cfg["cycles"] = 1
    rl_cfg["device"] = "mps" if torch.backends.mps.is_available() else "cpu"
    config["rl"] = rl_cfg

    train_cfg = config.get("train", {})
    train_cfg["max_epochs"] = 1
    train_cfg["device"] = "mps"
    train_cfg["precision"] = 32
    config["train"] = train_cfg
    return config

def load_config(config_path: str, debug: bool = False):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if debug:
        config = apply_debug_settings(config)   

    # Print config to screen
    print("\n🔹 Loaded Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    return config


def main():
    args = parse_args()

    debug = args.debug
    config = load_config(args.config, debug)

    pl.seed_everything(config.get("seed", 42))

    if args.checkpoint:
        model = ChessLightningModule.load_from_checkpoint(args.checkpoint, config=config)
    else:
        model = ChessLightningModule(config)

    rl_cfg = config.get("rl", {})
    device = torch.device(rl_cfg.get("device", "cpu"))
    const.move_constants_to(device)
    num_cycles = rl_cfg.get("cycles", 1)
    epochs_per_cycle = rl_cfg.get("epochs_per_cycle", 1)

    train_cfg = config["train"].copy()
    train_cfg["max_epochs"] = epochs_per_cycle
    trainer, checkpoint_dir = setup_trainer(train_cfg)

    for cycle in range(num_cycles):
        print(f"\n=== Self-play cycle {cycle + 1}/{num_cycles} ===")
        batch = run_self_play(model, rl_cfg, device)
        dataset = SelfPlayDataset(batch, device=device)
        dm = SelfPlayDataModule(dataset, config["dataloader"])
        dm.setup()

        target_epoch = trainer.current_epoch + epochs_per_cycle
        trainer.fit_loop.max_epochs = target_epoch
        trainer.fit(model, datamodule=dm)
        print(f"Self-play cycle {cycle + 1} complete. Checkpoints: {checkpoint_dir}")


def run_self_play(model, rl_cfg: dict, device: torch.device):
    sampler = InitialStateSampler(rl_cfg.get("sampler", {}))
    expansion = torch.tensor(rl_cfg.get("expansion_factors", [3, 2, 1]), dtype=torch.long)
    games = rl_cfg.get("games", 1)
    required = games * (expansion.numel() + 1)

    boards = sampler.get_boards()
    while len(boards) < required:
        boards.extend(
            sampler.sample_initial_positions(n1=required, n2=0, include_start=False)
        )
    boards = boards[:required]

    tb = TorchBoard.from_board_list(boards, device=device)
    model.eval()
    engine = SelfPlayEngine(model, expansion, device=device)
    engine.initialize(tb)
    engine.run(max_iterations=rl_cfg.get("max_steps", 500))
    model.train()
    return engine.sample_buffer.batch


if __name__ == "__main__":
    main()
