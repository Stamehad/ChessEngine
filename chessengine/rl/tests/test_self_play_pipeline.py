import os
import torch
import yaml
from dotenv import load_dotenv

from chessengine.rl.initial_state_sampler import InitialStateSampler
from chessengine.pytorchchess.beam_search.self_play import SelfPlayEngine
from pytorchchess import TorchBoard


def main(use_real_model: bool = False):
    config = {
        "prefetch": 2,
        "n_games": 2,
        "positions_per_game": 1,
        "max_ply": 30,
        "database_dir": "data/shards300_small/",
    }

    sampler = InitialStateSampler(config)
    expansion_factors = torch.tensor([3, 2, 1], dtype=torch.long)
    D = expansion_factors.numel()
    G = 1  # keep the test lightweight
    required_boards = G * (D + 1)

    boards = sampler.get_boards()
    if len(boards) < required_boards:
        extra = sampler.sample_initial_positions(n1=required_boards, n2=0, include_start=False)
        boards.extend(extra)

    print(f"Sampled {len(boards)} initial positions from dataset")

    device = torch.device("cpu")
    tb = TorchBoard.from_board_list(boards[:required_boards], device=device)
    model = load_real_model(device) if use_real_model else DummyModel()
    engine = SelfPlayEngine(model, expansion_factors, device=device)
    engine.initialize(tb)

    engine.run(max_iterations=1000)

    return engine.sample_buffer.batch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(64 * 21, 128)
        self.policy_head = torch.nn.Linear(128, 64 * 7)
        self.loss_module = DummyLossModule()

    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)
        hidden = torch.relu(self.encoder(flat))
        move_pred = self.policy_head(hidden).view(B, 64, 7)
        value_logits = torch.randn(B, 128)
        return value_logits, move_pred


class DummyLossModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prob_eval_loss = DummyEvalHead()


class DummyEvalHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(128, 3)
        self.dummy_linear = torch.nn.Linear(128, 3)

    def forward(self, x):
        return torch.randn(x.size(0), 3, device=x.device)


def load_real_model(device: torch.device):
    from model.engine_pl import ChessLightningModule

    load_dotenv()
    checkpoint = os.getenv("BASE_MODEL")
    if checkpoint is None:
        raise RuntimeError("BASE_MODEL environment variable is not set")

    with open("engine_config.yaml", "r") as cfg_file:
        config = yaml.safe_load(cfg_file)

    model = ChessLightningModule.load_from_checkpoint(checkpoint, config=config)
    model.eval().to(device)
    return model


if __name__ == "__main__":
    batch = main(use_real_model=True)
    print("Generated batch with", batch.features.shape[0], "samples")
