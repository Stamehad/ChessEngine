import os
import torch
import yaml
from dotenv import load_dotenv

from chessengine.rl.initial_state_sampler import InitialStateSampler
from chessengine.pytorchchess.beam_search.self_play import SelfPlayEngine, SelfPlayProfiler
from pytorchchess import TorchBoard
from torch.profiler import profile, ProfilerActivity


def main(
    device: torch.device,
    expansion_factors,
    games: int,
    sampler_cfg,
    use_real_model: bool = False,
    max_steps: int = 100,
    profile_steps: int = 50,
):
    sampler = InitialStateSampler(sampler_cfg)
    D = expansion_factors.numel()
    required_boards = games * (D + 1)

    boards = sampler.get_boards()
    while len(boards) < required_boards:
        boards.extend(
            sampler.sample_initial_positions(n1=required_boards, include_start=False)
        )
    boards = boards[:required_boards]

    print(f"Sampled {len(boards)} initial positions from dataset")

    tb = TorchBoard.from_board_list(boards, device=device)
    model = load_real_model(device) if use_real_model else DummyModel().to(device)
    profiler = SelfPlayProfiler(enabled=True)
    engine = SelfPlayEngine(model, expansion_factors, device=device, profiler=profiler)
    engine.initialize(tb)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=False,
        profile_memory=True,
    ) as prof:
        for step in range(max_steps):
            engine.step_once()
            if step < profile_steps:
                prof.step()

    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    print(profiler.summary())
    return engine.sample_buffer.batch, prof


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
        config["rl"] = True

    model = ChessLightningModule.load_from_checkpoint(checkpoint, config=config)
    model.eval().to(device)
    return model


if __name__ == "__main__":
    expansion = torch.tensor([3, 2, 1], dtype=torch.long)
    G = 2
    n_games = G * (expansion.numel() + 1)
    sampler_cfg = {
        "prefetch": 2,
        "n_games": n_games,
        "positions_per_game": 1,
        "max_ply": 30,
        "database_dir": "data/shards300_small/",
    }
    batch, prof = main(
        device=torch.device("cpu"),
        expansion_factors=expansion,
        games=G,
        sampler_cfg=sampler_cfg,
        use_real_model=False,
        max_steps=10,
        profile_steps=50,
    )
    print("Generated batch with", batch.features.shape[0], "samples")
