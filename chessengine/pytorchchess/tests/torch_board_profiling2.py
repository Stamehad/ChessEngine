import torch
from chessengine.rl.initial_state_sampler import InitialStateSampler
from pytorchchess import TorchBoard
from torch.profiler import profile, ProfilerActivity

def build_boards(batch_size, sampler_cfg, device):
    sampler = InitialStateSampler(sampler_cfg)
    boards = sampler.get_boards()
    return TorchBoard.from_board_list(boards, device=device)

def single_cycle(tb, k=1):
    legal_moves, _ = tb.get_legal_moves_fused(return_features=True)
    lm_tensor = legal_moves.tensor  # (B, 64, Lmax)
    
    B, _ , _ = lm_tensor.size()
    logits = torch.randn(B, 64, 7, device=tb.device)
    ks = torch.full((B,), k, dtype=torch.long, device=tb.device)
    move_data = legal_moves.rank_moves(logits, ks=ks)
        
    new_moves, board_idx, _ = move_data
    tb = tb.push(new_moves, board_idx)

    return tb

def benchmark(batch_size, cycles, device):
    sampler_cfg = {
        "prefetch": 2,
        "n_games": batch_size // 100,
        "positions_per_game": 1,
        "max_ply": 30,
        "database_dir": "data/shards300_small/",
    }
    tb = build_boards(batch_size, sampler_cfg, device)
    # warmup
    for _ in range(5):
        tb = single_cycle(tb)
    torch.cuda.synchronize()

    times = []
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for i in range(cycles):
            k = 10 if i < 2 else 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tb = single_cycle(tb, k)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
            prof.step()
    return times, prof

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 100 #8192
    cycles = 50
    times, prof = benchmark(batch_size, cycles, device)
    for i, t in enumerate(times):
        print(f"Cycle {i+1}/{cycles}: {t:.2f} ms")
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
