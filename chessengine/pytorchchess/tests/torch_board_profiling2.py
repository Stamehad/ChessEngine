import argparse
import torch
from chessengine.rl.initial_state_sampler import InitialStateSampler
from pytorchchess import TorchBoard
from pytorchchess.state.legal_moves_new import LegalMovesNew
from torch.profiler import profile, ProfilerActivity

def build_boards(batch_size, sampler_cfg, device):
    sampler = InitialStateSampler(sampler_cfg)
    boards = sampler.get_boards()
    return TorchBoard.from_board_list(boards, device=device)

def compile_wrappers():
    def rank_wrapper(legal_moves, logits, ks, sample=False, temp=1.0, generator=None):
        return LegalMovesNew.rank_moves(legal_moves, logits, ks=ks, sample=sample, temp=temp, generator=generator)
    def push_wrapper(tb, moves, board_idx):
        return TorchBoard.push(tb, moves, board_idx)
    compiled_rank = torch.compile(rank_wrapper)
    compiled_push = torch.compile(push_wrapper)
    return compiled_rank, compiled_push

def single_cycle(tb, k=1, compiled_ops=None):
    legal_moves, _ = tb.get_moves()
    
    B = tb.batch_size
    logits = torch.randn(B, 64, 7, device=tb.device)
    ks = torch.full((B,), k, dtype=torch.long, device=tb.device)
    if compiled_ops is None:
        move_data = legal_moves.rank_moves(logits, ks=ks)
        push_fn = TorchBoard.push
    else:
        rank_fn, push_fn = compiled_ops
        move_data = rank_fn(legal_moves, logits, ks=ks)

    new_moves, board_idx, _ = move_data
    tb = push_fn(tb, new_moves, board_idx)

    return tb

def benchmark(batch_size, cycles, device, use_compile):
    sampler_cfg = {
        "prefetch": 2,
        "n_games": batch_size // 100,
        "positions_per_game": 1,
        "max_ply": 30,
        "database_dir": "data/shards300_small/",
    }
    tb = build_boards(batch_size, sampler_cfg, device)
    print(f"Initial TorchBoard batch size: {tb.batch_size}")
    compiled_ops = compile_wrappers() if use_compile else None
    # warmup
    for _ in range(5):
        tb = single_cycle(tb, compiled_ops=compiled_ops)
    torch.cuda.synchronize()

    times = []
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for i in range(cycles):
            k = 10 if i < 2 else 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tb = single_cycle(tb, k, compiled_ops)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
            prof.step()
    return times, prof

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile TorchBoard fused moves + push")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile wrappers for rank_moves/push")
    args = parser.parse_args()

    print("Running benchmark with settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Compile: {args.compile}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    times, prof = benchmark(args.batch_size, args.cycles, device, args.compile)
    for i, t in enumerate(times):
        print(f"Cycle {i+1}/{args.cycles}: {t:.2f} ms")
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
