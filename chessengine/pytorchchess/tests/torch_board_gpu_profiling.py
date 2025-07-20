import torch
import chess
from pytorchchess import TorchBoard
import pytorchchess.utils.constants as const
import pytorchchess.torch_board.pseudo_move_gen_new as pmg_new


if __name__ == "__main__":
    import argparse
    from torch.profiler import profile, record_function, ProfilerActivity

    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--compile", action='store_true', help="Compile model with torch.compile")
    parser.add_argument("--export-trace", type=str, default=None, help="Export Chrome trace to file")
    args = parser.parse_args()

    torch.manual_seed(123456789)
    B = args.B
    DEVICE = torch.device(args.device)
    STEPS = args.steps
    VERBOSE = args.verbose
    COMPILE = args.compile
    TRACE_FILE = args.export_trace

    print(f"Running profiling with B={B}, STEPS={STEPS}, DEVICE={DEVICE}, COMPILE={COMPILE}")

    const.move_constants_to(DEVICE)

    # Monkey-patch PseudoMoveGeneratorNew.get_moves
    original_get_moves = pmg_new.PseudoMoveGeneratorNew.get_moves

    def profiled_get_moves(self, *args, **kwargs):
        with record_function("PseudoMoveGeneratorNew.get_moves"):
            return original_get_moves(self, *args, **kwargs)

    pmg_new.PseudoMoveGeneratorNew.get_moves = profiled_get_moves

    # Create TorchBoard with many positions
    boards = [chess.Board() for _ in range(B)]
    boards = TorchBoard.from_board_list(boards, device=DEVICE)

    # Warm-up GPU
    for _ in range(10):
        _, _, lm = boards.get_legal_moves_new()
        is_terminal, result = boards.is_game_over()
        if is_terminal.any():
            boards = boards[~is_terminal]
            lm = lm.select(~is_terminal)
        logits = torch.zeros_like(lm.encoded, dtype=torch.float32)
        ks = torch.ones(lm.encoded.shape[0], dtype=torch.long)
        moves, b_idx, _, _ = lm.sample_k(logits, ks)
        boards = boards.push(moves, b_idx)
    torch.cuda.synchronize()

    # Start profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(STEPS):
            _, _, lm = boards.get_legal_moves_new()  # premoves, in_check, legal_moves
            is_terminal, result = boards.is_game_over()
            if is_terminal.any():
                boards = boards[~is_terminal]
                lm = lm.select(~is_terminal)

            logits = torch.zeros_like(lm.encoded, dtype=torch.float32)  # Simulate logits
            ks = torch.ones(lm.encoded.shape[0], dtype=torch.long)  # Sample one move per board
            moves, b_idx, _, _ = lm.sample_k(logits, ks)
            boards = boards.push(moves, b_idx)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    if TRACE_FILE:
        prof.export_chrome_trace(TRACE_FILE)
        print(f"Chrome trace exported to {TRACE_FILE}")