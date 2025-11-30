import torch
import chess
from pytorchchess import TorchBoard
from pytorchchess.utils.profiler import profiler, auto_profile_class
import pytorchchess.utils.constants as const


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)  # CHANGED: steps instead of cycles
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--profile", action='store_true', help="Enable profiling output")
    parser.add_argument("--compile", action='store_true', help="Compile model with torch.compile")
    args = parser.parse_args()

    seed = 4
    torch.manual_seed(seed)
    B = args.B
    DEVICE = torch.device(args.device)
    STEPS = args.steps
    VERBOSE = args.verbose
    PROPILE = args.profile
    COMPILE = args.compile

    print(f"Running profiling with B={B}, STEPS={STEPS}, DEVICE={DEVICE}, "
          f"VERBOSE={VERBOSE}, PROPILE={PROPILE}, COMPILE={COMPILE}")

    const.move_constants_to(DEVICE)

    # Patch TorchBoard for profiling
    auto_profile_class(TorchBoard, {
        method: f"TorchBoard.{method}"
        for method in dir(TorchBoard)
        if callable(getattr(TorchBoard, method)) and not method.startswith("_")
    })

    # Create TorchBoard with many positions
    boards = [chess.Board() for _ in range(B)]
    boards = TorchBoard.from_board_list(boards, device=torch.device(DEVICE))

    if COMPILE:
        boards = torch.compile(boards)

    # Enable profiler
    if PROPILE:    
        profiler.reset()
        auto_profile_class(TorchBoard, {
            method: f"TorchBoard.{method}"
            for method in dir(TorchBoard)
            if callable(getattr(TorchBoard, method)) and not method.startswith("_")
        })
        profiler.enable()

    # Simulate random operations
    with profiler.time_block("full_run"):
        for i in range(STEPS):
            lm, _ = boards.get_moves()
            is_terminal, result = boards.is_game_over()
            if is_terminal.any():    
                boards = boards[~is_terminal]
                lm = lm.select(~is_terminal)

            logits = torch.zeros_like(lm.encoded, dtype=torch.float32)  # Simulate logits
            ks = torch.ones(lm.encoded.shape[0], dtype=torch.long)  # Sample one move per board
            moves, b_idx, _, _ = lm.sample_k(logits, ks)
            boards = boards.push(moves, b_idx)


    profiler.print_summary() if PROPILE else None
    profiler.reset()  # Reset profiler for next run if needed
