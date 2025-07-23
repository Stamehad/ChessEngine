import chess
import chess.pgn

import torch
from dataclasses import dataclass
import os
import io
import json
from torch.profiler import profile, record_function, ProfilerActivity
import importlib

import chessengine.pytorchchess.torch_board.pseudo_move_gen_new as pgm
from chessengine.pytorchchess import TorchBoard
import pytorchchess.utils.constants_new as const_new
import pytorchchess.utils.constants as const

importlib.reload(pgm)



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
    parser.add_argument("--fullgraph", action="store_true", help="Force fullgraph compilation")
    args = parser.parse_args()

    torch.manual_seed(123456789)
    B = args.B
    DEVICE = torch.device(args.device)
    STEPS = args.steps
    VERBOSE = args.verbose
    COMPILE = args.compile
    TRACE_FILE = args.export_trace

    print(f"Using device {DEVICE}, compile = {COMPILE}")

    const.move_constants_to(DEVICE)
    const_new.move_constants_to(DEVICE)

    # ==============================================
    # SETUP TORCHBOARD STATES
    # ==============================================
    pgn_file = "data/shards300_small/shard_0.pgn"
    max_games = 8
    move_max = 37 #36
    # read the first 10 games from the pgn file
    with open(pgn_file, 'r') as f0:
        for game_idx, line in enumerate(f0):
            if game_idx != max_games:
                if game_idx > max_games:
                    break
                continue
            obj = json.loads(line)
            pgn = obj["pgn"]
            game = chess.pgn.read_game(io.StringIO(pgn))
            board = game.board()
            boards = [board]
            for move_idx, move in enumerate(game.mainline_moves()):
                # if move_idx >= move_max:
                #     break
                board.push(move)
                boards.append(board.copy())

    tb = TorchBoard.from_board_list(boards, device=DEVICE)

    tb = tb.concat(tb)

    print(f"Batch size = {len(tb)}")
    print(tb)

    # ==============================================
    # PSEUDO-MOVES CLASS
    # ==============================================
    pgm = pgm.PseudoMoveGeneratorNew()
    pgm.board_flat = tb.board_flat
    pgm.side_to_move = tb.side_to_move
    pgm.ep = tb.ep
    
    @dataclass
    class State:
        castling: torch.Tensor

    pgm.state = State(castling = tb.state.castling)
    #pgm.castling = tb.state.castling


    # ==============================================
    # COMPILE AND PROFILE
    # ==============================================
    import torch._dynamo
    import torch._inductor
    torch._inductor.config.debug = True
    torch._dynamo.config.verbose = True
    #torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    
    if args.fullgraph:
        pgm.get_moves       = torch.compile(pgm.get_moves, fullgraph=True)
        #pgm.get_moves_fused = torch.compile(pgm.get_moves_fused, fullgraph=True)
    else:
        pgm.get_moves       = torch.compile(pgm.get_moves, mode="default")
        #pgm.get_moves_fused = torch.compile(pgm.get_moves_fused, mode="default")

    # ==============================================
    # RUN WARM-UP AND PROFILING
    # ==============================================
    print("warm-up...")
    for _ in range(5):
        pgm.get_moves()  # Warm-up run
        #pgm.get_moves_fused()
    torch.cuda.synchronize()

    print("profiling...")

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        premoves, incheck = pgm.get_moves()
        #premoves, incheck = pgm.get_moves_fused()

    print(prof.key_averages().table(sort_by="cuda_time_total"))


    trace_file = f"trace_get_moves_{'fullgraph' if args.fullgraph else 'default'}.json"
    prof.export_chrome_trace(trace_file)
    print(f"Chrome trace exported to {trace_file}")