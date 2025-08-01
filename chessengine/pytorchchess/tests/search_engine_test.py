import chess
import os
from dotenv import load_dotenv
import yaml
import torch

from model.engine_pl import ChessLightningModule
from pytorchchess import TorchBoard
from pytorchchess.beam_search.search_engine import BeamSearchEngine
import pytorchchess.utils.constants as const
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)  # CHANGED: steps instead of cycles
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--debug", action='store_true', help="Enable debug output")
    parser.add_argument("--large", action='store_true', help="Choose expansion factors for larger search depth")
    args = parser.parse_args()

    seed = 4
    torch.manual_seed(seed)
    GAMES = args.games
    DEVICE = torch.device(args.device)
    STEPS = args.steps
    VERBOSE = args.verbose
    DEBUG = args.debug


    # Load model
    load_dotenv()
    CHECKPOINT_PATH = os.getenv("BASE_MODEL")

    with open("engine_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
    model.eval()
    model.to(DEVICE)

    if DEVICE.type == "cuda":
        model = torch.compile(model)
        print("Model compiled for CUDA")

    # Initialize beam search engine
    if args.large:
        expansion_factors = torch.tensor([8, 5, 3, 2, 1, 1, 1], device=DEVICE)  # L = 7 (odd depth)
    else:
        expansion_factors = torch.tensor([3, 2, 1], device=DEVICE)  # Top-k moves to select at each step
        #expansion_factors = torch.tensor([2, 2, 2], device=DEVICE)  # Top-k moves to select at each step
    L = len(expansion_factors)

    print(f"VERBOSE: {VERBOSE}, DEBUG: {DEBUG}")
    
    engine = BeamSearchEngine(
        model=model,
        expansion_factors=expansion_factors,
        device=DEVICE,
        pv_depth=3,  # Apply first 3 PV moves
        verbose=VERBOSE,  # Enable verbose output
        debug=DEBUG,  # Enable debug output for detailed tracing
        profile=True,  # Enable profiling
        seed=seed
    )

    # Create initial positions
    const.move_constants_to(DEVICE)  # Move constants to the specified device
    initial_boards = [chess.Board() for _ in range(GAMES * (L + 1))]
    initial_torch_board = TorchBoard.from_board_list(initial_boards, device=DEVICE)
    
    # Initialize engine
    engine.initialize(initial_torch_board)
    exp_dim = torch.cumprod(expansion_factors, dim=0)
    print(f"Initialized engine with {L + 1} layers, {GAMES} games each")
    print(f"Expansion factors: {expansion_factors.tolist()}")
    print(f"Expected batch size per step: ~{(exp_dim.sum().item() + 1) * GAMES}")

    print(f"Devices: engine={engine.device}, model={model.device}, beam={engine.beam_state.idx.device}, queue={engine.position_queue.device}")

    # REMOVED: The confusing cycle loop
    # Now just run the continuous pipeline directly
    print(f"\nRunning continuous pipeline for {STEPS} steps...")
    

    engine.run_full_search(STEPS)

    all_positions = engine.position_queue.get_all_positions()
    plys = all_positions.state.plys
    terminal, result = all_positions.is_game_over()
    print(f"\nFinal positions game over: {terminal}")
    print(f"Final ply counts: {plys}")
    print(f"Final results: {result}")

    for i in range(len(all_positions)):
        all_positions.render(i)

    print(f"\n{'='*60}")
    print("CONTINUOUS PIPELINE COMPLETED")
    print(f"{'='*60}")
    print(f"Total steps: {STEPS}")
    #print(f"PV moves found and applied: {pv_count}")
    print(f"Final position queue size: {engine.get_all_positions().board_tensor.shape[0]}")
