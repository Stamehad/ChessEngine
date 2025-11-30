import chess # type: ignore
import os
from dotenv import load_dotenv
import yaml # type: ignore
import torch
from tqdm import tqdm

from model.engine_pl import ChessLightningModule
from pytorchchess import TorchBoard
from pytorchchess.beam_search.beam_search import BeamSearchState
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    torch.manual_seed(1)
    GAMES = args.games # Number of games to test
    DEVICE = args.device # Device to use for computation
    STEPS = args.steps # Number of steps to simulate

    if DEVICE == "gpu" and not torch.cuda.is_available():
        raise ValueError("GPU is not available. Please use 'cpu' or 'mps' instead.")
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Please use 'cpu' or 'cuda' instead.")

    #----------------------------------------
    # Load model from checkpoint
    #----------------------------------------
    load_dotenv()
    CHECKPOINT_PATH = os.getenv("BASE_MODEL")

    # load config from the file engine_config.yaml
    with open("engine_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
    model.eval() # Set model to evaluation mode
    model.to(DEVICE)

    #----------------------------------------
    # Initialize beam search parameters
    #----------------------------------------
    ks = torch.tensor([8, 5, 3, 2, 1, 1, 1], device=DEVICE)  # Top-k moves to select at each step
    ks = torch.tensor([5, 3, 2, 1, 1], device=DEVICE)  # Top-k moves to select at each step
    ks = torch.tensor([3, 2, 1], device=DEVICE)  # Top-k moves to select at each step
    ks = torch.tensor([2, 2, 2], device=DEVICE)  # Top-k moves to select at each step

    # Initialize unified beam search state
    state = BeamSearchState.create(GAMES, ks, device=DEVICE)

    #----------------------------------------
    # Initialize torch boards
    #----------------------------------------
    boards = [chess.Board() for _ in range(GAMES)]  # Create a list of chess boards
    torch_boards = TorchBoard.from_board_list(boards, device=DEVICE, LAYER=True) # (B, 8, 8)

    # torch board (tb) for beam search
    tb = torch_boards.clone()  # Clone the initial torch board for beam search    
    
    #for step in tqdm(range(steps), desc="Generating moves"):
    for step in range(STEPS):
        print()
        print(f"Step {step+1}/{STEPS}")
        print(f"Current positions: {state.idx.shape[0]}")
        print(f"Depths: {state.depth}")
        print(f"Games: {state.game}")
        print(f"Stacks: {state.stack}")

        #---------------------------------------
        # Get legal moves
        #---------------------------------------
        
        #---------------------------------------
        # filter dead positions (no legal moves)
        #---------------------------------------
        #dead_positions, result = tb.is_game_over() # (B,), (N_over,)
        dead_positions = torch.randint(0, 8, (state.idx.shape[0],), device=DEVICE) < 1  # Simulate some dead positions for testing
        result = torch.randint(0, 2, (dead_positions.sum().item(),), device=DEVICE).float()  # Simulate results for dead positions
        result = result * 2 - 1  # Convert to -1, 0, 1 for loss, draw, win
        if dead_positions.any():
            print(f"Dead positions: {dead_positions.sum()} out of {dead_positions.shape[0]}")
            
            state.store_early_terminated_evaluations(dead_positions, 2 * result)

            tb = tb.select(~dead_positions)
            state = state[~dead_positions]

        #---------------------------------------
        # model inference
        #---------------------------------------
        x = tb.feature_tensor().float()            # (B, 8, 8, 21)
        with torch.no_grad():
            x_out, move_pred = model(x) # (1, 65, H), (1, 64, 7)
            x_out = x_out.detach()  # Detach to avoid gradients
            move_pred = move_pred.detach()  # Detach to avoid gradients
            
            eval = model.loss_module.prob_eval_loss.head(x_out) # (B, 3)
            prob_eval = torch.nn.functional.softmax(eval, dim=-1)  # (B, 3)
            scalar_eval = prob_eval[:, 0] - prob_eval[:, 2]  # (B,)

        #----------------------------------------------------
        # Separate boards that are fully expanded
        #----------------------------------------------------
        
        # Use new method instead of manual check
        finished_expansion = state.get_finished_expansion()
        if finished_expansion.any():
            print(f"Finished boards: {finished_expansion.sum()} out of {finished_expansion.shape[0]}")

            state.store_terminal_evaluations(finished_expansion, scalar_eval[finished_expansion])

            tb = tb.select(~finished_expansion) 
            state = state[~finished_expansion]
            move_pred = move_pred[~finished_expansion]

        print(f"Evaluation stacks: {state.eval_stacks}")
        print(f"Move stacks: {state.move_stacks}")

        #----------------------------------------------------
        # Backprop and get principal variation (PV) moves
        #----------------------------------------------------
        # Use new method instead of manual calculation
        finished_stack = state.get_finished_stack(step)
        if finished_stack is not None:
            pv_values, pv_moves, target_layer = state.backpropagate(finished_stack)
            if pv_values is not None:
                print(f"Backpropagated evaluations: {pv_values}")
                print(f"Principal Variation moves: {pv_moves}")
                print(f"Target layer for PV: {target_layer}")
        
        #----------------------------------------
        # expand by choosing top-k moves
        #----------------------------------------
        if state.idx.shape[0] > 0:  # Only expand if there are positions
            new_moves, b_idx, m_idx = tb.get_topk_legal_moves(move_pred, ks=ks[state.depth]) # (B_new, ), (B_new, ), (B_new,)
            
            # Simplified expansion - single method call
            state = state.expand_positions(m_idx)
            
            #-----------------------------------------
            # push moves and store them
            #-----------------------------------------
            tb = tb.push(new_moves, b_idx)
            state.store_moves(new_moves)
        
        #----------------------------------------------------
        # Add new boards for next iteration
        #----------------------------------------------------
        state = state.add_new_stack(step + 1)

        new_boards = [chess.Board() for _ in range(GAMES)]
        new_tb = TorchBoard.from_board_list(new_boards, device=DEVICE, LAYER=True) # (B, 8, 8)
        tb = tb.concat(new_tb)  # Concatenate new boards

    print("\nTest completed successfully!")
    print(f"Final state - Positions: {state.idx.shape[0]}")
    print(f"Evaluation stacks: {state.eval_stacks}")
    print(f"Move stacks: {state.move_stacks}")
