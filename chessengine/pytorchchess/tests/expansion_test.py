import chess # type: ignore
import os
from dotenv import load_dotenv
import yaml # type: ignore
import torch
from tqdm import tqdm

from model.engine_pl import ChessLightningModule
from pytorchchess import TorchBoard
from pytorchchess.beam_search.beam_state import BeamState
from pytorchchess.beam_search.eval_state import EvalStates
from pytorchchess.beam_search.moves import Moves
        
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
    #ks = torch.tensor([2, 2, 2], device=DEVICE)  # Top-k moves to select at each step

    # exp_dim = torch.cumprod(ks[1:].flip(0), dim=0).flip(0)
    # exp_dim = torch.cat([exp_dim, torch.tensor([1], device=DEVICE)])  # (L+1,)
    # game_stride = ks.prod().item()

    MOVE_PAD = -1 # 2**15  # Padding value for moves

    beam = BeamState.from_num_games(GAMES, ks, device=DEVICE)  # Initial beam state
    evals = EvalStates.empty(ks, GAMES)  # Evaluation states for each game    
    moves = Moves(exp_f=ks, g=GAMES, PAD=MOVE_PAD)  # Moves state

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
        #print(tb)
        print(beam)

        #---------------------------------------
        # Get legal moves
        #---------------------------------------
        tb.get_legal_moves(get_tensor=True)                  # LegalMoves
        
        #---------------------------------------
        # filter dead positions (no legal moves)
        #---------------------------------------
        #dead_positions, result = tb.is_game_over() # (B,), (N_over,)
        dead_positions = torch.randint(0, 8, (beam.idx.shape[0],), device=DEVICE) < 1  # Simulate some dead positions for testing
        result = torch.randint(0, 2, (dead_positions.sum().item(),), device=DEVICE).float()  # Simulate results for dead positions
        result = result * 2 - 1  # Convert to -1, 0, 1 for loss, draw, win
        if dead_positions.any():
            print(f"Dead positions: {dead_positions.sum()} out of {dead_positions.shape[0]}")
            
            layer_terminal = tb.state.layer[dead_positions]  # Get layers of dead positions
            idx_terminal, stack_terminal = beam.compute_flat_indices(dead_positions)
            evals.update_early_terminated(idx_terminal, stack_terminal, 2 * result, layer_terminal)  # Update EvalStates with terminal results

            tb = tb.select(~dead_positions)
            beam = beam[~dead_positions]

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
        finished_expansion = beam.layer >= ks.shape[0]  # Boards that are fully expanded
        if finished_expansion.any():
            print(f"Finished boards: {finished_expansion.sum()} out of {finished_expansion.shape[0]}")

            flat_idx, finished_stack = beam.compute_flat_indices(finished_expansion)
            evals.update_terminal(flat_idx, finished_stack, scalar_eval[finished_expansion]) 

            tb = tb.select(~finished_expansion) 
            beam = beam[~finished_expansion]
            move_pred = move_pred[~finished_expansion]

        print(evals)

        #----------------------------------------------------
        # Backprop and get principal variation (PV) moves
        #----------------------------------------------------
        L = len(ks)  # Number of layers
        finished_stack = step - L if step >= L else None  # Use step as stack index for finished boards
        if finished_stack is not None:
            v, PV_idx = evals.backprop(finished_stack) # Backpropagate evaluations, PV_idx.shape = (G,)
            PV = moves.finished_moves(finished_stack, PV_idx) # (G, L)

            print(f"Backpropagated evaluations: {v}")
            print(f"Principal Variation moves: {PV}")
        

        #----------------------------------------
        # expand by choosing top-k moves
        #----------------------------------------
        new_moves, b_idx, m_idx = tb.get_topk_legal_moves(move_pred, ks=ks[beam.layer]) # (B_new, ), (B_new, ), (B_new,)
        beam = beam.repeat_interleave(m_idx)  # Repeat beam state according to top-k moves
        
        #-----------------------------------------
        # push moves
        #-----------------------------------------
        tb = tb.push(new_moves, b_idx)
        flat_idx, stack = beam.compute_flat_indices()  # Compute flat indices for the new beams
        moves.update(stack, beam.layer, flat_idx, new_moves)
        
        #----------------------------------------------------
        # Add new boards
        #----------------------------------------------------
        beam.add_stack(step + 1)  
        moves.new_stack(step + 1) 

        new_boards = [chess.Board() for _ in range(GAMES)]
        new_tb = TorchBoard.from_board_list(new_boards, device=DEVICE, LAYER=True) # (B, 8, 8)
        tb = tb.concat(new_tb)  # Concatenate new boards
