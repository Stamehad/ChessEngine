import chess.pgn # type: ignore
import chess # type: ignore
import os
import io
from dotenv import load_dotenv
import json
import yaml # type: ignore
import torch
from preprocessing.position_parsing import encode_board
#from pytorchchess import TorchBoard
from pytorchchess.utils import int_to_squares
from tqdm import tqdm
from model.engine_pl import ChessLightningModule
from model.utils import masked_one_hot
from pytorchchess.utils.constants import move_constants_to
import pytorchchess.utils.constants as c

import time


def sample_one_move_per_valid_board(lm, mask):
    """
    lm:   (B, L_max) LongTensor of legal moves (padded)
    mask: (B, L_max) BoolTensor indicating valid entries

    Returns:
        selected_moves: (B_valid,) LongTensor
        b_indices:      (B_valid,) LongTensor
    """
    B, L_max = lm.shape

    valid_board_mask = mask.any(dim=1)              # (B,) → True for boards with ≥1 legal move
    valid_indices = torch.nonzero(valid_board_mask).squeeze(1)  # Indices of valid boards

    selected_moves = []
    b_indices = []

    for b in valid_indices:
        valid_moves = lm[b][mask[b]]  # (Lᵢ,)
        idx = torch.randint(0, valid_moves.shape[0], (1,)).item()
        selected_moves.append(valid_moves[idx])
        b_indices.append(b)

    return torch.tensor(selected_moves), torch.tensor(b_indices)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--moves", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"],)
    args = parser.parse_args()

    GAMES = args.games # Number of games to test
    MOVES = args.moves # Number of moves to generate per game
    DEVICE = args.device # Device to use for computation

    if DEVICE == "gpu" and not torch.cuda.is_available():
        raise ValueError("GPU is not available. Please use 'cpu' or 'mps' instead.")
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Please use 'cpu' or 'cuda' instead.")

    c.move_constants_to(torch.device(DEVICE))  # ✅ Load constants onto the right device
    tensor = c.SHORT_RANGE_MOVES

    load_dotenv()
    CHECKPOINT_PATH = os.getenv("BASE_MODEL")

    # load config from the file engine_config.yaml
    with open("engine_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
    model.eval() # Set model to evaluation mode
    model.to(DEVICE)
    from pytorchchess import TorchBoard

    boards = [chess.Board() for _ in range(GAMES)]
    torch_boards = TorchBoard.from_board_list(boards, device=DEVICE) # (B, 8, 8)

    inference_time = 0.0
    move_time = 0.0
    for step in tqdm(range(MOVES), desc="Generating moves"):
        time_start = time.time()
        legal_moves, features = torch_boards.get_moves()
        lm_mask = legal_moves.mask
        x = features.float()        # (B, 8, 8, 21)
        end_time = time.time()
        move_time += end_time - time_start
        #---------------------------------------
        # model inference
        #---------------------------------------
        time_start = time.time()
        with torch.no_grad():
            x_out, move_pred = model(x) # (1, 65, H), (1, 64, 7)
            x_out = x_out.detach()  # Detach to avoid gradients
            move_pred = move_pred.detach()  # Detach to avoid gradients
        
        move_logits = legal_moves.get_logits(move_pred)

        end_time = time.time()
        inference_time += end_time - time_start
        #----------------------------------------
        # Sample one move per valid board
        #----------------------------------------
        # choose most likely move
        start_time = time.time()
        m_idx = move_logits.argmax(dim=1).long() # (B,)
        b_idx = torch.arange(m_idx.shape[0], dtype=torch.long, device=m_idx.device)  # batch indices

        moves = legal_moves.encoded[b_idx, m_idx]

        #-----------------------------------------
        # push moves
        #-----------------------------------------
        torch_boards = torch_boards.push(moves, b_idx)

        # print(torch_boards.board_tensor.shape)
        #torch_boards.render(0)  # Render the board after each move
        end_time = time.time()
        move_time += end_time - start_time

    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Total move time: {move_time:.4f} seconds")
    print(f"Average inference time per move: {inference_time / MOVES:.4f} seconds")
    print(f"Average move time per move: {move_time / MOVES:.4f} seconds")
