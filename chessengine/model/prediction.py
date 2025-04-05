import torch
import chess
import numpy as np
from typing import Tuple, List
from chessengine.preprocessing.position_parsing import encode_board, generate_legal_move_tensor
from chessengine.model.utils import masked_one_hot

def get_move(move_tensor: torch.tensor) -> chess.Move:
    """
    Args:
        move_tensor: A tensor of shape (64,) with each square has a value of 0-6 or -100.
        E.g. value 0 means the square was vacated by a piece, 1-6 type of piece that moved to that square.
        -100 means no change to that square. 
        (Black and white have the same piece number as that leads to no ambiguity.)
        Castling could involve 3 squares for king side and 4 squares for queen side.
        (E1 = 4 (rook), F1 = 6 (king), H1 = 0 (vacated) of white king side castling)
        En passant involves 3 squares.
        
    Returns:
        A chess.Move object representing the move made.
    """
    pass

def generate_legal_moves(board: chess.Board):
    legal_move_tensors = []
    legal_move_list = []

    for move in board.legal_moves:
        legal_move_list.append(move)

        move_tensor = torch.full((64,), -100, dtype=torch.int8)

        # From square becomes empty
        move_tensor[move.from_square] = 0

        # Handle en passant BEFORE push
        if board.is_en_passant(move):
            ep_row_offset = -1 if board.turn == chess.WHITE else 1
            captured_square = move.to_square + (8 * ep_row_offset)
            move_tensor[captured_square] = 0

        # Handle castling BEFORE push
        if board.is_castling(move):
            rook_from_to = {
                (chess.E1, chess.G1): (chess.H1, chess.F1),
                (chess.E1, chess.C1): (chess.A1, chess.D1),
                (chess.E8, chess.G8): (chess.H8, chess.F8),
                (chess.E8, chess.C8): (chess.A8, chess.D8),
            }
            rf, rt = rook_from_to.get((move.from_square, move.to_square), (None, None))
            if rf is not None and rt is not None:
                move_tensor[rf] = 0
                move_tensor[rt] = 4  # rook type

        # Push move to observe post-move board
        board.push(move)
        piece = board.piece_at(move.to_square)
        if piece:
            move_tensor[move.to_square] = piece.piece_type
        board.pop()

        legal_move_tensors.append(move_tensor)

        
    if len(legal_move_tensors) == 0:
        return torch.empty((64, 0), dtype=torch.int8), []

    return torch.stack(legal_move_tensors, dim=1), legal_move_list # (64, L), [chess.Move]

def predict(model, board, device) -> Tuple[chess.Move, chess.Move, chess.Move, float]:
    """
    Predict the best move for a given board state using the provided model.

    Args:
        model: The trained chess model.
        board: The current state of the chess board. 

    Returns:
        A tuple containing the predicted move and the predicted value.
    """
    # Convert the board to a tensor and add a batch dimension
    x = encode_board(board).unsqueeze(0).float().to(device) # (1, 8, 8, 21)

    legal_moves, legal_moves_list = generate_legal_moves(board) # (64, L)
    legal_moves = legal_moves.unsqueeze(0).long().to(device)  # (1, 64, L) 

    # Get the model's predictions
    with torch.no_grad():
        model.eval()
        x_out, move_pred = model(x) # (1, 65, H), (1, 64, 7)

        eval = model.loss_module.eval_loss.head(x_out).squeeze(0) # (1,)
        eval = eval.to("cpu").item()  # Convert to CPU for easier handling
        
        legal_moves_one_hot = masked_one_hot(legal_moves, num_classes=7, mask_value=-100)  # (1, 64, L, 7)
        legal_moves_mask = (legal_moves != -100).any(dim=1)  # (1, L)
        move_pred = move_pred.unsqueeze(2)  # (1, 64, 1, 7)
        move_logits = (move_pred * legal_moves_one_hot).sum(dim=-1)  # (1, 64, L)

        # Average over changed squares
        mask = (legal_moves != -100)  # (1, 64, L)
        move_logits = move_logits.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (1, L)
        move_logits = move_logits.masked_fill(~legal_moves_mask, float('-inf'))

        move_logits = move_logits.to("cpu")

        # Get the index of the 3 best moves
        num_moves = len(legal_moves_list)

        if num_moves == 0:
            move1 = move2 = move3 = None  # or raise an exception if that suits you better
        elif num_moves == 1:
            move1 = legal_moves_list[0]
            move2 = move3 = None
        elif num_moves == 2:
            _, top_indices = torch.topk(move_logits, k=2, dim=-1)
            move1 = legal_moves_list[top_indices[0, 0]]
            move2 = legal_moves_list[top_indices[0, 1]]
            move3 = None
        else:
            _, top_indices = torch.topk(move_logits, k=3, dim=-1)
            move1 = legal_moves_list[top_indices[0, 0]]
            move2 = legal_moves_list[top_indices[0, 1]]
            move3 = legal_moves_list[top_indices[0, 2]]

    return move1, move2, move3, eval

