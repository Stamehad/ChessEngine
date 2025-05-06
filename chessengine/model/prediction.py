import torch
import chess
import numpy as np
from typing import Tuple, List
from chessengine.preprocessing.position_parsing import encode_board, generate_legal_move_tensor
from chessengine.model.utils import masked_one_hot, batch_legal_moves

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

def generate_legal_moves(board: chess.Board) -> Tuple[torch.Tensor, List[chess.Move]]:
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

def get_batch_legal_moves(boards: List[chess.Board]) -> Tuple[torch.Tensor, List[List[chess.Move]]]:
    """
    Args:
        boards: List of chess.Board objects.
    Returns:
        A tensor of shape (B, 64, L_max) where B is the number of boards and L_max is the maximum number of legal moves
        across all boards. The tensor contains -100 for empty squares and 0-6 for piece types.
        A list of lists of chess.Move objects corresponding to each board.
    """
    legal_move_tensors = []
    legal_move_lists = []
    for b in boards:
        legal_moves, legal_move_list = generate_legal_moves(b)
        legal_move_tensors.append(legal_moves)
        legal_move_lists.append(legal_move_list)
    legal_move_tensors, mask = batch_legal_moves(legal_move_tensors) # (B, 64, L_max), (B, L_max)
    ################# CAUTION #################
    # legal_move_lists is not padded to L_max!
    ###########################################
    return legal_move_tensors, legal_move_lists # (B, 64, L_max), [List[chess.Move]]

def get_eval_prob(model, x_out: torch.Tensor, device = "cpu") -> torch.Tensor:
    eval = model.loss_module.prob_eval_loss.head(x_out) # (B, 3)
    prob_eval = torch.nn.functional.softmax(eval, dim=-1)  # (B, 3)
    prob_eval = prob_eval.detach().to(device)  # Convert to CPU for easier handling

    return prob_eval # (B, 3)

def get_move_probs(move_pred: torch.Tensor, legal_moves: torch.Tensor, device="cpu") -> torch.Tensor:
    legal_moves_one_hot = masked_one_hot(legal_moves, num_classes=7, mask_value=-100)  # (B, 64, L, 7)
    legal_moves_mask = (legal_moves != -100).any(dim=1)  # (B, L)
    move_pred = move_pred.unsqueeze(2)  # (B, 64, 1, 7)
    move_logits = (move_pred * legal_moves_one_hot).sum(dim=-1)  # (B, 64, L)

    # Average over changed squares
    mask = (legal_moves != -100)  # (B, 64, L)
    move_logits = move_logits.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, L)
    move_logits = move_logits.masked_fill(~legal_moves_mask, float('-inf'))

    move_logits = move_logits.detach().to(device)
    probs = torch.nn.functional.softmax(move_logits, dim=-1) # (B, L)

    return probs # (B, L)

def predict(model, board, device, out_device="cpu") -> Tuple[torch.Tensor, torch.Tensor, List[chess.Move]]:
    """
    Predict the best move for a given board state using the provided model.

    Args:
        model: The trained chess model.
        board: The current state of the chess board. 

    Returns:
        probs: A tensor of shape (L,) containing the probabilities of each legal move.
        prob_eval: A tensor of shape (3,) containing the evaluation probabilities (black win, draw, white win).
        legal_moves_list: A list of legal moves in chess.Move format.
    """
    # Convert the board to a tensor and add a batch dimension
    x = encode_board(board).unsqueeze(0).float().to(device) # (1, 8, 8, 21)

    legal_moves, legal_moves_list = generate_legal_moves(board) # (64, L)
    legal_moves = legal_moves.unsqueeze(0).long().to(device)  # (1, 64, L) 

    # Get the model's predictions
    with torch.no_grad():
        model.eval()
        x_out, move_pred = model(x) # (1, 65, H), (1, 64, 7)
        prob_eval = get_eval_prob(model, x_out, device=out_device).squeeze(0) # (3,)
        probs = get_move_probs(move_pred, legal_moves, device=out_device).squeeze(0) # (L,)

    return probs, prob_eval, legal_moves_list # (L,), (3,), [chess.Move]

def batch_predict(model, boards: List[chess.Board], device="cuda", out_device="cpu"):
    """
    Predict move distributions and evaluations for a batch of board positions.

    Args:
        model: The trained chess model.
        boards: List of chess.Board objects.
        device: Device to perform model inference on ("cuda" or "cpu").

    Returns:
        probs: Tensor of shape (B, L_max) with move probabilities (padded).
        prob_eval: Tensor of shape (B, 3) with (black win, draw, white win) distributions.
        legal_move_lists: List of lists of chess.Move objects per board.
    """
    if isinstance(boards, chess.Board):
        boards = [boards]
    x_batch = [encode_board(b) for b in boards]          # List[Tensor (8,8,21)]
    x_batch = torch.stack(x_batch).float().to(device)    # (B, 8, 8, 21)

    legal_moves, legal_move_lists = get_batch_legal_moves(boards) # (B, 64, L), [List[chess.Move]]
    legal_moves = legal_moves.long().to(device)  # (B, 64, L)

    with torch.no_grad():
        model.to(device)
        model.eval()
        x_out, move_pred = model(x_batch)                # (B, 65, H), (B, 64, 7)
        prob_eval = get_eval_prob(model, x_out, device=out_device) # (B, 3)
        probs = get_move_probs(move_pred, legal_moves, device=out_device) # (B, L)

    # Debugging
    assert probs.shape[0] == len(boards) == len(legal_move_lists)

    return probs, prob_eval, legal_move_lists # (B, L), (B, 3), [List[chess.Move]]

def predict3(model, board, device) -> Tuple[chess.Move, chess.Move, chess.Move, float]:
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
        prob_eval = get_eval_prob(model, x_out, device="cpu") # (3,)
        probs = get_move_probs(move_pred, legal_moves, device="cpu") # (L,)

        # Get the index of the 3 best moves
        num_moves = len(legal_moves_list)

        if num_moves == 0:
            move1 = move2 = move3 = None 
            p1 = p2 = p3 = None
        elif num_moves == 1:
            move1 = legal_moves_list[0]
            move2 = move3 = None
            p1, p2, p3 = probs[0], None, None
        elif num_moves == 2:
            p, idx = torch.topk(probs, k=2)
            move1 = legal_moves_list[idx[0]]
            move2 = legal_moves_list[idx[1]]
            move3 = None
            p1, p2, p3 = p[0], p[1], None
        else:
            p, idx = torch.topk(probs, k=3)
            move1 = legal_moves_list[idx[0]]
            move2 = legal_moves_list[idx[1]]
            move3 = legal_moves_list[idx[2]]
            p1, p2, p3 = p[0], p[1], p[2]

    return move1, move2, move3, p1, p2, p3, prob_eval

def mcts_predict(model, board, mcts, device="mps", out_device="cpu"):
    """
    Use MCTS to predict the best moves for a given board state.

    Args:
        model: The trained chess model.
        board: The current state of the chess board.
        mcts: The MCTS object initialized with the model.
        device: Device for inference ("cuda", "mps", "cpu").
        out_device: Where to place outputs.

    Returns:
        A tuple containing the predicted top 3 moves and the predicted evaluation.
    """

     # Convert the board to a tensor and add a batch dimension
    x = encode_board(board).unsqueeze(0).float().to(device) # (1, 8, 8, 21)

    legal_moves, _ = generate_legal_moves(board) # (64, L)
    legal_moves = legal_moves.unsqueeze(0).long().to(device)  # (1, 64, L) 

    # Get the model's predictions
    with torch.no_grad():
        model.eval()
        x_out, _ = model(x) # (1, 65, H), (1, 64, 7)
        prob_eval = get_eval_prob(model, x_out, device="cpu") # (3,)

    # Run MCTS search
    pis = mcts.run_mcts_search([board])  # policies: List[Dict[chess.Move, prob]], evals: Tensor (B, 3)
    legal_moves, pi = pis[0] 
    num_moves = len(legal_moves)
    k = 3
    if num_moves < 3:
        k = num_moves

    p, idx = torch.topk(pi, k=k)
    
    move1 = move2 = move3 = None 
    p1 = p2 = p3 = None
    if num_moves == 1:
        move1 = legal_moves[0]
        p1 = p[0]
    elif num_moves == 2:
        move1 = legal_moves[idx[0]]
        move2 = legal_moves[idx[1]]
        p1 = p[0]
        p2 = p[1]
    else:
        p, idx = torch.topk(pi, k=3)
        move1 = legal_moves[idx[0]]
        move2 = legal_moves[idx[1]]
        move3 = legal_moves[idx[2]]
        p1, p2, p3 = p[0], p[1], p[2]

    return move1, move2, move3, p1, p2, p3, prob_eval