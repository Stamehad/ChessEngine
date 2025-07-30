import chess # type: ignore
import torch
from pytorchchess.state.game_state import GameState
from pytorchchess.utils.utils import move_dtype, encode_move

def piece_to_index(piece):
    """Maps a chess.Piece to channel index (1-12), or 0 for empty"""
    if piece is None:
        return 0
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }[piece.piece_type]

def board_enconding(board: chess.Board, device) -> torch.Tensor:
    plane = torch.zeros((8, 8), dtype=torch.uint8, device=device)

    # 0–12: piece encodings
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        row, col = divmod(square, 8)
        plane[row][col]= piece_to_index(piece)

    #print(plane.flip(0))
    
    return plane

def encode(boards: list[chess.Board], device: str = "cpu") -> torch.Tensor:
    if isinstance(boards, list):
        assert len(boards) > 0, "Input list of boards cannot be empty."
    if not isinstance(boards, list):
        boards = [boards]
    if not all(isinstance(b, chess.Board) for b in boards):
        raise TypeError("All elements in boards must be chess.Board instances.")
    
    tensor_boards = [board_enconding(b, device=device) for b in boards] 
    tensor_boards = torch.stack(tensor_boards, dim=0)  # (B, 8, 8)
    
    return tensor_boards

def get_side(boards: list[chess.Board], device) -> torch.Tensor:
    side = torch.tensor([1 if b.turn == chess.WHITE else 0 for b in boards], dtype=torch.uint8, device=device)
    
    return side.view(-1, 1)  # (B, 1)

def get_fullmoves(boards: list[chess.Board], device) -> torch.Tensor:
    plys = torch.tensor([b.fullmove_number - 1 for b in boards], dtype=torch.long, device=device)
    
    return plys # (B,)

def castling_rights(boards: list[chess.Board], device) -> torch.Tensor:
    castling_rights = [[b.has_kingside_castling_rights(chess.WHITE),
                       b.has_queenside_castling_rights(chess.WHITE),
                       b.has_kingside_castling_rights(chess.BLACK),
                       b.has_queenside_castling_rights(chess.BLACK)]
                        for b in boards]
    castling_rights = torch.tensor(castling_rights, dtype=torch.uint8, device=device)
    castling_rights = castling_rights.view(-1, 4)  # (B, 4)
    
    return castling_rights

def ep_square(boards: list[chess.Board], device) -> torch.Tensor:
    ep_squares = torch.tensor([b.ep_square if b.ep_square is not None else 64 for b in boards], dtype=torch.uint8, device=device)
    ep_squares = ep_squares.view(-1)  # (B,)
    
    return ep_squares

def get_fifty_move_clock(boards: list[chess.Board], device) -> torch.Tensor:
    """Extract fifty-move clock (halfmoves since last pawn move or capture) from boards"""
    fifty_move_clocks = torch.tensor([b.halfmove_clock for b in boards], dtype=torch.uint8, device=device)
    return fifty_move_clocks  # (B,)

def get_previous_moves(boards: list[chess.Board], device) -> torch.Tensor:
    """
    Returns a (B,) int16 tensor encoding the previous move of each board.

    Encoding:
        move = from_sq + to_sq * 64 + move_type * 4096
        If no previous move exists: move = 32768 (i.e., padding flag)
    """
    encoded_moves = []
    for board in boards:
        if board.move_stack:
            last_move = board.peek()
            encoded = encode_move(last_move, board)
        else:
            encoded = 32768  # padding flag (2^15)
        encoded_moves.append(encoded)

    return torch.tensor(encoded_moves, dtype=torch.int16, device=device)

def state_from_board(boards: list[chess.Board], device = torch.device("cpu")) -> GameState:
    """
    Convert a list of chess.Board instances to a GameState object.
    
    Args:
        boards (list[chess.Board]): List of chess.Board instances.
        device (str): Device to use for the tensor (default: "cpu").
    
    Returns:
        GameState: A GameState object containing the board state.
    """
    if not isinstance(boards, list):
        boards = [boards]
    if not all(isinstance(b, chess.Board) for b in boards):
        raise TypeError("All elements in boards must be chess.Board instances.")
    
    side = get_side(boards, device=device)  # (B, 1)
    full_moves = get_fullmoves(boards, device=device)  # (B,)
    plys = torch.where(side.squeeze() == 1, full_moves * 2, full_moves * 2 + 1)  # (B,) - convert to plys (0 for white, 1 for black)
    castling = castling_rights(boards, device=device)  # (B, 4)
    ep = ep_square(boards, device=device)  # (B,)
    fifty_move_clock = get_fifty_move_clock(boards, device=device)  # (B,)
    position_history = torch.zeros((len(boards), 0), dtype=torch.long, device=device)  # (B, 0) - empty history for now
    
    # previous_moves = torch.tensor([-1]*ep.shape[0], dtype=move_dtype(device), device=device)  # (B,) - no previous moves
    previous_moves = get_previous_moves(boards, device) # (B,) int16
    
    return GameState(
        side_to_move=side,  # (B, 1)
        plys=plys,  # (B,)
        castling=castling,  # (B, 4)
        ep=ep,  # (B,)
        previous_moves=previous_moves,  # (B,)
        fifty_move_clock=fifty_move_clock,  # (B,)
        position_history=position_history  # (B, 0)
    )

PIECE_SYMBOLS = {
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

def tensor_to_fen(tensor_8x8, side, castling, ep):
    """
    Convert a (8,8) board tensor back into a FEN string.
    
    Args:
        tensor_8x8 (torch.Tensor): (8, 8) tensor of piece indices.
        side (int): 0 = white to move, 1 = black to move.
        castling (list of int): [K, Q, k, q] castling rights.
        ep (int): En passant square index (0–63) or invalid (e.g., 64).
    
    Returns:
        str: FEN string.
    """
    tensor_flat = tensor_8x8.flip(0).view(-1)  # (64,)
    fen_rows = []

    for rank in range(8):
        row_str = ""
        empty = 0
        for file in range(8):
            idx = rank * 8 + file
            val = tensor_flat[idx].item()
            if val == 0:
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += PIECE_SYMBOLS[val]
        if empty > 0:
            row_str += str(empty)
        fen_rows.append(row_str)

    fen_board = "/".join(fen_rows)

    fen_side = "w" if side == 0 else "b"

    flags = ['K', 'Q', 'k', 'q']
    fen_castling = "".join(f for f, v in zip(flags, castling) if v) or "-"

    if 0 <= ep < 64:
        file = ep % 8
        rank = ep // 8
        ep_str = f"{chr(ord('a') + file)}{rank + 1}"
    else:
        ep_str = "-"

    return f"{fen_board} {fen_side} {fen_castling} {ep_str} 0 1"