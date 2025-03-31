import chess.pgn
import torch
import json
from pathlib import Path
from tqdm import tqdm
import io


########################################################################################
########################################################################################
# | Channel | Description                                                              |
# |---------|--------------------------------------------------------------------------|
# | 0       | Empty square (1 if empty, else 0)                                        |
# | 1–6     | White pawn, knight, bishop, rook, queen, king                            |
# | 7–12    | Black pawn, knight, bishop, rook, queen, king                            |
# | 13      | Side to move (1 = white to move, 0 = black)                              |
# | 14      | In-check flag for the player's king (broadcasted to all squares)         |
# | 15      | Threatened flag — this piece is under threat (regardless of color)       |
# | 16      | Threatening flag — this piece threatens opponent pieces (player-relative)|
# | 17      | Legal move — this piece has at least one legal move                      |
# | 18      | White has castling rights (1 if yes, 0 if not)                           |
# | 19      | Black has castling rights (1 if yes, 0 if not)                           |
# | 20      | En passant square (1 at en passant target square, else 0)                |
########################################################################################
########################################################################################


##################################################################################################
##################################################################################################
# Each training batch consists of a dictionary with the following entries:
# | Key              | Shape      | Description                                                  |
# |------------------|------------|--------------------------------------------------------------|
# | `board`          | (8, 8, 20) | The raw input tensor for a position                          |
# | `eval`           | (1,)       | Scalar win target (2=white win, 1= draw, 0=black win)        |
# | `move_target`    | (64,)      | Labels for changed squares (0=empty, 1–6=piece type)         |
# | `king_square`    | (1,)       | Index [0–63] of opponent king square                         |
# | `check`          | (1,)       | Whether opponent king is in check (0 or 1)                   |
# | `threat_target`  | (64,)      | Labels for newly threatened opponent pieces (-100 or 0 or 1) |
# | `terminal_flag`  | (1,)       | Game state (0=active, 1=stalemate, 2=checkmate)              |
# | `legal_moves`    | (64, L)    | Like move_target (=ground truth) but for all L legal moves   |
##################################################################################################
##################################################################################################


def result_to_eval(result_str):
    return {
        "1-0": 2,    # White win
        "1/2-1/2": 1,  # Draw
        "0-1": 0     # Black win
    }.get(result_str, 1)

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

def encode_board(board):
    """Returns (8, 8, 21) uint8 tensor"""
    plane = torch.zeros((8, 8, 21), dtype=torch.uint8)

    # 0–12: piece encodings
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        row, col = divmod(square, 8)
        plane[row][col][piece_to_index(piece)] = 1

    # Channel 13: side to move
    if board.turn == chess.WHITE:
        plane[:, :, 13] = 1

    # Channel 14: is my king in check (broadcast to all squares)
    if board.is_check():
        plane[:, :, 14] = 1

    # Channel 15: is this piece under threat (regardless of color)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            attackers = board.attackers(not piece.color, square)
            if attackers:
                plane[row][col][15] = 1

    # Channel 16: is this piece threatening opponent (regardless of color)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            targets = board.attacks(square)
            for target in targets:
                defender = board.piece_at(target)
                if defender and defender.color != piece.color:
                    plane[row][col][16] = 1
                    break

    # Channel 17: has legal move flag
    for move in board.legal_moves:
        from_row, from_col = divmod(move.from_square, 8)
        plane[from_row][from_col][17] = 1

    # Channel 18: White has castling rights
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        plane[:, :, 18] = 1

    # Channel 19: Black has castling rights
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        plane[:, :, 19] = 1

    # Channel 20: en passant target square
    ep_square = board.ep_square
    if ep_square is not None:
        row, col = divmod(ep_square, 8)
        plane[row][col][20] = 1

    return plane

def compute_threat_target(board):
    """Returns (64,) tensor with 1 if square is threatened by opponent, else 0. Uses -100 for empty squares."""
    threat_target = torch.full((64,), -100, dtype=torch.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(not piece.color, square)
            threat_target[square] = 1 if attackers else 0
    return threat_target

def compute_move_target(board, move):
    """Returns (64,) tensor with 0 for origin, piece index for destination, including castling/en passant."""
    move_target = torch.full((64,), -100, dtype=torch.int8)
    move_target[move.from_square] = 0

    # Handle en passant
    if board.is_en_passant(move):
        ep_row_offset = -1 if board.turn == chess.WHITE else 1
        captured_square = move.to_square + (8 * ep_row_offset)
        move_target[captured_square] = 0

    # Handle castling
    if board.is_castling(move):
        rook_from_to = {
            (chess.E1, chess.G1): (chess.H1, chess.F1),
            (chess.E1, chess.C1): (chess.A1, chess.D1),
            (chess.E8, chess.G8): (chess.H8, chess.F8),
            (chess.E8, chess.C8): (chess.A8, chess.D8),
        }
        rf, rt = rook_from_to.get((move.from_square, move.to_square), (None, None))
        if rf is not None and rt is not None:
            move_target[rf] = 0
            rook_piece = board.piece_at(rt)
            move_target[rt] = rook_piece.piece_type if rook_piece else 4  # fallback to rook

    # Push to observe destination piece
    board.push(move)
    piece = board.piece_at(move.to_square)
    if piece:
        move_target[move.to_square] = piece.piece_type
    board.pop()

    return move_target

def generate_legal_move_tensor(board: chess.Board):
    legal_move_tensors = []

    for move in board.legal_moves:
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
        return torch.empty((64, 0), dtype=torch.int8)

    return torch.stack(legal_move_tensors, dim=1)  # (64, L)

def parse_games(jsonl_path, output_path, max_games=None):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Parsing games"):
            obj = json.loads(line)
            pgn = obj["pgn"]
            result = obj["result"]
            eval_val = result_to_eval(result)

            game = chess.pgn.read_game(io.StringIO(pgn))
            board = game.board()

            for move in game.mainline_moves():
                board_tensor = encode_board(board)  # snapshot before move

                # Game termination flag
                terminal_flag = 2 if board.is_checkmate() else 1 if board.is_stalemate() else 0

                # Supervision targets
                legal_moves_tensor = generate_legal_move_tensor(board)
                move_target = compute_move_target(board, move)

                # Apply the move to evaluate result
                board.push(move)

                king_sq = board.king(not board.turn)
                is_check = board.is_check()
                threat_target = compute_threat_target(board)

                sample = {
                    "board": board_tensor,
                    "eval": torch.tensor([eval_val], dtype=torch.uint8),
                    "move_target": move_target,
                    "king_square": torch.tensor([king_sq], dtype=torch.uint8),
                    "check": torch.tensor([int(is_check)], dtype=torch.uint8),
                    "threat_target": threat_target,
                    "terminal_flag": torch.tensor([terminal_flag], dtype=torch.uint8),
                    "legal_moves": legal_moves_tensor,
                }
                data.append(sample)

            # Capture final board state after the last move
            board_tensor = encode_board(board)
            legal_moves_tensor = generate_legal_move_tensor(board)
            move_target = torch.full((64,), -100, dtype=torch.int8)
            terminal_flag = 2 if board.is_checkmate() else 1 if board.is_stalemate() else 0
            king_sq = board.king(not board.turn)
            is_check = board.is_check()
            threat_target = compute_threat_target(board)

            final_sample = {
                "board": board_tensor,
                "eval": torch.tensor([eval_val], dtype=torch.uint8),
                "move_target": move_target,
                "king_square": torch.tensor([king_sq], dtype=torch.uint8),
                "check": torch.tensor([int(is_check)], dtype=torch.uint8),
                "threat_target": threat_target,
                "terminal_flag": torch.tensor([terminal_flag], dtype=torch.uint8),
                "legal_moves": legal_moves_tensor,
            }
            data.append(final_sample)

            if max_games and len(data) >= max_games:
                break

    torch.save(data, output_path)
    print(f"Saved {len(data)} positions to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_games", type=int, default=None)
    args = parser.parse_args()

    parse_games(args.input, args.output, args.max_games)

    # # New script to parse filtered_games.jsonl and extract labeled tensors
    # filtered_games_path = 'filtered_games.jsonl'
    # output_tensor_path = 'positions.pt'
    
    # parse_games(filtered_games_path, output_tensor_path)
