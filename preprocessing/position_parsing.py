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
# | `move_weight`    | (1,)       | Optional per-sample weighting for move prediction loss       |
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
    """Returns (8, 8, 20) uint8 tensor"""
    plane = torch.zeros((8, 8, 20), dtype=torch.uint8)

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

    return plane

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
                
                # Check terminal flag
                if board.is_checkmate():
                    terminal_flag = 2
                elif board.is_stalemate():
                    terminal_flag = 1
                else:
                    terminal_flag = 0

                board.push(move)

                piece = board.piece_at(move.to_square)  # after push
                move_target = torch.full((64,), -100, dtype=torch.int8)
                move_target[move.from_square] = 0
                if piece:
                    move_target[move.to_square] = piece.piece_type  # 1–6

                # King square and check
                king_sq = board.king(not board.turn)  # opponent king square
                is_check = board.is_check()

                # Compute threat_target (with -100 masking)
                threat_target = torch.full((64,), -100, dtype=torch.int8)
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        attackers = board.attackers(not piece.color, square)
                        threat_target[square] = 1 if attackers else 0

                sample = {
                    "board": board_tensor,
                    "eval": torch.tensor([eval_val], dtype=torch.uint8),
                    "move_target": move_target,
                    "king_square": torch.tensor([king_sq], dtype=torch.uint8),
                    "check": torch.tensor([int(is_check)], dtype=torch.uint8),
                    "threat_target": threat_target,
                    "terminal_flag": torch.tensor([terminal_flag], dtype=torch.uint8),
                }
                data.append(sample)

            # Capture final board state after the last move
            board_tensor = encode_board(board)
            move_target = torch.full((64,), -100, dtype=torch.int8)
            terminal_flag = 2 if board.is_checkmate() else 1 if board.is_stalemate() else 0
            king_sq = board.king(not board.turn)
            is_check = board.is_check()
            threat_target = torch.full((64,), -100, dtype=torch.int8)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    attackers = board.attackers(not piece.color, square)
                    threat_target[square] = 1 if attackers else 0

            final_sample = {
                "board": board_tensor,
                "eval": torch.tensor([eval_val], dtype=torch.uint8),
                "move_target": move_target,
                "king_square": torch.tensor([king_sq], dtype=torch.uint8),
                "check": torch.tensor([int(is_check)], dtype=torch.uint8),
                "threat_target": threat_target,
                "terminal_flag": torch.tensor([terminal_flag], dtype=torch.uint8),
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
