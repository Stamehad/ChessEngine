import chess.pgn # type: ignore
import chess # type: ignore
import os
import io
import json
import torch
from preprocessing.position_parsing import encode_board
from pytorchchess import TorchBoard
from pytorchchess.utils import int_to_squares
from pytorchchess.utils.constants import LONG_RANGE_MOVES
from tqdm import tqdm


# ------------------------------------------------------------------
# 3-bit move-type table (bits 12-14)
# ------------------------------------------------------------------
PROMO_MAP = {
    chess.QUEEN: 1,
    chess.ROOK:  2,
    chess.BISHOP:3,
    chess.KNIGHT:4
}
TYPE_NORMAL      = 0
TYPE_EP_OR_2PUSH = 5
TYPE_OO          = 6   # kingside castle
TYPE_OOO         = 7   # queenside castle


def move_to_int(board: chess.Board, move: chess.Move) -> torch.Tensor:
    """
    Convert python-chess Move to 16-bit integer encoding.
    bit 0-5   : from_sq  (0-63)
    bit 6-11  : to_sq    (0-63)
    bit 12-14 : moveType (0-7)  per table above
    bit 15    : padding  (0 = valid)
    """
    from_sq = move.from_square
    to_sq   = move.to_square

    # --- determine 3-bit type ---------------------------------------
    if board.is_castling(move):
        move_type = TYPE_OO if chess.square_file(to_sq) == 6 else TYPE_OOO
    elif move.promotion:
        move_type = PROMO_MAP[move.promotion]
    elif board.is_en_passant(move) or (
        board.piece_at(from_sq).piece_type == chess.PAWN
        and abs(chess.square_rank(to_sq) - chess.square_rank(from_sq)) == 2):
        # en-passant capture OR double-pawn-push
        move_type = TYPE_EP_OR_2PUSH
    else:
        move_type = TYPE_NORMAL

    encoded = (from_sq & 0x3F) | ((to_sq & 0x3F) << 6) | (move_type << 12)
    return torch.tensor([encoded], dtype=torch.long)

def to_chess_move(encoded: int, board: chess.Board) -> chess.Move:
    from_sq, to_sq, promo_type = int_to_squares(encoded)
    promo_piece = None
    if promo_type in (1, 2, 3, 4):          # your promotion codes
        mapping = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}
        promo_piece = mapping[promo_type.item()]
    return chess.Move(from_sq, to_sq, promotion=promo_piece)

def compare_moves(board, moves, game_idx=None, j=None):
    # python-chess legal moves
    moves_py = {m.uci() for m in board.legal_moves}          # set of strings

    # your engine’s moves (encoded uint16)
    moves_cpu = moves.encoded.int()
    moves_my_raw = moves_cpu[0][moves.mask[0]]  # 1st board, flatten
    moves_my = {to_chess_move(m, board).uci() for m in moves_my_raw}
    #moves_my = {to_chess_move(int(m.item()), board).uci() for m in moves_my_raw}

    missing   = moves_py - moves_my     # moves python-chess has that you don’t
    extra     = moves_my - moves_py     # moves you generate that are illegal

    if missing or extra:
        print(f"Game {game_idx} board {j} - missing: {missing}, Extra: {extra}")
        return False
    else:
        return True

def check_moves_and_features(game, game_idx):
    all_good = True
    board = game.board()
    torch_boards = TorchBoard.from_board_list(board, device=torch.device('cpu'))
    
    for move_idx, move in enumerate(game.mainline_moves()):
        #print()
        #torch_boards.render()
        #-------------------------------------------------------
        # get legal moves
        #-------------------------------------------------------
        moves = torch_boards.get_legal_moves(get_tensor=True)
        #print(f"moves.encoded: {moves.encoded.shape}, moves.mask: {moves.mask.shape}")
        check = compare_moves(board, moves, game_idx, move_idx)
        all_good &= check

        #-------------------------------------------------------
        # get feature tensors
        #------------------------------------------------------
        feature_tensor = torch_boards.feature_tensor()
        feature_tensor2 = encode_board(board)
        check = feature_tensor[0] == feature_tensor2
        
        if not check.all():
            #torch_boards.render()
            print(f"Tensors are not equal for game {game_idx} board {move_idx}")
            for i in range(21):
                check_i = feature_tensor[0, :, :, i] == feature_tensor2[:, :, i]
                if not check_i.all():
                    print(f"Tensors are not equal at index {i}")
                    # print("torch board:")
                    # print(feature_tensor[0, :, :, i].flip(0))
                    # print()
                    # print("chess board:")
                    # print(feature_tensor2[:, :, i].flip(0))
                    # print()
            all_good = False
        
        # if game_idx == 0 and move_idx < 5:
        #     torch_boards.render()
        
        #-------------------------------------------------------
        # push moves
        #-------------------------------------------------------
        torch_move = move_to_int(board, move)
        b_idx = torch.zeros_like(torch_move).long()
        torch_boards = torch_boards.push(torch_move, b_idx)
        board.push(move)
    
    # if all_good:
    #     print(f"✔ all good!")

    return all_good


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    args = parser.parse_args()

    max_games = args.games

    pgn_file = "data/shards300_small/shard_0.pgn"
    check = True
    # read the first 10 games from the pgn file
    with open(pgn_file, 'r') as f0:
        for game_idx, line in tqdm(enumerate(f0), total=max_games, desc="Processing games"):
            obj = json.loads(line)
            pgn = obj["pgn"]

            # if game_idx < 570 or game_idx > 600:
            #     continue
            # print(f"Processing game {game_idx}...")
            game = chess.pgn.read_game(io.StringIO(pgn))
            
            #print(f"Checking game {game_idx}")
            result = check_moves_and_features(game, game_idx)
            if not result:
                print(f"Game {game_idx} failed!")
            check &= result

            if game_idx == max_games-1:
                print(f"Stopping after {max_games} games.")             
                break

    if check:
        print("✔ All checks passed!")