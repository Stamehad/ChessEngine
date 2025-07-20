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
from typing import List, Tuple, Dict


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


def compare_moves_parallel(boards: List[chess.Board], moves, board_indices: List[int]) -> List[bool]:
    """
    Compare moves for multiple boards in parallel.
    
    Args:
        boards: List of chess.Board objects
        moves: TorchBoard legal moves (with batch dimension)
        board_indices: Which boards in the batch to check
    
    Returns:
        List of booleans indicating if each board's moves are correct
    """
    results = []
    
    for i, board_idx in enumerate(board_indices):
        board = boards[i]
        
        # python-chess legal moves
        moves_py = {m.uci() for m in board.legal_moves}
        
        # your engine's moves for this specific board
        moves_cpu = moves.encoded.int()
        moves_my_raw = moves_cpu[board_idx][moves.mask[board_idx]]  # specific board, flatten
        moves_my = {to_chess_move(m, board).uci() for m in moves_my_raw}
        
        missing = moves_py - moves_my     # moves python-chess has that you don't
        extra = moves_my - moves_py       # moves you generate that are illegal
        
        if missing or extra:
            print(f"Board {board_idx} - missing: {missing}, Extra: {extra}")
            results.append(False)
        else:
            results.append(True)
    
    return results


def compare_features_parallel(boards: List[chess.Board], feature_tensors: torch.Tensor, board_indices: List[int]) -> List[bool]:
    """
    Compare feature tensors for multiple boards in parallel.
    
    Args:
        boards: List of chess.Board objects  
        feature_tensors: Batched feature tensors from TorchBoard (B, 8, 8, 21)
        board_indices: Which boards in the batch to check
    
    Returns:
        List of booleans indicating if each board's features are correct
    """
    results = []
    
    for i, board_idx in enumerate(board_indices):
        board = boards[i]
        
        # Get expected features from preprocessing
        expected_features = encode_board(board)
        
        # Get actual features from TorchBoard
        actual_features = feature_tensors[board_idx]
        
        # Compare
        check = actual_features == expected_features
        
        if not check.all():
            print(f"Feature mismatch for board {board_idx}")
            for layer in range(21):
                layer_check = actual_features[:, :, layer] == expected_features[:, :, layer]
                if not layer_check.all():
                    print(f"  Layer {layer} differs")
            results.append(False)
        else:
            results.append(True)
    
    return results


class ParallelGameProcessor:
    def __init__(self, games_batch: List[chess.pgn.Game], device: str = 'cpu', verbose: bool = False):
        self.VERBOSE = verbose
        self.games_batch = games_batch
        self.device = device
        self.B = len(games_batch)
        
        # Initialize boards and move sequences
        self.boards = [game.board() for game in games_batch]
        self.move_sequences = [list(game.mainline_moves()) for game in games_batch]
        self.max_moves = max(len(seq) for seq in self.move_sequences)
        
        # Track which games are still active
        self.active_games = list(range(self.B))
        self.current_move_idx = 0
        
        # Initialize TorchBoard with all starting positions
        self.torch_boards = TorchBoard.from_board_list(self.boards, device=device)
    
    def get_active_boards(self) -> Tuple[List[chess.Board], List[int]]:
        """Get currently active boards and their indices."""
        active_boards = [self.boards[i] for i in self.active_games]
        return active_boards, self.active_games
    
    def check_current_position(self) -> bool:
        """Check legal moves and features for current position."""
        if not self.active_games:
            return True
        
        active_boards, original_indices = self.get_active_boards()
        idx = torch.tensor(original_indices, dtype=torch.long, device=self.device)
        
        # Create a simple batch index list
        batch_indices = list(range(len(active_boards)))
        tb = self.torch_boards[idx]
        
        # Check legal moves
        moves = tb.get_legal_moves(get_tensor=True)
        moves_correct = self.compare_moves_parallel_v2(active_boards, moves, batch_indices, original_indices)
        
        # Check feature tensors
        feature_tensors = tb.feature_tensor()
        features_correct = compare_features_parallel_v2(active_boards, feature_tensors, batch_indices, original_indices)
        
        # Combine results
        all_correct = all(moves_correct) and all(features_correct)
        
        if not all_correct:
            print(f"Errors found at move {self.current_move_idx}")
        
        return all_correct
    
    def advance_one_move(self) -> bool:
        """Advance all active games by one move."""
        if not self.active_games or self.current_move_idx >= self.max_moves:
            return False
        
        # Collect moves for active games
        moves_to_apply = []
        new_active_games = []
        
        for game_idx in self.active_games:
            if self.current_move_idx < len(self.move_sequences[game_idx]):
                move = self.move_sequences[game_idx][self.current_move_idx]
                torch_move = move_to_int(self.boards[game_idx], move)
                moves_to_apply.append(torch_move)
                new_active_games.append(game_idx)
                
                # Apply move to python-chess board
                self.boards[game_idx].push(move)
        
        if not moves_to_apply:
            return False
        
        # Apply moves to TorchBoard in batch
        if len(moves_to_apply) > 0:
            batch_moves = torch.cat(moves_to_apply, dim=0)
            b_idx = torch.tensor(new_active_games, dtype=torch.long, device=self.device)
            padded_moves = torch.full((self.B,), fill_value=-1, dtype=torch.long, device=self.device)  
            padded_moves[b_idx] = batch_moves
            # CRITICAL FIX: Use batch indices, not original game indices
            # The current active games correspond to batch positions [0, 1, 2, ...]
            batch_board_indices = torch.arange(self.B, dtype=torch.long, device=self.device)
            if self.VERBOSE:
                print(f"len(torch_board) = {len(self.torch_boards)}")
                print(f"batch_board_indices: {batch_board_indices}")
                print(f"b_idx: {b_idx}")
                print(f"batch_moves: {batch_moves}")

            
            self.torch_boards = self.torch_boards.push(padded_moves, batch_board_indices)
        
        # Update active games
        self.active_games = new_active_games
        self.current_move_idx += 1
        
        return True
    
    def run_full_test(self) -> bool:
        """Run the complete test for all games."""
        all_correct = True
        
        # Check initial position
        all_correct &= self.check_current_position()
        
        # Process all moves
        while self.advance_one_move():
            position_correct = self.check_current_position()
            all_correct &= position_correct
            
            if not position_correct:
                print(f"Error at move {self.current_move_idx}")
        
        return all_correct


    def compare_moves_parallel_v2(self, boards: List[chess.Board], moves, batch_indices: List[int], original_indices: List[int]) -> List[bool]:
        """Version with explicit separation of batch vs original indices."""
        results = []
        
        for batch_idx, original_idx, board in zip(batch_indices, original_indices, boards):
            # python-chess legal moves
            moves_py = {m.uci() for m in board.legal_moves}
            
            # your engine's moves
            moves_cpu = moves.encoded.int()
            moves_my_raw = moves_cpu[batch_idx][moves.mask[batch_idx]]
            moves_my = {to_chess_move(m, board).uci() for m in moves_my_raw}
            
            missing = moves_py - moves_my
            extra = moves_my - moves_py
            
            if missing or extra:
                print(f"Game {original_idx} b_idx {batch_idx} - missing: {missing}, Extra: {extra}")
                tb = self.torch_boards[original_idx:original_idx + 1]  # Get TorchBoard for this game
                from pytorchchess.torch_board.board import BoardCache
                print(tb.board_tensor.flip(1))  # flip to match python-chess orientation
                tb.cache = BoardCache()
                tb.cache.check_info = tb.compute_check_info()
                print(tb)
                print(tb.state)
                lm = tb.get_legal_moves()
                print(lm.moves_to_standard_format())
                tb.render()
                results.append(False)
            else:
                results.append(True)
        
        return results

def compare_features_parallel_v2(boards: List[chess.Board], feature_tensors: torch.Tensor, batch_indices: List[int], original_indices: List[int]) -> List[bool]:
    """Version with explicit separation of batch vs original indices."""
    results = []
    
    for batch_idx, original_idx, board in zip(batch_indices, original_indices, boards):
        # Get expected features from preprocessing
        expected_features = encode_board(board)
        
        # Get actual features from TorchBoard
        actual_features = feature_tensors[batch_idx]
        
        # Compare
        check = actual_features == expected_features
        
        if not check.all():
            print(f"Feature mismatch for game {original_idx}")
            for layer in range(21):
                layer_check = actual_features[:, :, layer] == expected_features[:, :, layer]
                if not layer_check.all():
                    print(f"  Layer {layer} differs")
            results.append(False)
        else:
            results.append(True)
    
    return results

def load_games_batch(pgn_file: str, start_idx: int, batch_size: int) -> List[chess.pgn.Game]:
    """Load a batch of games from PGN file."""
    games = []
    
    with open(pgn_file, 'r') as f:
        for game_idx, line in enumerate(f):
            if game_idx < start_idx:
                continue
            if len(games) >= batch_size:
                break
                
            obj = json.loads(line)
            pgn = obj["pgn"]
            game = chess.pgn.read_game(io.StringIO(pgn))
            games.append(game)
    
    return games


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100, help="Total number of games to test")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of games to process in parallel (G)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    pgn_file = "data/shards300_small/shard_0.pgn"
    total_games = args.games
    device = torch.device(args.device)
    B = args.batch_size  # Games processed in parallel
    
    all_tests_passed = True
    games_processed = 0
    
    print(f"Testing {total_games} games in batches of {B} on {device}")
    print(f"PGN file: {pgn_file}")
    
    # Process games in batches
    with tqdm(total=total_games, desc="Processing games") as pbar:
        for start_idx in range(0, total_games, B):
            batch_size = min(B, total_games - start_idx)
            # if start_idx < 550 or start_idx >= 600:
            #     continue
            
            # Load batch of games
            games_batch = load_games_batch(pgn_file, start_idx, batch_size)
            
            if not games_batch:
                break
            
            # Process this batch in parallel
            processor = ParallelGameProcessor(games_batch, device=device)
            batch_result = processor.run_full_test()
            
            if not batch_result:
                print(f"Batch starting at game {start_idx} failed!")
                all_tests_passed = False
            
            games_processed += len(games_batch)
            pbar.update(len(games_batch))
            
            # Memory cleanup
            del processor, games_batch
            if args.device == "cuda":
                torch.cuda.empty_cache()
    
    print(f"\nProcessed {games_processed} games total")
    if all_tests_passed:
        print("✔ All parallel tests passed!")
    else:
        print("❌ Some tests failed!")