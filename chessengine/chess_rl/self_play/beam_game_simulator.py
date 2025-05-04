import chess
from typing import List, Optional, Tuple
import torch
from chessengine.chess_rl.simulator.simulator import Simulator
from chessengine.preprocessing.position_parsing import encode_board, generate_legal_move_tensor
from chessengine.model.utils import pad_and_stack
from chessengine.chess_rl.beam_search.beam import BEAM

class BeamGameSimulator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def play_games(self, start_fens=None, n_games=1) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Plays a batch of games using MCTS for self-play. Returns training tuples (X_t, π_t, z).
        """
        if start_fens is None:
            start_fens = [None] * n_games
        beam = BEAM(self.model, self.config['beam'])

        all_data = []
        boards = [chess.Board(fen=fen) if fen else chess.Board() for fen in start_fens]
        games = Games.from_boards(boards)  # List[Game]

        # Remove finished games and get their data
        finished = games.pop_finished()  # List[Game]
        all_data.extend(finished.extract_finished_data())
        del finished

        while games:
            # 1. Run MCTS 
            moves_list, _ = beam.run_beam_search(games.get_boards()) 
            moves_list = list(map(list, zip(*moves_list)))
            #print(f"Moves: {moves_list}")  # Debugging output

            # 2. Log data before making the move
            games.log_policies(moves_list[0])  # Log moves and policies

            # 3. Sample π and play moves
            for moves in moves_list:
                games.push(moves)  # Push moves to games

            # 4. Process finished games
            finished = games.pop_finished()  # List[Game]
            all_data.extend(finished.extract_finished_data()) 
            del finished
            
        return self._stack_data(all_data)
    
    def _stack_data(self, data):
        """
        Stack the data from multiple games into a single batch.
        """
        x_list, moves_list, true_idx_list, z_list = zip(*data) 
        x = torch.cat(x_list, dim=0) # (B, 8, 8, 21)
        lm = pad_and_stack(moves_list, BATCH_DIM=True, pad_value=-100) # (B, 64, L_max)
        true_idx = torch.cat(true_idx_list, dim=0) # (B,)
        z = torch.cat(z_list, dim=0) # (B,)
        return x, lm, true_idx, z
            
class Game:
    def __init__(self, idx: int, board: chess.Board):
        self.idx = idx
        self.board = board
        self.history = []  # List of tuples (x_t, legal_moves_tensor, true_idx)
        self.outcome = None  # int (0/1/2) or None
        self.is_active = True

        if board.is_game_over():
            self.outcome = self._get_outcome(board)
            self.is_active = False

    def log_policy(self, move: chess.Move): 
        x = self._get_positions() # (8, 8, 21)
        legal_moves, true_idx = self._get_move_tensor(move) # (64, L)
        self.history.append((x, legal_moves, true_idx)) # (x_t, legal_moves_tensor, true_idx)

    def push(self, move: chess.Move):
        if move is not None:
            self.board.push(move)
            if self.board.is_game_over():
                self.outcome = self._get_outcome(self.board)
                self.is_active = False

    def is_done(self) -> bool:
        return self.board.is_game_over()

    def finish(self, z: int):
        self.outcome = z

    def get_history(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Add terminal state to history
        self.log_policy(None) 

        # Stack history
        x_list, legal_moves_list, true_idx_list = zip(*self.history)
        x = torch.stack(x_list)  # (T, 8, 8, 21)
        lm = pad_and_stack(legal_moves_list, pad_value=-100) # (T, 64, L_max)
        true_idx = torch.tensor(true_idx_list).long()  # (T,)
        z = torch.tensor(self.outcome, dtype=torch.int8).expand(x.shape[0])  # (T,)        
        return x, lm, true_idx, z

    def _get_outcome(self, board: chess.Board) -> int:
        outcome = board.outcome()
        if outcome.winner is None:
            return 1
        elif outcome.winner == chess.WHITE:
            return 2
        else:
            return 0
        
    def _get_positions(self) -> torch.Tensor:
        return encode_board(self.board) # (8, 8, 21) 
    
    def _get_move_tensor(self, move) -> torch.Tensor:
        lm_tensor, true_idx = generate_legal_move_tensor(self.board, ground_truth_move=move)
        return lm_tensor, true_idx # (64, L), float


# Batch-wise wrapper for multiple games
class Games:
    def __init__(self, games: List[Game]):
        self.games = games

    @classmethod
    def from_boards(cls, boards: List[chess.Board]) -> 'Games':
        return cls([Game(idx=i, board=b) for i, b in enumerate(boards)])

    def __len__(self):
        return len(self.games)

    def get_boards(self) -> List[chess.Board]:
        return [g.board for g in self.games]

    def log_policies(self, moves):
        assert len(moves) == len(self.games), "Number of moves must match number of games."
        for game, move in zip(self.games, moves):
            game.log_policy(move)

    def push(self, moves: List[chess.Move]):
        for game, move in zip(self.games, moves):
            game.push(move)

    def pop_finished(self) -> 'Games':
        finished = [g for g in self.games if not g.is_active]
        self.games = [g for g in self.games if g.is_active]
        return Games(finished)

    def extract_finished_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        return [g.get_history() for g in self.games if not g.is_active]
