import chess
from typing import List, Optional, Tuple
import torch
from chessengine.chess_rl.simulator.simulator import Simulator
from chessengine.preprocessing.position_parsing import encode_board, generate_legal_move_tensor
from chessengine.model.utils import batch_legal_moves
from chessengine.chess_rl.mcts.mcts import MCTS, BATCHED_MCTS

def cat_pi_list(pi_list) -> torch.Tensor:
        """
        Arg:
            pi_list: list of tensors of shape (T_i, L_i)
        Return: 
            a padded tensor of shape (T, L_max) where T = sum(T_i) and L_max = max(L_i)
        """
        L_max = max(pi.shape[-1] for pi in pi_list)
        padded = []
        for pi in pi_list:
            T, L = pi.shape[0], pi.shape[-1]
            if L < L_max:
                pad = torch.zeros(T, L_max - L, dtype=pi.dtype)
                pi = torch.cat([pi, pad], dim=-1) # (T, L_max)
            padded.append(pi)
        
        return torch.cat(padded, dim=0)  # (T_max, L_max)

class GameSimulator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        #self.simulator = Simulator(model, device=config.get("device", "cuda"))
    
    def play_games(self, start_fens: List[Optional[str]]) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Plays a batch of games using MCTS for self-play. Returns training tuples (X_t, π_t, z).
        """
        mcts = BATCHED_MCTS(self.model, self.config)

        all_data = []
        boards = [chess.Board(fen=fen) if fen else chess.Board() for fen in start_fens]
        games = Games.from_boards(boards)  # List[Game]

        # Remove finished games and get their data
        finished = games.pop_finished()  # List[Game]
        all_data.extend(finished.extract_finished_data())
        del finished

        while games:
            # 1. Run MCTS 
            pis = mcts.run_mcts_search(games.get_boards()) # [(legal_moves, pi), ...]

            # 2. Log data before making the move
            games.log_policies(pis)

            # 3. Sample π and play moves
            move_idx = [(moves, torch.multinomial(pi, 1).item()) for moves, pi in pis]
            moves = [moves[idx] for moves, idx in move_idx]  
            games.push(moves)  # Push moves to games

            # 4. Process finished games
            finished = games.pop_finished()  # List[Game]
            all_data.extend(finished.extract_finished_data())  # List[Tuple[X, Pi, z]]
            del finished
            
        return self._stack_data(all_data)
    
    def _stack_data(self, data):
        """
        Stack the data from multiple games into a single batch.
        """
        x_list, pi_list, moves_list, z_list = zip(*data) 
        x = torch.cat(x_list, dim=0) # (B, 8, 8, 21)
        pi = cat_pi_list(pi_list) # (B, L_max)
        lm = batch_legal_moves(moves_list, NO_BATCH_DIM=False) # (B, 64, L_max)
        z = torch.cat(z_list, dim=0) # (B,)
        return x, pi, lm, z
            
class Game:
    def __init__(self, idx: int, board: chess.Board):
        self.idx = idx
        self.board = board
        self.history = []  # List[Tuple[x_t, π_t]]
        self.outcome = None  # int (0/1/2) or None
        self.is_active = True

        if board.is_game_over():
            self.outcome = self._get_outcome(board)
            self.is_active = False

    def log_policy(self, pi: torch.Tensor, moves: List[chess.Move]): # pi: Tensor (L,)
        # Sanity check
        assert len(pi) == len(moves), f"Mismatch: pi has shape {pi.shape}, but {len(moves)} moves given"
        
        x = self._get_positions() # (8, 8, 21)
        legal_moves = self._get_move_tensor(moves) # (64, L)
        self.history.append((x, pi, legal_moves)) # (x_t, π_t, legal_moves_tensor)

    def push(self, move: chess.Move):
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
        self.log_policy(torch.zeros(0), []) 

        # Stack history
        x_list, pi_list, legal_moves_list = zip(*self.history)
        x = torch.stack(x_list)  # (T, 8, 8, 21)
        pi = self._pad_pis(pi_list) # (T, L_max)
        lm = batch_legal_moves(legal_moves_list) # (T, 64, L_max)
        z = torch.tensor(self.outcome, dtype=torch.long).expand(x.shape[0])  # (T,)        
        return x, pi, lm, z

    def _pad_pis(self, pi_list: list) -> torch.Tensor:
        max_len = max(pi.shape[0] for pi in pi_list)
        padded = [torch.cat([pi, torch.zeros(max_len - pi.shape[0])]) for pi in pi_list]
        return torch.stack(padded)
    
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
    
    def _get_move_tensor(self, moves) -> torch.Tensor:
        lm_tensor, _ = generate_legal_move_tensor(self.board, ground_truth_move=None, legal_moves=moves) # (64, L)
        return lm_tensor # (64, L)


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

    def log_policies(self, pis):
        for game, (pi, moves) in zip(self.games, pis):
            game.log_policy(pi, moves)

    def push(self, moves: List[chess.Move]):
        for game, move in zip(self.games, moves):
            game.push(move)

    def pop_finished(self) -> 'Games':
        finished = [g for g in self.games if not g.is_active]
        self.games = [g for g in self.games if g.is_active]
        return Games(finished)

    def extract_finished_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        return [g.get_history() for g in self.games if not g.is_active]
