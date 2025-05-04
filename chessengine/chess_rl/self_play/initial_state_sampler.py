import numpy as np
import chess
from chessengine.chess_rl.beam_search.beam_tree import BeamTree
from chessengine.chess_rl.self_play.initial_states import STATES1, STATES2
import json
import random
import glob
import os
import chess.pgn
from io import StringIO

class State:
    def __init__(self, s):
        self.board_fen = s[1]
        self.p = s[0]
        
class InitialStates:
    def __init__(self, STATES):
        self.states = []
        for s in STATES:
            self.states.append(State(s))
    
    def sample(self, n=1, temp=1.0):
        """
        Sample without replacement n states using probabilities from the list of states.
        """
        if n > len(self.states):
            raise ValueError("Requested number of samples exceeds available states.")
        
        ps = [state.p for state in self.states]
        ps = np.array(ps)
        ps = np.power(ps, 1.0 / temp)  # Apply temperature scaling
        ps /= ps.sum()  # Normalize probabilities
        
        indices = np.random.choice(len(self.states), size=n, replace=False, p=ps)
        sampled_states = [self.states[i] for i in indices]
        return [chess.Board(state.board_fen) for state in sampled_states]
    
    @classmethod
    def recompute_probs(self, model, device='cuda'):
        board = chess.Board()
        tree = BeamTree(model, device=device, topk_schedule={0: 40, 1: 40}, clear_boards=False)
        tree.setup(board)
        tree.expand_to_depth()
        layer1 = tree.layers[1]
        layer2 = tree.layers[2]
        STATES1 = [(pos.prob_from_root(), pos.board.fen()) for pos in layer1.positions]
        STATES2 = [(pos.prob_from_root(), pos.board.fen()) for pos in layer2.positions]

        return InitialStates(STATES1), InitialStates(STATES2)
    
class DataBaseSampler:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def sample_from_database(self, n_games=10, positions_per_game=1, max_ply=60):
        """
        Sample random positions from PGN game files efficiently using indexed sampling.

        Assumes each line in the file is a JSON object with key 'pgn' containing the game PGN.
        """
        def count_lines(file_path):
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)

        boards = []
        files = glob.glob(os.path.join(self.data_dir, "*.pgn"))
        random.shuffle(files)

        for fname in files:
            total_lines = count_lines(fname)
            if total_lines == 0:
                continue

            line_indices = sorted(random.sample(range(total_lines), min(n_games, total_lines)))
            current_index = 0
            collected = 0

            with open(fname, 'r') as f:
                for line in f:
                    if collected >= len(line_indices):
                        break
                    if current_index == line_indices[collected]:
                        try:
                            pgn_dict = json.loads(line)
                            pgn_string = pgn_dict.get("pgn", "")
                            game = chess.pgn.read_game(StringIO(pgn_string))
                            if game is None:
                                continue

                            node = game
                            plies = 0
                            ply_boards = []

                            while node.variations and plies < max_ply:
                                node = node.variations[0]
                                plies += 1
                                ply_boards.append(node.board().copy())

                            if ply_boards:
                                boards.extend(random.sample(ply_boards, min(len(ply_boards), positions_per_game)))
                            collected += 1
                        except Exception:
                            pass
                    current_index += 1

            if len(boards) >= n_games * positions_per_game:
                break

        return boards
    
    def pre_sample(self, prefetch=5, n_games=10, positions_per_game=1, max_ply=60):
        """
        Pre-sample random positions from PGN game files efficiently using indexed sampling.
        """       
        return self.sample_from_database(
            n_games=n_games * prefetch, 
            positions_per_game=positions_per_game, 
            max_ply=max_ply
        )

class InitialStateSampler:
    """
    Samples opening board positions for self-play training.

    Provides a mix of:
    - Standard starting board
    - n1 boards from STATES1 (after 1 move)
    - n2 boards from STATES2 (after 2 moves)
    """

    def __init__(self, config, states1=STATES1, states2=STATES2):
        self.level1 = InitialStates(states1)
        self.level2 = InitialStates(states2)
        self.database_sampler = DataBaseSampler(data_dir="data/shards300_small/")
        self.database_boards = self.prefetch_from_database()
        self.config = config
        self.prefecth = config.get('prefetch', 10)
        self.n_games = config.get('n_games', 30)
        self.positions_per_game = config.get('positions_per_game', 1)
        self.max_ply = config.get('max_ply', 60)

    def get_boards(self):
        boards1 = self.sample_initial_postions()
        try:
            boards2 = next(self.database_boards)            
        except StopIteration:
            self.database_boards = self.prefetch_from_database()
            boards2 = next(self.database_boards)

        boards = boards1 + boards2
        return boards
        
    def prefetch_from_database(self):
        boards = self.database_sampler.pre_sample(
            prefetch=self.prefecth,
            n_games=self.n_games,
            positions_per_game=self.positions_per_game,
            max_ply=self.max_ply
        )

        for i in range(0, len(boards), self.n_games):
            yield boards[i:i + self.n_games]

    def sample_initial_postions(self, n1=10, n2=10, include_start=True, temp=1.0):
        boards = []
        if include_start:
            boards.append(chess.Board())  # Standard start position
        boards.extend(self.level1.sample(n=n1, temp=temp))
        boards.extend(self.level2.sample(n=n2, temp=temp))
        return boards
    
    def recompute_probs_of_initial_states(self, model, device='cuda'):
        self.level1, self.level2 = InitialStates.recompute_probs(model, device=device)
    
    