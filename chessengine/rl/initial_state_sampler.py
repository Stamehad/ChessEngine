import glob
import json
import os
import random
from io import StringIO
from typing import Iterable, List, Sequence, Tuple

import chess
import chess.pgn
import torch

from pytorchchess import TorchBoard
from pytorchchess.utils.utils import to_chess_move

from rl.initial_states import STATES1, STATES2


class State:
    def __init__(self, data: Tuple[float, str]):
        self.p = data[0]
        self.board_fen = data[1]


class InitialStates:
    def __init__(self, states: Sequence[Tuple[float, str]]):
        self.states = [State(s) for s in states]

    def sample(self, n: int = 1, temp: float = 1.0) -> List[chess.Board]:
        if n > len(self.states):
            raise ValueError("Requested number of samples exceeds available states.")

        ps = torch.tensor([state.p for state in self.states], dtype=torch.float64)
        ps = torch.pow(ps, 1.0 / temp)
        ps /= ps.sum()
        idx = torch.multinomial(ps, n, replacement=False).tolist()
        return [chess.Board(self.states[i].board_fen) for i in idx]

    @classmethod
    def recompute_probs(
        cls,
        model,
        device: torch.device = torch.device("cuda"),
        topk_root: int = 40,
        topk_child: int = 40,
    ) -> Tuple["InitialStates", "InitialStates"]:
        root_board = chess.Board()
        level1 = cls._expand_position(root_board, model, device, topk_root)

        level2: List[Tuple[float, str]] = []
        for prob, fen in level1:
            board = chess.Board(fen)
            children = cls._expand_position(board, model, device, topk_child)
            for child_prob, child_fen in children:
                level2.append((prob * child_prob, child_fen))

        level1 = cls._normalize(level1)
        level2 = cls._normalize(level2)
        return cls(level1), cls(level2)

    @staticmethod
    def _normalize(states: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        if not states:
            return []
        total = sum(p for p, _ in states)
        if total == 0:
            total = 1.0
        return [(p / total, fen) for p, fen in states]

    @staticmethod
    def _expand_position(
        board: chess.Board,
        model,
        device: torch.device,
        topk: int,
    ) -> List[Tuple[float, str]]:
        tb = TorchBoard.from_board_list(board, device=device)
        legal_moves, features = tb.get_legal_moves_fused(return_features=True)
        mask = legal_moves.mask[0]
        if not mask.any():
            return []

        features = features.float()
        with torch.no_grad():
            _, move_pred = model(features)
        logits = legal_moves.get_logits(move_pred)
        probs = torch.softmax(logits, dim=-1)[0]
        encoded = legal_moves.encoded[0][mask]
        probs = probs[mask]
        if probs.numel() == 0:
            return []

        k = min(topk, probs.numel())
        top_probs, order = probs.topk(k)
        moves = encoded[order]
        top_probs = top_probs / top_probs.sum()

        states: List[Tuple[float, str]] = []
        for prob, move_value in zip(top_probs.tolist(), moves.tolist()):
            move = to_chess_move(torch.tensor(move_value))
            next_board = board.copy()
            next_board.push(move)
            states.append((prob, next_board.fen()))
        return states


class DataBaseSampler:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def sample_from_database(
        self,
        n_games: int = 10,
        positions_per_game: int = 1,
        max_ply: int = 60,
    ) -> List[chess.Board]:
        def count_lines(path: str) -> int:
            with open(path, "r") as fh:
                return sum(1 for _ in fh)

        boards: List[chess.Board] = []
        files = glob.glob(os.path.join(self.data_dir, "*.pgn"))
        random.shuffle(files)

        for fname in files:
            total_lines = count_lines(fname)
            if total_lines == 0:
                continue

            line_indices = sorted(random.sample(range(total_lines), min(n_games, total_lines)))
            current_index = 0
            collected = 0

            with open(fname, "r") as fh:
                for line in fh:
                    if collected >= len(line_indices):
                        break
                    if current_index == line_indices[collected]:
                        try:
                            pgn_dict = json.loads(line)
                            game = chess.pgn.read_game(StringIO(pgn_dict.get("pgn", "")))
                            if game is None:
                                continue

                            node = game
                            plies = 0
                            ply_boards: List[chess.Board] = []
                            while node.variations and plies < max_ply:
                                node = node.variations[0]
                                plies += 1
                                ply_boards.append(node.board().copy())

                            if ply_boards:
                                sample_count = min(len(ply_boards), positions_per_game)
                                boards.extend(random.sample(ply_boards, sample_count))
                            collected += 1
                        except Exception:
                            pass
                    current_index += 1

            if len(boards) >= n_games * positions_per_game:
                break

        return boards

    def pre_sample(
        self,
        prefetch: int = 5,
        n_games: int = 10,
        positions_per_game: int = 1,
        max_ply: int = 60,
    ) -> List[chess.Board]:
        return self.sample_from_database(
            n_games=n_games * prefetch,
            positions_per_game=positions_per_game,
            max_ply=max_ply,
        )


class InitialStateSampler:
    def __init__(
        self,
        config: dict,
        states1: Sequence[Tuple[float, str]] = STATES1,
        states2: Sequence[Tuple[float, str]] = STATES2,
    ):
        self.level1 = InitialStates(states1)
        self.level2 = InitialStates(states2)
        self.config = config

        self.database_sampler = DataBaseSampler(config.get("database_dir", "data/shards300_small/"))
        self.prefetch = config.get("prefetch", 10)
        self.n_games = config.get("n_games", 30)
        self.positions_per_game = config.get("positions_per_game", 1)
        self.max_ply = config.get("max_ply", 60)
        self.database_boards = self._prefetch_from_database()

    def get_boards(self) -> List[chess.Board]:
        boards = self.sample_initial_positions()
        try:
            boards.extend(next(self.database_boards))
        except StopIteration:
            self.database_boards = self._prefetch_from_database()
            boards.extend(next(self.database_boards))
        return boards

    def _prefetch_from_database(self) -> Iterable[List[chess.Board]]:
        boards = self.database_sampler.pre_sample(
            prefetch=self.prefetch,
            n_games=self.n_games,
            positions_per_game=self.positions_per_game,
            max_ply=self.max_ply,
        )

        for idx in range(0, len(boards), self.n_games):
            yield boards[idx : idx + self.n_games]

    def sample_initial_positions(
        self, n1: int = 10, n2: int = 10, include_start: bool = True, temp: float = 1.0
    ) -> List[chess.Board]:
        boards: List[chess.Board] = []
        if include_start:
            boards.append(chess.Board())
        boards.extend(self.level1.sample(n=n1, temp=temp))
        boards.extend(self.level2.sample(n=n2, temp=temp))
        return boards

    def recompute_probs_of_initial_states(
        self, model, device: torch.device = torch.device("cuda")
    ) -> None:
        self.level1, self.level2 = InitialStates.recompute_probs(model, device=device)
