import chess
import torch
from chessengine.model.prediction import batch_predict
from typing import List

class Simulator:
    def __init__(self, model, model2=None, model_is_white=True, device="cuda", GREEDY=False):
        self.model = model
        self.model2 = model2
        self.model_is_white = model_is_white
        self.device = device
        self.GREEDY = GREEDY    

    def get_model_for_turn(self, turn: bool):
        """
        Returns the appropriate model for the given turn.
        Args:
            turn (bool): True if it's white's turn, False if black's.
        Returns:
            model: The model assigned to play this turn.
        """
        if self.model2:
            if self.model_is_white:
                return self.model if turn else self.model2
            else:
                return self.model2 if turn else self.model
        else:
            return self.model

    def rollout(self, boards: List[chess.Board]) -> List[int]:
        """
        Simulates multiple games in parallel using the model's policy.
        Returns:
            List of result indices: 0 (black win), 1 (draw), 2 (white win)
        """
        results = [None] * len(boards)
        active_indices = [i for i in range(len(boards))]

        while active_indices:
            current_boards = [boards[i] for i in active_indices]

            turn = current_boards[0].turn  # assume all boards are on the same player's turn
            model = self.get_model_for_turn(turn)

            probs_batch, _, legal_moves_batch = batch_predict(model, current_boards, self.device)

            selected_moves = self.select_move(probs_batch, legal_moves_batch)
            for board_idx, move in zip(active_indices, selected_moves):
                board = boards[board_idx]
                board.push(move)
                if board.is_game_over():
                    results[board_idx] = self.result_to_index(board.result())

            active_indices = [i for i in active_indices if not boards[i].is_game_over()]

        return results
    
    def rollout_single(self, board: chess.Board) -> int:        
        return self.rollout_batch([board])[0]
    
    def select_move(self, probs_batch: torch.Tensor, legal_moves_batch: list[list[chess.Move]]) -> List[chess.Move]:
        """
        Selects moves stochastically from the model's policy distribution for a batch of boards.
        """
        if self.GREEDY:
            idx = probs_batch.argmax(dim=1).squeeze(-1)  # (B,)
        else:
            idx = torch.multinomial(probs_batch, 1).squeeze(-1)  # (B,)
        moves = [l[i] for l, i in zip(legal_moves_batch, idx)]
        return moves

    def result_to_index(self, result: str) -> int:
        """
        Converts chess.Board.result() string to scalar reward label {0, 1, 2}
        """
        if result == "1-0":
            return 2
        elif result == "0-1":
            return 0
        else:
            return 1
