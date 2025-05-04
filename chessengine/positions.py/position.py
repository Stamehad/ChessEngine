import chess
import torch
import numpy as np
from typing import List, Optional
from chessengine.preprocessing.position_parsing import encode_board

class UCB:
    def __init__(self, prior_p: float):
        self.n_visits = 0                       # N(s,a) for this position
        self.q_value = 0.0                      # Q(s,a): mean evaluation
        self.w_value = 0.0                      # W(s,a): total evaluation
        self.prior_p = prior_p                  # P(s,a): prior probability from policy head

    def score(self, parent_visits: int, cpuct: float) -> float:
        """Calculate the UCB score."""
        u = cpuct * self.prior_p * (torch.sqrt(torch.tensor(parent_visits)) / (1 + self.n_visits))
        return self.q_value + u.item()

    def update(self, value: float):
        """Update Q, W, N statistics."""
        self.n_visits += 1
        self.w_value += value
        self.q_value = self.w_value / self.n_visits

class Position:
    def __init__(self, board: chess.Board, parent: Optional['Position'] = None, move: Optional[chess.Move] = None):
        self.board = board.copy()               # chess.Board
        self.color = board.turn                 # True = white, False = black
        self.is_terminal = board.is_game_over() # Terminal position flag
        self.parent = parent                    # Parent Position object. None if root
        self.move = move                        # chess.Move leading to this position. None if root
        
        # Search / training information
        self.children = []                      # List[Position] (expanded children)

        # Prediction information
        self.eval = None                        # Tensor (3,) eval (black win, draw, white win)
        self.moves = None                       # List[chess.Move] (either all moves or top-k moves depending on context)
        self.move_probs = None                  # Tensor (L,) move probabilities
        self.best_child = None                  # Position: best child selected from expansion

        # MCTS-specific fields
        self.ucb = None

    # --- Universal methods ---

    def clear_board(self):
        """Free memory by removing board and predicted moves."""
        self.board = None
        self.moves = None

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0
    
    def get_tensor(self):
        """Encode the board state."""
        return encode_board(self.board) # (8, 8, 21) 
    
    def get_prediction(self, moves: List[chess.Move], eval: torch.Tensor, probs: torch.Tensor = None):
        """Assign top-k moves and evaluation."""
        self.moves = moves
        self.eval = eval
        if probs is None:
            probs = torch.zeros(len(moves), dtype=torch.float32, device=eval.device)
        else:
            self.move_probs = probs

    # --- Methods for Tree Search ---

    def expand(self, USING_MCTS) -> List['Position']:
        """Expand using top-k moves (Beam search expansion)."""
        if self.is_terminal:
            return []
        if self.moves is None:
            raise ValueError("Cannot expand without assigned moves.")

        for move, p in zip(self.moves, self.move_probs):
            next_board = self.board.copy()
            next_board.push(move)
            child = Position(next_board, parent=self, move=move)
            if USING_MCTS:
                child.initialize_ucb(prior_p=p)
            self.children.append(child)

        # Prioritize checkmate moves (keep only if decisive)
        decisive = [child for child in self.children if child.board.is_checkmate()]
        if decisive:
            self.children = decisive[:1]  # Keep only one winning move

        return self.children

    def backpropagate_from_children(self):
        """Backpropagate evaluation from expanded children (Beam search)."""
        if not self.children:
            return
        children_with_eval = [c for c in self.children if c.eval is not None]
        if not children_with_eval:
            return
        
        evals = torch.stack([c.eval for c in children_with_eval], dim=0)  # (k, 3)
        scalar_evals = evals[:, 2] - evals[:, 0]  # White perspective
        idx = torch.argmax(scalar_evals) if self.color else torch.argmin(scalar_evals)

        self.eval = evals[idx]
        self.best_child = children_with_eval[idx]

    # --- Methods for MCTS ---

    def initialize_ucb(self, prior_p: float):
        """Initialize UCB statistics for MCTS."""
        self.ucb = UCB(prior_p=prior_p)

    def select_ucb_child(self, cpuct: float):
        """Select the child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        parent_visits = max(1, self.n_visits)

        for child in self.children:
            score = child.ucb.score(parent_visits, cpuct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def update_ucb(self, value: float):
        """Update Q, W, N statistics (MCTS)."""
        if self.ucb is None:
            raise ValueError("UCB statistics not initialized.")
        self.ucb.update(value)

    def update_recursive(self, value: float):
        """Backpropagate value recursively up the tree."""
        if self.parent:
            self.parent.update_recursive(-value)
        self.update_ucb(value)

    def get_visit_distribution(self, temperature: float = 1.0):
        visit_counts = [(moves, child.ucb.n_visits) for moves, child in zip(self.moves, self.children)]

        if not visit_counts:
            return [], torch.tensor([])

        if temperature == 0:
            # Greedy — select the most visited move only
            best_action, _ = max(visit_counts, key=lambda x: x[1])
            return [action for action, _ in visit_counts], torch.tensor([
                1.0 if action == best_action else 0.0 for action, _ in visit_counts
            ])

        moves, counts = zip(*visit_counts)
        counts = np.array(counts, dtype=np.float32)
        counts = counts ** (1 / temperature)
        probs = counts / counts.sum()

        return list(moves), torch.from_numpy(probs).float()