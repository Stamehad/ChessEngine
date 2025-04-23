import chess
import torch
from typing import List, Optional, Dict, Tuple
from chessengine.chess_rl.mcts.tree import Node
from chessengine.model.prediction import batch_predict


class SearchTree:
    def __init__(self, board: chess.Board):
        self.board = board.copy()
        self.root = Node(parent=None, prior_p=1.0, state=self.board)
        self.leaf_node: Optional[Node] = self.root
        self.leaf_board: Optional[chess.Board] = self.board.copy()
        self.leaf_probs: Optional[torch.Tensor] = None
        self.leaf_moves: Optional[List[chess.Move]] = None
        self.leaf_value: Optional[float] = None

    def select_leaf(self, cpuct: float = 1.0):
        node = self.root
        board = self.board.copy()

        while not node.is_leaf():
            action, child = node.select_child(cpuct)
            if child is None:
                break # fallback: treat current node as leaf
            board.push(action)
            node = child

        self.leaf_node = node
        self.leaf_board = board

    def update_leaf(self, move_probs, value_probs, legal_moves):
        assert self.leaf_node is not None
        if self.leaf_board.is_game_over():
            self.leaf_value = self.get_outcome()
        else:
            self.leaf_value = value_probs[2].item() - value_probs[0].item()
        
        self.leaf_probs = move_probs # includes padding
        self.leaf_moves = legal_moves

    def expand_leaf(self):
        self.leaf_node.expand(zip(self.leaf_moves, self.leaf_probs.tolist()), self.leaf_board)

    def backprop_leaf(self):
        self.leaf_node.update_recursive(self.leaf_value)

    def get_outcome(self):
        assert self.leaf_board.is_game_over()
        outcome = self.leaf_board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        else:
            return 1.0 if outcome.winner == chess.WHITE else -1.0
        
    def get_policy(self, temperature: float = 1.0) -> Tuple[List[chess.Move], torch.Tensor]:
        return self.root.get_visit_distribution(temperature) # (legal_moves, pi)


class SearchForest:
    def __init__(self, boards: List[chess.Board], model):
        self.trees = [SearchTree(board) for board in boards]
        self.model = model
        self.device = next(model.parameters()).device

    def evaluate_leaves(self):
        leaf_boards = [tree.leaf_board for tree in self.trees]
        move_probs, value_probs, legal_moves = batch_predict(self.model, leaf_boards, self.device)
        self.update_leaves(move_probs, value_probs, legal_moves)
    
    def update_leaves(self, move_probs, value_probs, legal_moves):
        for i, tree in enumerate(self.trees):
            tree.update_leaf(move_probs[i], value_probs[i], legal_moves[i])
    
    def select_leaves(self, cpuct: float = 1.0):
        for tree in self.trees:
            tree.select_leaf(cpuct)

    def expand_and_backprop(self):
        for tree in self.trees:
            tree.backprop_leaf()
            tree.expand_leaf()

    def get_policies(self, temperature: float):
        return [tree.get_policy_dict(temperature) for tree in self.trees]
