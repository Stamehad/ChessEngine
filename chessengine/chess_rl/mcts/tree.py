import numpy as np
import math

class Node:
    """Represents a node in the Monte Carlo Tree Search."""
    def __init__(self, parent, prior_p, state):
        self.parent = parent
        self.children = {}  # Map from action (move) to Node
        self.state = state  # Board state associated with this node

        self.n_visits = 0      # N(s, a): Visit count for edge leading to this node
        self.q_value = 0.0     # Q(s, a): Mean action value for edge leading to this node
        self.w_value = 0.0     # W(s, a): Total action value for edge leading to this node
        self.prior_p = prior_p # P(s, a): Prior probability for edge leading to this node

    def expand(self, action_probs, board):
        """Expand tree by creating new children.
        action_probs: list of (action, probability) tuples for legal moves.
        board: The current python-chess board object after the move leading here.
        """
        policy = {action: prob for action, prob in action_probs}
        for action, prob in policy.items():
            if action not in self.children:
                # Create child node with its own state (board after the action)
                # Cloning the board is important!
                next_board = board.copy()
                next_board.push(action)
                self.children[action] = Node(parent=self, prior_p=prob, state=next_board)

    def select_child(self, cpuct):
        """Select the child node with the highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        parent_visits = self.n_visits # Should be sum of child visits if expanded

        for action, child in self.children.items():
            score = child.get_ucb_score(parent_visits, cpuct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_ucb_score(self, parent_visits, cpuct):
        """Calculate the UCB score for this node (edge P(s,a))."""
        # Q value: Average reward from this action
        q_score = self.q_value

        # U value: Exploration bonus based on prior probability and visit counts
        # Adding a small epsilon to prevent division by zero if n_visits is 0 initially
        u_score = cpuct * self.prior_p * (math.sqrt(parent_visits) / (1 + self.n_visits))

        return q_score + u_score

    def update(self, value):
        """Update node values from leaf evaluation."""
        self.n_visits += 1
        self.w_value += value
        # Update Q value (average)
        self.q_value = self.w_value / self.n_visits

    def update_recursive(self, value):
        """Backpropagate the value up the tree."""
        # If it's not root, continue backpropagating
        if self.parent:
            # Value should be negated for the parent, as it's from the opponent's perspective
            self.parent.update_recursive(-value)
        self.update(value)

    def is_leaf(self):
        """Check if the node is a leaf node (has no children)."""
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None