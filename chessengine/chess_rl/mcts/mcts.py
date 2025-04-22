import torch
import numpy as np
import chess
import math
from .tree import Node
from chessengine.model.prediction import predict, batch_predict
import logging

logger = logging.getLogger(__name__)

class MCTS:
    """Monte Carlo Tree Search algorithm."""
    def __init__(self, model, config):
        self.model = model # The PyTorch transformer model (in eval mode)
        self.config = config['mcts'] # Dictionary containing MCTS parameters
        self.device = next(model.parameters()).device # Use model's device
        

    def run_mcts_search(self, root_board):
        """Run MCTS from root_board using the model."""
        num_simulations = self.config['num_simulations']
        cpuct = self.config['cpuct']
        temperature = self.config['temperature_start']

        root_node = Node(parent=None, prior_p=1.0, state=root_board.copy())

        if root_board.is_game_over():
            return self._get_policy_target(root_node, temperature=0.0)

        probs, value_probs, legal_moves = predict(self.model, root_board, self.device) # (L,), (3,), List[chess.Move]
        scalar_value = value_probs[2].item() - value_probs[0].item()
        root_node.expand(zip(legal_moves, probs.tolist()), root_board)
        root_node.update_recursive(scalar_value)

        for _ in range(num_simulations):
            node = root_node
            search_path = [node]
            current_board = root_board.copy()

            # Selection
            while not node.is_leaf():
                action, node = node.select_child(cpuct)
                if node is None:
                    logger.error("Selection returned None node.")
                    break
                current_board.push(action)
                search_path.append(node)

            if node is None:
                continue

            # Expansion & Evaluation
            if current_board.is_game_over():
                outcome = current_board.outcome()
                if outcome:
                    scalar_value = 1.0 if outcome.winner == chess.WHITE else -1.0 if outcome.winner == chess.BLACK else 0.0
                else:
                    scalar_value = 0.0
            else:
                probs, value_probs, legal_moves = predict(self.model, current_board, self.device) # (L,), (3,), List[chess.Move]
                scalar_value = value_probs[2].item() - value_probs[0].item()
                if probs is not None and legal_moves:
                    node.expand(zip(legal_moves, probs.tolist()), current_board)

            node.update_recursive(scalar_value)

        return self._get_policy_target(root_node, temperature)

    def _get_policy_target(self, node, temperature):
        """Calculate the MCTS policy target distribution based on visit counts."""
        visit_counts = {action: child.n_visits for action, child in node.children.items()}
        total_visits = sum(visit_counts.values())

        if total_visits == 0: # No visits (shouldn't happen after sims)
            # Fallback: return uniform distribution over legal moves? Or prior?
            num_children = len(node.children)
            if num_children == 0: return {} # No legal moves
            return {action: 1.0 / num_children for action in node.children.keys()}

        if temperature == 0: # Deterministic: choose the most visited move
            most_visited_action = max(visit_counts, key=visit_counts.get)
            policy = {action: 1.0 if action == most_visited_action else 0.0 for action in node.children.keys()}
        else:
            # Apply temperature scaling
            scaled_visits = {action: count**(1.0 / temperature) for action, count in visit_counts.items()}
            total_scaled_visits = sum(scaled_visits.values())
            policy = {action: scaled_count / total_scaled_visits for action, scaled_count in scaled_visits.items()}

        return policy
    
class BATCHED_MCTS:
    """Monte Carlo Tree Search algorithm with batch predictions."""
    def __init__(self, model, config):
        self.model = model # The PyTorch transformer model (in eval mode)
        self.config = config # Dictionary containing MCTS parameters
        self.device = next(model.parameters()).device # Use model's device
        self.node_cache = {} # Optional: Cache nodes based on board FEN for reuse across batches if applicable

    def _remove_padding(self, move_probs, legal_moves):
        moves_probs = move_probs.unbind(0) # (B, L)
        moves_probs = [p[:len(moves)].tolist() for p, moves in zip(moves_probs, legal_moves)] # remove any padding
        return moves_probs
    
    def _get_scalar_value(self, value_probs):
        """Returns a list of float scalar values."""
        values = value_probs[:, 2] - value_probs[:, 0] # (B,)
        values = values.tolist() 
        return values
    
    def _get_outcome(self, board):
        outcome = board.outcome()
        if outcome:
            value = 1.0 if outcome.winner == chess.WHITE else -1.0 if outcome.winner == chess.BLACK else 0.0
        else:
            value = 0.0 # Draw
        return value
    
    def _initialize_roots(self, root_boards):
        """
        For each non-terminal root board, expand the root node using model predictions.
        Return a tuple: (root_nodes, results, active_boards)
        """
        batch_size = len(root_boards)
        active_roots = {} # {idx: node} for active boards
        results = [None] * batch_size # Store final policies

        # 1. Initialize Root Nodes and Identify Initial Predictions
        for i, board in enumerate(root_boards):
            if board.is_game_over():
                logger.warning(f"Board {i} is already game over.")
                results[i] = {}
            else:
                root_node = Node(parent=None, prior_p=1.0, state=board.copy())
                active_roots[i] = root_node # Store active boards for batch prediction

        return active_roots, results 
    
    def _get_node_prediction(self, nodes_dict): 
        """Expand root nodes for active boards using model predictions.
        Args:
            nodes_dict: Dictionary of active root nodes (game not over) - {idx: node}
        Returns:
            Dictionary of predictions for each node (idx: (moves, probs, value))
        """
        if not nodes_dict:
            return {}
        # --- Parallel Tensor Operations ---
        node_idxs = list(nodes_dict.keys())
        boards = [nodes_dict[idx].state.copy() for idx in node_idxs]

        # (B, L), (B, 3), List[List[chess.Move]], List[int]
        move_probs, value_probs, legal_moves = batch_predict(self.model, boards, self.device)
        
        moves_probs = self._remove_padding(move_probs, legal_moves) # List[List[float]]
        values = self._get_scalar_value(value_probs) # List[float]

        pred = zip(node_idxs, legal_moves, moves_probs, values)
        pred = {idx: (moves, probs, value) for idx, moves, probs, value in pred}
        return pred
    
    def _expand_nodes(self, nodes_dict, pred, results):
        """Expand root nodes using predictions."""
        if not pred:
            return nodes_dict, results
        
        for idx, (moves, probs, value) in pred.items():
            node = nodes_dict[idx]
            board = node.state.copy() # Get the board state from the root node

            # If the root_node was already marked as done (e.g. error in previous step), skip
            if node is None:
                continue

            # Expand: create children nodes for each legal move with its corresponding probability
            actions_probs = zip(moves, probs)
            node.expand(actions_probs, board)
            # Backpropagate the initial evaluation value.
            node.update_recursive(value)
            # root_nodes[idx] = root_node # Update the root node in the list (not strictly necessary)
        return nodes_dict, results
    
    def _select_leaf(self, active_roots, cpuct):
        """Select leaf nodes for simulation."""
        leaf_nodes = {}
        for idx, node in active_roots.items():
            if node is None:
                continue
            search_path = [node]
            current_board = node.state.copy()

            # Use policy to find a new leaf node to expand, moving step by step from the root
            while not node.is_leaf():
                action, node = node.select_child(cpuct)
                if node is None:
                    logger.error(f"Selection returned None node for tree {idx}.")
                    search_path = None
                    break
                current_board.push(action)
                search_path.append(node)

            if search_path is None:
                node = None
                logger.warning(f"Skipping simulation for tree {idx} due to selection error.")
                continue

            # 3.b Check if selected leaf is terminal or needs evaluation
            if current_board.is_game_over():
                value = self._get_outcome(current_board)
                node.update_recursive(value)
            else:
                leaf_nodes[idx] = node # Store leaf nodes for batch evaluation
        return leaf_nodes
    
    def run_batched_mcts_search(self, root_boards):
        """Run MCTS for a batch of root_boards using the model."""
        # Initialize Root Nodes
        
        # active_roots: dictionary of active root nodes (game not over) - {idx: node}
        # results: result of game if board is terminal, otherwise None               
        active_roots, results = self._initialize_roots(root_boards)
        
        # For non-terminal roots, get predictions and expand nodes (update results if terminal nodes is reached)
        pred = self._get_node_prediction(active_roots)
        active_roots, results = self._expand_nodes(active_roots, pred, results)

        # Run Simulations in Batches
        num_simulations = self.config['mcts']['num_simulations']
        cpuct = self.config['mcts']['cpuct']
        temperature = self.config['mcts']['temperature_start'] # Use configured temperature

        for sim in range(num_simulations):
            # print(f"--- Simulation {sim+1}/{num_simulations} ---")
            leaf_nodes = self._select_leaf(active_roots, cpuct)

            pred = self._get_node_prediction(leaf_nodes)
            leaf_nodes, results = self._expand_nodes(leaf_nodes, pred, results)

        # 4. Calculate Final Policies
        for idx, node in active_roots.items():
            if node is None:
                logger.warning(f"Node is None in final policy computation for tree {idx}. Returning empty policy.")
                results[idx] = {}
                continue
            # Get the final policy target for each root node
            results[idx] = self._get_policy_target(node, temperature)

        return results # List of policy dictionaries

    def _get_policy_target(self, node, temperature):
        """Calculate the MCTS policy target distribution based on visit counts."""
        if not node.children: # Handle nodes with no children (terminal or not expanded)
             return {}

        visit_counts = {action: child.n_visits for action, child in node.children.items()}
        # Check if any visits occurred. If the root was terminal or no sims ran, visits might be 0.
        if not any(visit_counts.values()):
             # Fallback: Uniform distribution over children? Or based on prior? Use Uniform for now.
             num_children = len(node.children)
             # print("Warning: No visits recorded for policy target. Returning uniform.")
             return {action: 1.0 / num_children for action in node.children.keys()}


        if temperature == 0: # Deterministic: choose the most visited move
            # Find the action(s) with the maximum visit count
            max_visits = max(visit_counts.values())
            # Handle ties by selecting one, e.g., the first one found or randomly
            most_visited_actions = [action for action, count in visit_counts.items() if count == max_visits]
            # Deterministic choice (e.g., first max)
            chosen_action = most_visited_actions[0]
            # Create policy dict
            policy = {action: 1.0 if action == chosen_action else 0.0 for action in node.children.keys()}

        else:
            # Apply temperature scaling
            scaled_visits = {action: count**(1.0 / temperature) for action, count in visit_counts.items()}
            total_scaled_visits = sum(scaled_visits.values())
            if total_scaled_visits == 0: # Avoid division by zero if all counts were 0 and temp > 0
                 num_children = len(node.children)
                 # print("Warning: Scaled visits sum to zero. Returning uniform.")
                 return {action: 1.0 / num_children for action in node.children.keys()}

            policy = {action: scaled_count / total_scaled_visits for action, scaled_count in scaled_visits.items()}

        return policy

# Helper function to reconstruct board state from path (needed for expansion)
# This assumes root node stores the initial state.
def board_from_node(node):
    """ Reconstructs the board state leading to this node by traversing up. """
    actions = []
    temp_node = node
    while temp_node.parent is not None:
        # Find the action that led to temp_node from its parent
        action = None
        for act, child in temp_node.parent.children.items():
            if child is temp_node:
                action = act
                break
        if action is None:
             raise RuntimeError(f"Could not find action leading to node {temp_node} from parent {temp_node.parent}")
        actions.append(action)
        temp_node = temp_node.parent

    # The final temp_node is the root node
    if temp_node.state is None:
        raise RuntimeError("Root node does not contain the initial board state.")

    board = temp_node.state.copy()
    for action in reversed(actions):
        board.push(action)
    return board