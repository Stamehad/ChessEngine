# chess_rl/mcts/mcts.py
import torch
import numpy as np
import chess
import math
from .tree import Node

class MCTS:
    """Monte Carlo Tree Search algorithm."""
    def __init__(self, model, config):
        self.model = model # The PyTorch transformer model (in eval mode)
        self.config = config # Dictionary containing MCTS parameters
        self.device = next(model.parameters()).device # Use model's device

    def _preprocess_state(self, board):
        """Converts python-chess board to model input tensor.
           Needs to match the preprocessing used during model training.
           Placeholder implementation - REPLACE with your actual preprocessing.
        """
        # Example: Replace with your actual board representation logic
        # This might involve creating planes for piece positions, castling rights, etc.
        # Returning a dummy tensor for structure
        return torch.randn(1, 8, 8, 18).to(self.device) # Example shape

    def _get_model_output(self, board):
        """Get policy and value from the neural network."""
        self.model.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            state_tensor = self._preprocess_state(board).to(self.device)
            policy_logits, value_pred = self.model(state_tensor)

            # Ensure policy_logits are for legal moves only
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                 return {}, value_pred.item() # No legal moves

            # Map model output indices to actual chess moves
            # This requires knowing the mapping used by your model.
            # Placeholder: Assuming model outputs logits for all possible moves (e.g., 4672)
            # and you have a function `map_logits_to_legal_moves`
            move_probs = self._map_logits_to_legal_moves(policy_logits, legal_moves, board)

            # Optional Top-K/Nucleus Filtering
            if self.config['mcts']['top_k_filtering'] < 1.0:
                 move_probs = self._filter_policy(move_probs, self.config['mcts']['top_k_filtering'])

            # Normalize probabilities
            prob_sum = sum(p for _, p in move_probs)
            if prob_sum > 1e-6:
                normalized_probs = [(move, p / prob_sum) for move, p in move_probs]
            else: # If sum is tiny, distribute uniformly (should be rare)
                 normalized_probs = [(move, 1.0/len(move_probs)) for move, _ in move_probs]


        return dict(normalized_probs), value_pred.item()

    def _map_logits_to_legal_moves(self, policy_logits, legal_moves, board):
        """Maps policy logits to legal moves and applies softmax.
           REPLACE with your model's specific move encoding/decoding.
        """
        # Example Placeholder: Assumes policy_logits is [1, num_possible_moves]
        # Needs functions `move_to_index` and `index_to_move` based on your model
        move_probs = []
        logit_values = []
        legal_move_indices = [] # Indices in the logit tensor corresponding to legal moves

        # --- Replace this section with your actual logic ---
        # Example: Suppose you have a function move_to_index(move, board)
        all_possible_moves = self._get_all_possible_moves() # Function returning list/dict of all moves your model predicts
        move_map = {move: i for i, move in enumerate(all_possible_moves)} # Example map

        for move in legal_moves:
            uci_move = move.uci()
            if uci_move in move_map: # Or however you map your moves
                idx = move_map[uci_move]
                legal_move_indices.append(idx)
                logit_values.append(policy_logits[0, idx]) # Assuming batch size 1
            else:
                 # This should not happen if your model covers all move types
                 print(f"Warning: Legal move {uci_move} not found in model output map.")
                 pass # Handle missing moves if necessary

        if not logit_values:
            return [] # No legal moves map?

        # Apply softmax only to legal move logits
        legal_logits_tensor = torch.tensor(logit_values, device=policy_logits.device)
        probabilities = torch.softmax(legal_logits_tensor, dim=0)

        for i, move in enumerate(legal_moves):
             if move.uci() in move_map: # Check again for consistency
                move_probs.append((move, probabilities[i].item()))
        # --- End Replace Section ---

        return move_probs

    def _get_all_possible_moves(self):
        # Placeholder: Return a list of all possible UCI moves your model handles
        # This is crucial for mapping logits correctly.
        # A common approach is representing moves like 'e2e4', 'a7a8q', etc.
        # This list should be fixed and match your model's output layer definition.
        raise NotImplementedError("Please implement _get_all_possible_moves based on your model's output layer.")
        # Example (very simplified): return ['e2e4', 'd2d4', ...]

    def _filter_policy(self, move_probs, threshold):
        """Applies top-k/nucleus filtering."""
        if not move_probs:
            return []
        # Sort by probability descending
        sorted_probs = sorted(move_probs, key=lambda item: item[1], reverse=True)
        cumulative_prob = 0.0
        filtered_probs = []
        for move, prob in sorted_probs:
            filtered_probs.append((move, prob))
            cumulative_prob += prob
            if cumulative_prob >= threshold:
                break
        return filtered_probs


    def run_simulations(self, root_board, num_simulations):
        """Run MCTS simulations from the root node."""
        root_node = Node(parent=None, prior_p=1.0, state=root_board.copy())

        # Initial expansion of the root node
        policy, value = self._get_model_output(root_board)
        if root_board.is_game_over():
            # Handle game over at root - MCTS normally not called here, but for safety:
             return self._get_policy_target(root_node, temperature=0.0) # Return deterministic

        root_node.expand(policy.items(), root_board)
        root_node.update_recursive(value) # Initial backprop of root value

        for _ in range(num_simulations):
            node = root_node
            search_path = [node]
            current_board = root_board.copy() # Board state for the traversal

            # 1. Select
            while not node.is_leaf():
                action, node = node.select_child(self.config['mcts']['cpuct'])
                if node is None: # Should not happen if selection is correct
                    print("Error: Selection returned None node.")
                    # Fallback or error handling needed
                    break # Exit simulation
                current_board.push(action)
                search_path.append(node)

            if node is None: continue # Skip if selection failed

            # 2. Expand & Evaluate
            leaf_value = 0.0
            if current_board.is_game_over():
                outcome = current_board.outcome()
                if outcome:
                    if outcome.winner == chess.WHITE:
                        leaf_value = 1.0
                    elif outcome.winner == chess.BLACK:
                        leaf_value = -1.0
                    else: # Draw
                        leaf_value = 0.0
                else: # Draw by insufficient material, 50-move rule etc.
                    leaf_value = 0.0
            else:
                # Expand the leaf node using the network
                policy, leaf_value = self._get_model_output(current_board)
                if policy: # Only expand if there are legal moves
                    node.expand(policy.items(), current_board)

            # 3. Backpropagate
            # Value needs to be from the perspective of the player whose turn it was
            # *at the parent* of the leaf node.
            # The network's value `leaf_value` is from the perspective of the current player
            # at the leaf state. Backpropagation negates it at each step up.
            node.update_recursive(leaf_value)

        # After simulations, return the improved policy distribution
        return self._get_policy_target(root_node, self.config['mcts']['temperature_start']) # Use configured temp


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