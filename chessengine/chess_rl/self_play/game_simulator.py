# chess_rl/self_play/game_simulator.py
import chess
import numpy as np
import random
import time
import os
from ..mcts.mcts import MCTS

class GameSimulator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.mcts = MCTS(model, config)

    def _get_temperature(self, move_count):
        """Get temperature based on move count."""
        start_temp = self.config['mcts']['temperature_start']
        end_temp = self.config['mcts']['temperature_end']
        decay_moves = self.config['mcts']['temp_decay_moves']

        if move_count < decay_moves:
            # Linear decay, could use exponential etc.
            return start_temp - (start_temp - end_temp) * (move_count / decay_moves)
        else:
            return end_temp

    def _select_move(self, policy_target, temperature):
        """Selects a move based on the policy target distribution."""
        moves = list(policy_target.keys())
        probabilities = list(policy_target.values())

        if not moves:
            return None # No legal moves

        if temperature == 0: # Deterministic selection
            best_move_index = np.argmax(probabilities)
            return moves[best_move_index]
        else:
            # Sample according to probabilities (adjusted by temperature implicitly in policy_target)
            # Ensure probabilities sum to 1 (might have float inaccuracies)
            prob_sum = sum(probabilities)
            if abs(prob_sum - 1.0) > 1e-6:
                 probabilities = [p / prob_sum for p in probabilities] # Normalize

            # Add small epsilon to handle potential zero probabilities if needed after filtering
            # probabilities = [p + 1e-9 for p in probabilities]
            # probabilities = [p / sum(probabilities) for p in probabilities]

            try:
                selected_move = np.random.choice(moves, p=probabilities)
                return selected_move
            except ValueError as e:
                 print(f"Error sampling move: {e}")
                 print(f"Moves: {moves}")
                 print(f"Probabilities: {probabilities}")
                 # Fallback: choose uniformly or deterministically
                 return moves[np.argmax(probabilities)]


    def play_game(self, start_fen=None):
        """Plays a single game of self-play."""
        board = chess.Board(fen=start_fen) if start_fen else chess.Board()
        game_history = [] # Stores (state_repr, policy_target, current_player_turn)
        move_count = 0

        while not board.is_game_over() and move_count < self.config['self_play']['max_game_length']:
            start_time = time.time()

            # 1. Run MCTS simulations
            num_sims = self.config['mcts']['num_simulations']
            policy_target = self.mcts.run_simulations(board.copy(), num_sims) # Pass copy

            if not policy_target: # No legal moves found by MCTS (should match board.is_game_over typically)
                print("Warning: MCTS returned empty policy. Game should be over.")
                break

            # 2. Store data for training
            # Convert board state to the representation your model expects
            # This MUST match mcts._preprocess_state (without batch dim usually)
            state_representation = self.mcts._preprocess_state(board).squeeze(0).cpu().numpy() # Example
            current_player = 1 if board.turn == chess.WHITE else -1 # White = 1, Black = -1

            # Store policy target keyed by move UCI string for serialization
            policy_target_serializable = {move.uci(): prob for move, prob in policy_target.items()}
            game_history.append((state_representation, policy_target_serializable, current_player))

            # 3. Select and play move
            temperature = self._get_temperature(move_count)
            move = self._select_move(policy_target, temperature)

            if move is None:
                 print("Warning: No move selected. Breaking game loop.")
                 break # Should not happen if policy_target is valid

            board.push(move)
            move_count += 1
            # print(f"Move {move_count}: {move.uci()}, Sims: {num_sims}, Temp: {temperature:.2f}, Time: {time.time()-start_time:.2f}s")


        # 4. Determine game outcome
        outcome = board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                z = 1.0
            elif outcome.winner == chess.BLACK:
                z = -1.0
            else: # Draw
                z = 0.0
        else: # Draw by other means (e.g., max length)
            z = 0.0
        print(f"Game finished in {move_count} moves. Outcome: {board.result()} (z={z})")


        # 5. Prepare final training data: (state, policy_target, outcome_z)
        training_data = []
        for state_repr, pi_target, player_turn in game_history:
            # Value target is the final outcome z, from the perspective of the player whose turn it was
            value_target = z * player_turn
            training_data.append((state_repr, pi_target, value_target))

        return training_data