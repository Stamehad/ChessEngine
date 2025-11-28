import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from pytorchchess import TorchBoard
from pytorchchess.state import LegalMoves
from .beam_search import BeamSearchState
from .position_queue import PositionQueue


@dataclass
class SelfPlayBatch:
    step_id: torch.Tensor   # (B,)
    layer_id: torch.Tensor  # (B,)
    game_ids: torch.Tensor  # (B,)
    features: torch.Tensor  # (B, 64, 21)
    lm_tensor: torch.Tensor # (B, 64, L_max)
    encoded: torch.Tensor   # (B, L_max)
    move_idx: torch.Tensor  # (B,) long, -1 until assigned
    values: torch.Tensor    # (B,) uint8, 255 until assigned

    def concat(self, other: "SelfPlayBatch") -> "SelfPlayBatch":
        return SelfPlayBatch(
            step_id=torch.cat([self.step_id, other.step_id], dim=0),
            layer_id=torch.cat([self.layer_id, other.layer_id], dim=0),
            game_ids=torch.cat([self.game_ids, other.game_ids], dim=0),
            features=torch.cat([self.features, other.features], dim=0),
            lm_tensor=torch.cat([self.lm_tensor, other.lm_tensor], dim=0),
            encoded=torch.cat([self.encoded, other.encoded], dim=0),
            move_idx=torch.cat([self.move_idx, other.move_idx], dim=0),
            values=torch.cat([self.values, other.values], dim=0),
        )
    
    # layer_mask = (layer_id == layer)
    # v (N,)


class SelfPlayBuffer:
    """Flat storage for per-ply training samples."""

    def __init__(self):
        self.batch: Optional[SelfPlayBatch] = None

    def add_batch(
        self,
        game_ids: torch.Tensor,
        features: torch.Tensor,
        legal_moves,
        mask: torch.Tensor,
        layer_id: int,
        step_id: int,
    ):
        lm_tensor = legal_moves.tensor
        encoded = legal_moves.encoded
        if lm_tensor is None:
            raise ValueError("Legal move tensor is required for training samples.")

        selected = mask.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            return

        feats = features[selected].reshape(selected.numel(), 64, 21).clone().detach()
        lm = lm_tensor[selected].clone().detach()
        enc = encoded[selected].clone().detach()
        games = game_ids[selected].clone().detach()
        layer = torch.full_like(games, layer_id)
        steps = torch.full_like(games, step_id)
        move_idx = torch.full((selected.numel(),), -1, dtype=torch.long, device=games.device)
        values = torch.full((selected.numel(),), 255, dtype=torch.uint8, device=games.device)

        new_batch = SelfPlayBatch(steps, layer, games, feats, lm, enc, move_idx, values)
        if self.batch is None:
            self.batch = new_batch
        else:
            self.batch = self.batch.concat(new_batch)

    def assign_policy(self, step_id: int, layer_id: int, root_moves: torch.Tensor):
        if self.batch is None:
            return

        if root_moves.dim() > 1:
            root_moves = root_moves[:, 0] # (G,D) -> (G,)

        mask = (self.batch.step_id == step_id) & (self.batch.layer_id == layer_id)
        if not mask.any():
            return

        mask_idx = mask.nonzero(as_tuple=True)[0]
        games = self.batch.game_ids[mask].long()
        moves = root_moves[games]
        valid_mask = moves >= 0

        if not valid_mask.any():
            return

        mask_idx = mask_idx[valid_mask]
        moves = moves[valid_mask]
        encoded = self.batch.encoded[mask][valid_mask]
        match_matrix = encoded == moves.unsqueeze(1)
        move_indices = match_matrix.long().argmax(dim=1)
        self.batch.move_idx[mask_idx] = move_indices

    def mark_results(
        self,
        layer_ids: torch.Tensor,
        game_ids: torch.Tensor,
        results: torch.Tensor,
    ):
        if self.batch is None or results.numel() == 0:
            return

        layer_mask = self.batch.layer_id.unsqueeze(1) == layer_ids.view(1, -1)
        game_mask = self.batch.game_ids.unsqueeze(1) == game_ids.view(1, -1)
        match_matrix = layer_mask & game_mask
        if not match_matrix.any():
            return

        encoded = (results + 1).to(torch.float32)
        row_values = match_matrix.float() @ encoded
        matched_rows = match_matrix.any(dim=1)
        if matched_rows.any():
            self.batch.values[matched_rows] = row_values[matched_rows].to(
                self.batch.values.dtype
            )


class SelfPlayEngine:
    """
    A beam-search based self-play loop.

    The class keeps the high level flow explicit:
        1. pull a fresh layer of root positions
        2. compute legal moves + feature tensors via the fused board API
        3. drop terminal boards and store their result
        4. evaluate the remaining boards with the model
        5. backpropagate finished trees / push PV moves to the queue
        6. expand the unfinished boards

    This file intentionally skips profiling / training-buffer plumbing so it can
    serve as a clear template for the future RL manager.
    """

    def __init__(
        self,
        model,
        expansion_factors: torch.Tensor,
        pv_depth: int = 3,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ):
        self.model = model
        self.device = device
        self.expansion_factors = torch.as_tensor(
            expansion_factors, device=device, dtype=torch.long
        )
        self.D = self.expansion_factors.numel()
        self.pv_depth = pv_depth
        self.generator = generator

        self.position_queue: Optional[PositionQueue] = None
        self.beam_state: Optional[BeamSearchState] = None
        self.beam_boards: Optional[TorchBoard] = None
        self.num_games: Optional[int] = None
        self.step = 0
        self.sample_buffer = SelfPlayBuffer()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize(self, initial_boards: TorchBoard):
        """Register the initial batch of positions and allocate state."""
        self.num_games = initial_boards.batch_size // (self.D + 1)
        self.position_queue = PositionQueue.from_board_list(
            initial_boards, self.num_games, self.device
        )
        self.beam_state = BeamSearchState.initialize(
            self.num_games, self.expansion_factors, debug=False, device=self.device
        )
        self.beam_boards = None
        self.step = 0

    # ------------------------------------------------------------------ #
    # One self-play step
    # ------------------------------------------------------------------ #
    def step_once(self):
        """Run a single beam-search iteration."""
        # if self.step % 50 == 0 or self.step > 430:
        #     print("--------------------------------------------------")
        #     print(f"Step {self.step}: beam search boards = {len(self.beam_state)}")

        assert self.position_queue is not None and self.beam_state is not None
        
        # ------------------------------------------------------------------ #
        # Add root positions to start search
        # ------------------------------------------------------------------ #
        current_layer = self._add_layer()

        # ------------------------------------------------------------------ #
        # Get legal moves and features
        # ------------------------------------------------------------------ #
        
        if len(self.beam_state) > 0:
            legal_moves, features = self.beam_boards.get_legal_moves_fused(return_features=True)
            features = features.float()
        
        # ------------------------------------------------------------------ #
        # Separate finished positions
        # ------------------------------------------------------------------ #
            terminal_mask, terminal_results = self.beam_boards.is_game_over(
                max_plys=300,
                enable_fifty_move_rule=True,
                enable_insufficient_material=True,
                enable_threefold_repetition=True,
            )

        # ------------------------------------------------------------------ #
        # Save root positions for training data
        # ------------------------------------------------------------------ #
            root_mask = self.beam_state.depth == 0
            self.sample_buffer.add_batch(
                self.beam_state.game,
                features,
                legal_moves,
                mask=root_mask,
                layer_id=current_layer,
                step_id=self.step,
            )
            self._handle_terminal_positions(terminal_mask, terminal_results)

            features = features[~terminal_mask]
            scalar_eval, move_pred = self._evaluate_positions(features)
            scalar_eval, move_pred = self._handle_finished_expansion(scalar_eval, move_pred)
        
        self._backpropagate_finished_trees()

        if len(self.beam_state) > 0:
            self._expand_remaining_positions(legal_moves, move_pred)

        self.step += 1

    # ------------------------------------------------------------------ #
    # Step helpers
    # ------------------------------------------------------------------ #
    def _add_layer(self):
        """Append the next layer of positions to the working beam."""
        next_layer = self.step % (self.D + 1)
        new_positions, finished_mask = self.position_queue.get_active_layer(next_layer, clone=True)
        if new_positions.board_tensor.shape[0] == 0:
            return next_layer
        self.beam_state.add_new_layer(next_layer, finished_mask)
        if self.beam_boards is None:
            self.beam_boards = new_positions
        else:
            self.beam_boards = self.beam_boards.concat(new_positions)
        return next_layer

    def _handle_terminal_positions(self, terminal_mask: torch.Tensor, results: torch.Tensor):
        """Drop terminal boards and store their evaluations."""
        if not terminal_mask.any():
            return

        assert (
            results.numel() == terminal_mask.sum().item()
        ), "Mismatch between terminal mask and results length"

        self.beam_state.store_early_evaluations(terminal_mask, results)

        full_results = torch.zeros(terminal_mask.shape[0], dtype=results.dtype, device=self.device)
        full_results[terminal_mask] = results

        root_mask = terminal_mask & (self.beam_state.depth == 0)
        if root_mask.any():
            layers = self.beam_state.layer[root_mask]
            games = self.beam_state.game[root_mask]
            root_results = full_results[root_mask]

            # update finished masks in position queue
            self.position_queue.update_finished(layers, games)

            print(f"Finished games at step {self.step}: {root_mask.sum().item()}, results={root_results.tolist()}, layers={layers.tolist()}, games={games.tolist()}")
            self.sample_buffer.mark_results(layers, games, root_results)

        self.beam_boards = self.beam_boards.select(~terminal_mask)
        self.beam_state = self.beam_state[~terminal_mask]

    def _evaluate_positions(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the chess network on the prepared feature tensor."""
        with torch.no_grad():
            x_out, move_pred = self.model(features)
            eval_logits = self.model.loss_module.prob_eval_loss.head(x_out)
            prob_eval = F.softmax(eval_logits, dim=-1)
            scalar_eval = prob_eval[:, 0] - prob_eval[:, 2]
        return scalar_eval, move_pred.detach()

    def _handle_finished_expansion(
        self, scalar_eval: torch.Tensor, move_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store evaluations for beams that reached max depth and drop them."""
        finished_mask = self.beam_state.get_finished_expansion()
        if not finished_mask.any():
            return scalar_eval, move_pred

        self.beam_state.store_final_evaluations(
            finished_mask, scalar_eval[finished_mask]
        )
        self.beam_boards = self.beam_boards.select(~finished_mask)
        self.beam_state = self.beam_state[~finished_mask]

        if len(self.beam_state) == 0:
            empty_moves = move_pred.new_zeros((0,) + move_pred.shape[1:])
            return scalar_eval.new_zeros(0), empty_moves

        return scalar_eval[~finished_mask], move_pred[~finished_mask]

    def _backpropagate_finished_trees(self):
        """Perform minimax backpropagation for the finished layer."""
        finished_layer = self.beam_state.get_finished_layer(self.step)
        if finished_layer is None or self.position_queue.all_finished[finished_layer]:
            return

        side = self.position_queue[finished_layer].side.clone()
        pv_values, pv_moves, _ = self.beam_state.backpropagate(finished_layer, side)
        if pv_values is None:
            return

        origin_step = max(self.step - self.D, 0)
        self.sample_buffer.assign_policy(origin_step, finished_layer, pv_moves)

        moves_to_apply = pv_moves[:, : self.pv_depth]
        self.position_queue.apply_moves_to_layer(finished_layer, moves_to_apply)

    def _expand_remaining_positions(self, legal_moves: LegalMoves ,move_pred: torch.Tensor):
        """Sample top-k legal moves and push them to create the next frontier."""
        move_data = legal_moves.rank_moves(
            move_pred, 
            ks=self.expansion_factors[self.beam_state.depth], 
            sample=False, 
            temp=3.0, 
            generator=self.generator
        )
        new_moves, board_idx, move_indices, ks = move_data

        self.beam_state = self.beam_state.expand(move_indices, ks)
        self.beam_boards = self.beam_boards.push(new_moves, board_idx)
        self.beam_state.store_moves(new_moves)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def run(self, max_iterations: int):
        """Convenience wrapper to execute a full self-play rollout."""
        for _ in range(max_iterations):
            self.step_once()
            if self.position_queue.all_game_over():
                break
