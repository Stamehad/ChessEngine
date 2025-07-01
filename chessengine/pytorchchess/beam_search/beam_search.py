import torch
from dataclasses import dataclass

@dataclass
class BeamSearchState:
    """Unified beam search state managing positions, evaluations, and moves"""
    
    # Core state tensors
    idx: torch.Tensor        # (B,) flat index into evaluation tensor
    game: torch.Tensor       # (B,) game index for each position  
    stack: torch.Tensor      # (B,) stack/step index
    depth: torch.Tensor      # (B,) current depth in search tree  # RENAMED from layer
    
    # Configuration
    num_games: int
    exp_f: torch.Tensor      # (D,) expansion factors [k0, k1, ..., kD-1]
    
    # Data storage tensors
    evaluations: torch.Tensor  # (S, G, n0, n1, ..., nD-1) terminal evaluations
    moves: torch.Tensor        # (S, D, G, n0, n1, ..., nD-1) moves at each position
    eval_stacks: torch.Tensor  # (S,) which stacks have evaluations
    move_stacks: torch.Tensor  # (S,) which stacks have moves
    
    # Constants
    MOVE_PAD: int = 2**15
    EVAL_PAD: float = 0.0
    
    @property
    def exp_dim(self):
        """Dimension strides for flat indexing: [n0*n1*...*nD-1, n1*...*nD-1, ..., nD-1, 1]"""
        return torch.cat([
            torch.cumprod(self.exp_f.flip(0), dim=0).flip(0),
            torch.tensor([1], device=self.exp_f.device)
        ])
    
    @property 
    def game_stride(self):
        """Stride between games in flattened evaluation tensor"""
        return self.exp_f.prod().item()
    
    @classmethod
    def create(cls, num_games, expansion_factors, device="cpu"):
        """Factory method to create initial state"""
        exp_f = expansion_factors.clone().to(device=device, dtype=torch.long)
        
        return cls(
            # Initial positions (one per game at root)
            idx=torch.zeros(num_games, dtype=torch.long, device=device),
            game=torch.arange(num_games, dtype=torch.long, device=device), 
            stack=torch.zeros(num_games, dtype=torch.long, device=device),
            depth=torch.zeros(num_games, dtype=torch.long, device=device),
            
            # Config
            num_games=num_games,
            exp_f=exp_f,
            
            # Storage with initial capacity (size 1, like original Moves)
            evaluations=torch.zeros((0, num_games, *exp_f), device=device),  # Keep 0 for evals
            moves=torch.full((1, len(exp_f), num_games, *exp_f), cls.MOVE_PAD, device=device),  # Size 1!
            eval_stacks=torch.zeros((0,), dtype=torch.long, device=device),  # Keep 0 for evals
            move_stacks=torch.zeros((1,), dtype=torch.long, device=device),  # Size 1!
        )
    
    # Position management
    def expand_positions(self, move_indices, ks=None):
        """Expand current positions by selected moves"""
        #k = self.exp_f[self.depth]  # expansion factor per position
        k = self.exp_f[self.depth] if ks is None else ks
        
        # Repeat all position attributes 
        expanded = self._repeat_interleave(k)
        expanded.depth += 1
        
        # Update indices based on selected moves
        if move_indices is not None:
            exp_rep = expanded.exp_dim[expanded.depth]
            idx_rep = self.idx.repeat_interleave(k)
            expanded.idx = move_indices * exp_rep + idx_rep
            
        return expanded
    
    def add_new_stack(self, stack_id):
        """Add new root positions for next search iteration"""
        new_positions = self.__class__.create(self.num_games, self.exp_f, device=self.idx.device)
        new_positions.stack.fill_(stack_id)
        new_positions.move_stacks.fill_(stack_id)  # Set the stack ID for the pre-allocated storage
        return self + new_positions
    
    # Evaluation management  
    def store_terminal_evaluations(self, position_mask, values):
        """Store evaluations for terminal positions"""
        flat_idx, stacks = self._compute_flat_indices(position_mask)
        
        # Create new evaluation tensor
        new_evals = torch.zeros((1, self.num_games, *self.exp_f), device=values.device)
        new_evals = new_evals.flatten(start_dim=1)  # (1, G*n0*n1*...*nD-1)
        new_evals[0, flat_idx] = values
        new_evals = new_evals.view(1, self.num_games, *self.exp_f)
        
        # Append to storage
        self.evaluations = torch.cat([self.evaluations, new_evals], dim=0)
        self.eval_stacks = torch.cat([self.eval_stacks, stacks.unique()], dim=0)

        self._reduce_eval_stacks()  # Consolidate duplicate stacks if needed
        
    def store_early_terminated_evaluations(self, position_mask, values):
        """
        Store evaluations for positions that terminated early (checkmate/stalemate).
        For each early-terminated position, fills all descendant leaves with the terminal value.
        
        Args:
            position_mask: (B,) boolean mask selecting early terminated positions
            values: (N,) terminal values for the terminated positions  
        """
        if not position_mask.any():
            return
            
        # Get indices and info for terminated positions
        flat_idx, stacks = self._compute_flat_indices(position_mask)
        depths = self.depth[position_mask] 
        N = flat_idx.size(0)  # Number of terminated positions
        
        # Calculate expansion dimensions for each terminated layer
        exp_dim = self.exp_dim[depths]  
        exp_max = exp_dim.max().item()
        
        # Create range tensor and mask for valid descendants
        device = values.device
        t = torch.arange(exp_max, device=device).expand(N, -1)  # (N, exp_max)
        mask = t < exp_dim.unsqueeze(1)  # (N, exp_max) - mask for valid descendants
        
        # Get indices for all valid descendants
        idx0, idx1 = mask.nonzero(as_tuple=True)
        flat_idx_expanded = flat_idx[idx0] + idx1  # Flat indices for all descendants
        values_expanded = values[idx0]  # Corresponding values
        
        # Create new evaluation tensor with N entries (one per terminated position)
        new_evals = torch.zeros((N, self.num_games, *self.exp_f), device=device, dtype=values.dtype)
        new_evals_flat = new_evals.view(N, -1)  # (N, G*n0*n1*...*nD-1)
        new_evals_flat[idx0, flat_idx_expanded] = values_expanded
        new_evals = new_evals_flat.view(N, self.num_games, *self.exp_f)
        
        # Append to storage
        self.evaluations = torch.cat([self.evaluations, new_evals], dim=0)
        self.eval_stacks = torch.cat([self.eval_stacks, stacks], dim=0)
        
        # Reduce/consolidate duplicate stacks if needed
        self._reduce_eval_stacks()

    def _reduce_eval_stacks(self):
        """
        Consolidate evaluations that belong to the same stack by summing them.
        Similar to EvalStates.reduce_stack()
        """
        if len(self.eval_stacks) <= 1:
            return
        
        unique_stacks, inverse_indices = torch.unique(self.eval_stacks, return_inverse=True)
        
        if len(unique_stacks) == len(self.eval_stacks):
            return  # No duplicates
        
        # Sum evaluations for duplicate stacks
        from torch_scatter import scatter
        consolidated_evals = scatter(self.evaluations, inverse_indices, dim=0, reduce='sum')
        
        self.evaluations = consolidated_evals
        self.eval_stacks = unique_stacks

    def store_moves(self, new_moves):
        """
        Store moves made from current positions after expansion.
        Must be called immediately after expand_positions().
        
        Args:
            new_moves: (B,) moves made from the expanded positions
        """
        B = new_moves.size(0)
        
        # Validation assertions
        assert B == self.idx.size(0), f"new_moves size {B} doesn't match current batch size {self.idx.size(0)}"
        assert torch.all(self.depth > 0), "store_moves() can only be called after expand_positions() (depth > 0)"  # CHANGED
        assert new_moves.device == self.idx.device, f"Device mismatch: new_moves on {new_moves.device}, expected {self.idx.device}"
        
        if B == 0:
            return
        
        # Get current state
        flat_idx, stacks = self._compute_flat_indices()
        
        # Normalize stacks
        min_stack = self.move_stacks.min().item()
        normalized_stacks = stacks - min_stack
        
        # Vectorized expansion: for each move at layer l, fill all descendant slots
        exp_dim = self.exp_dim[self.depth]  # CHANGED
        exp_max = exp_dim.max().item()
        
        # Create mask for valid expansions
        device = new_moves.device
        t = torch.arange(exp_max, device=device).expand(B, -1)  # (B, exp_max)
        mask = t < exp_dim.unsqueeze(1)  # (B, exp_max)
        idx0, idx1 = mask.nonzero(as_tuple=True)
        
        # Expand indices and moves according to mask
        expanded_stacks = normalized_stacks[idx0]
        expanded_depths = self.depth[idx0] - 1
        expanded_flat_idx = flat_idx[idx0] + idx1
        expanded_moves = new_moves[idx0]
        
        # Update moves tensor
        self.moves = self.moves.flatten(start_dim=2)  # (S, D, G * n0 * n1 * ... * nD-1)
        self.moves[expanded_stacks, expanded_depths, expanded_flat_idx] = expanded_moves
        
        # Reshape back to original structure
        self.moves = self.moves.view(-1, len(self.exp_f), self.num_games, *self.exp_f)

    # Backpropagation
    def backpropagate(self, stack_id, side=True):
        """Backpropagate evaluations and return principal variation with target layer"""
        # Get evaluations for this stack
        assert isinstance(stack_id, int), "stack_id must be an integer"
        assert stack_id >= 0, "stack_id must be a non-negative integer"
        stack_mask = self.eval_stacks == stack_id
        if not stack_mask.any():
            return None, None, None
            
        game_evals = self.evaluations[stack_mask].sum(dim=0)  # (G, n0, n1, ..., nD-1)
        
        pv_values, pv_indices = self._minimax_backprop(game_evals, side)
        
        # Get corresponding moves
        pv_moves = self._extract_pv_moves(stack_id, pv_indices)
        
        # Calculate target layer for PV application
        D = len(self.exp_f)
        target_layer = stack_id % (D + 1)
        
        # Clean up processed evaluations
        self.evaluations = self.evaluations[~stack_mask]
        self.eval_stacks = self.eval_stacks[~stack_mask]
        
        return pv_values, pv_moves, target_layer
    
    # Utility methods
    def _compute_flat_indices(self, selection=None):
        """Your existing compute_flat_indices logic"""
        if selection is None:
            flat_idx = self.idx + self.game_stride * self.game
            return flat_idx, self.stack
        
        selected_idx = self.idx[selection]
        selected_games = self.game[selection] 
        flat_idx = selected_idx + self.game_stride * selected_games
        selected_stacks = self.stack[selection]
        return flat_idx, selected_stacks
    
    def _repeat_interleave(self, repeats):
        """Create new state with positions repeated"""
        return BeamSearchState(
            idx=self.idx.repeat_interleave(repeats),
            game=self.game.repeat_interleave(repeats),
            stack=self.stack.repeat_interleave(repeats), 
            depth=self.depth.repeat_interleave(repeats),
            num_games=self.num_games,
            exp_f=self.exp_f,
            evaluations=self.evaluations,
            moves=self.moves,
            eval_stacks=self.eval_stacks,
            move_stacks=self.move_stacks,
        )
    
    # Standard dataclass operations
    def __getitem__(self, key):
        """Index into position tensors"""
        return BeamSearchState(
            idx=self.idx[key],
            game=self.game[key], 
            stack=self.stack[key],
            depth=self.depth[key],
            num_games=self.num_games,
            exp_f=self.exp_f,
            evaluations=self.evaluations,  # Shared
            moves=self.moves,              # Shared  
            eval_stacks=self.eval_stacks,  # Shared
            move_stacks=self.move_stacks,  # Shared
        )
    
    def __add__(self, other):
        """Concatenate position tensors"""
        return BeamSearchState(
            idx=torch.cat([self.idx, other.idx]),
            game=torch.cat([self.game, other.game]),
            stack=torch.cat([self.stack, other.stack]),
            depth=torch.cat([self.depth, other.depth]),
            num_games=self.num_games,
            exp_f=self.exp_f,
            # Merge storage tensors
            evaluations=torch.cat([self.evaluations, other.evaluations], dim=0),
            moves=torch.cat([self.moves, other.moves], dim=0),
            eval_stacks=torch.cat([self.eval_stacks, other.eval_stacks]),
            move_stacks=torch.cat([self.move_stacks, other.move_stacks]),
        )
    
    def _minimax_backprop(self, game_evals, side=True):
        """
        Perform minimax backpropagation on game evaluations.
        
        Args:
            game_evals: (G, n0, n1, ..., nD-1) evaluation tensor
            side: bool or (G,) tensor indicating starting side for each game
            
        Returns:
            pv_values: (G,) final principal variation values
            pv_indices: (G, D) principal variation indices for each depth
        """
        D = self.exp_f.shape[0]
        g = game_evals.shape[0]
        
        # Handle side parameter - convert to tensor if needed
        if isinstance(side, bool):
            side_tensor = torch.full((g,), side, dtype=torch.bool, device=game_evals.device)
        else:
            side_tensor = side.to(device=game_evals.device, dtype=torch.bool)
            assert side_tensor.shape == (g,), f"side tensor must have shape ({g},), got {side_tensor.shape}"
        
        # Shape current_side to broadcast with game_evals: (G, 1, 1, ..., 1)
        dims_to_add = game_evals.dim() - 2  # D - 1
        current_side = side_tensor.view(g, *([1] * dims_to_add))
        
        best_idx = []
        for k in range(D):
            # Compute both max and min for all positions
            max_vals, max_idx = torch.max(game_evals, dim=-1)  # (G, n0, ..., nD-k-2)
            min_vals, min_idx = torch.min(game_evals, dim=-1)  # (G, n0, ..., nD-k-2)
            
            # Select values and indices based on current_side for each game
            game_evals = torch.where(current_side, max_vals, min_vals)
            idx = torch.where(current_side, max_idx, min_idx)
            
            best_idx.append(idx)
            
            # Flip sides and remove the last dimension for next iteration
            current_side = ~current_side.squeeze(-1)  # Remove one dimension and flip

        # Unravel the PV path to get indices for each depth
        pv_indices = self._unravel_pv_path(best_idx)
        
        return game_evals, pv_indices  # (G,), (G, D)

    def _unravel_pv_path(self, best_idx):
        """
        Given a list of best_idx as returned from minimax backprop, extract for each game
        the PV path (indices for each depth), handling multidimensional chains.
        
        Args:
            best_idx: list of tensors, length D.
              - best_idx[0]: (G, n0, ..., nD-2)
              - best_idx[1]: (G, n0, ..., nD-3)
              - ...
              - best_idx[-1]: (G,)
              
        Returns: 
            pv_idx: (G, D) tensor, so that pv_idx[i] = (i0, ..., iD-1) is the PV for game i.
        """
        D = len(best_idx)
        g = self.num_games
        pv_idx = torch.zeros((g, D), dtype=torch.long, device=best_idx[-1].device) # (G, D)

        for d in range(D): 
            idx_d = best_idx[-d-1] # (G, n0, ..., nl-1)
            previous = pv_idx[:, :d]  # (G, l)
            new = self._batched_index(idx_d, previous)
            pv_idx[:, d] = new.view(g)

        exp_dim = self.exp_dim[1:]  # (D,)
        pv_idx = (pv_idx * exp_dim).sum(-1)  # (G,)

        return pv_idx # (G,)

    def _batched_index(self, t, idx):
        """
        Batched indexing helper function.
        
        Args:
            t: tensor of shape [G, n0, ..., nD]
            idx: tensor of shape [G, D] containing indices
            
        Returns:
            Indexed values from t using idx
        """
        # t: shape [g, n0, ..., nD]
        # idx: shape [g, D]
        g, D = idx.shape
        batch_indices = torch.arange(g, device=idx.device)
        return t[(batch_indices, *[idx[:, d] for d in range(D)])]
    
    def _extract_pv_moves(self, stack_id, pv_indices):
        """
        Extract principal variation moves for a given stack and PV indices.
        
        Args:
            stack_id: Stack identifier
            pv_indices: (G, D) principal variation indices (flat indices)
            
        Returns:
            pv_moves: (G, D) moves along the principal variation
        """
        # Find moves for this stack
        stack_mask = self.move_stacks == stack_id
        if not stack_mask.any():
            return None
        
        finished_moves = self.moves[stack_mask]  # (N_finished, D, G, n0, ..., nD-1)

        # Assert exactly one stack matches (beam search invariant)
        assert finished_moves.size(0) == 1, f"Expected exactly 1 stack match, got {finished_moves.size(0)}"
        finished_moves = finished_moves.squeeze(0)  # (D, G, n0, ..., nD-1)
    
        
        # Clean up processed moves (like in original finished_moves)
        self.moves = self.moves[~stack_mask]
        self.move_stacks = self.move_stacks[~stack_mask]
        
        finished_moves = finished_moves.flatten(start_dim=2)  # (D, G, n0 * ... * nD-1)
        g_idx = torch.arange(self.num_games, device=pv_indices.device)  # (G,)
        pv_moves = finished_moves[:, g_idx, pv_indices]  # (D, G)
        pv_moves = pv_moves.transpose(0, 1)  # Transpose to (G, D)
        
        return pv_moves
    
    def get_finished_expansion(self):
        """Get mask of positions that have reached maximum depth."""
        return self.depth >= len(self.exp_f)
    
    def get_finished_stack(self, current_step):
        """
        Get the stack ID that should be ready for backpropagation.
        
        Args:
            current_step: Current step/iteration number
            
        Returns:
            stack_id: Stack ID ready for backprop, or None if not ready
        """
        D = len(self.exp_f)
        if current_step >= D:
            return current_step - D
        return None