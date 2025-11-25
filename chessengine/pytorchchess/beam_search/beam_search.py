import torch
from dataclasses import dataclass

#=================================================================
# HELPER FUNCTIONS: MASKED MAX/MIN
#=================================================================
def masked_max(tensor, mask, dim):
    masked_tensor = tensor.masked_fill(mask, float('-inf'))
    return torch.max(masked_tensor, dim=dim)

def masked_min(tensor, mask, dim):
    masked_tensor = tensor.masked_fill(mask, float('inf'))
    return torch.min(masked_tensor, dim=dim)
    
#=================================================================
# BEAM SEARCH STATE CLASS
#=================================================================
@dataclass
class BeamSearchState:
    """Unified beam search state managing positions, evaluations, and moves
    
    Example:
    If exp_f = [4,3,2] then for a single board we have the expansion
    Layer | boards
      0   |   1  
      1   |   4  
      2   |   12 
      3   |   24

    i.e. choose top 4 moves for the original board, then top 3 for each of 4 and top 2 for each of 12. 
    The depth D = 3 (length of exp_f or plys played) and the number of layers is L = D + 1 = 4. 

    Backpropagation:
    If it's white to play on the original boards, then it's black in layer 1 and white again in layer 2. 
    - For each of the 12 boards in layer 2 white chooses the best two moves. 
    - We then get an evaluation from the model for each of the resulting 24 boards. 
    - For each of 12 boards choose the move (out of two) that led to the max evaluation. 
    This becoemse the evalution of of the 12 boards. 
    - We get 3 evaluations for each of the 4 boards in layer 1. 
    Choose the move the led to the min evaluation (black's turn!). 
    - In layer 0 choose move that leads to max evaluation out of 4.

    The resulting 3 moves are the principal value (PV) of the search. Once the beam search is concluded,
    and PV moves are obtained, we push one (or more) of the moves and start a search from the new position we get.

    Layer Interleaving:
    To achieve constant batch size we time the expansion one step apart
    Steps | boards
      0   |   1
      1   |   4     1   
      2   |   12    4     1
      3   |   24    12    4     1
      4   |   1     24    12    4
      5   |   4     1     24    12
      6   |   12    4     1     24

    (Going from 24 to 1 means we've backpropagated, pushed PV moves and starting a new search from the resulting position)
    Note the number of boards becomes constant 1 + 4 + 12 + 24 = 41 per step.
    Duplicating this pattern G times gives us a total of L x G = 4 x G simultaneous games (or beam searches)
    The batch size is B = 41 x G.  

    Main Complication:
    The branches of the search may not extend to full length, e.g. not enough legal moves or checkmate/stalemate has been reached. 
    The search must record evaluations for such early terminated branches, and the back-prop must take them into account correctly.  

    Evaluation tensor:
    For each of L x G simultaneous beam searches we need to keep 24 (= k0 x ... x kD-1) evals for each leaf of the search.
    If the branch terminates early the evaluation fills all the descendants, to simplify backprop. 

    Example 2: 
    G = 2, exp_f = [3,2]
    we get the tensors:

    ----------------------------------------------------------------------------------
    Step 0:
    BeamSearchState (B=2):
        Idx:   tensor([0, 0])
        Game:  tensor([0, 1])
        Layer: tensor([0, 0])
        Depth: tensor([0, 0])
    ----------------------------------------------------------------------------------
    Step 1:
    BeamSearchState (B=8):
        Idx:   tensor([0, 2, 4, 0, 2, 4, 0, 0])
        Game:  tensor([0, 0, 0, 1, 1, 1, 0, 1])
        Layer: tensor([0, 0, 0, 0, 0, 0, 1, 1])
        Depth: tensor([1, 1, 1, 1, 1, 1, 0, 0])
    ----------------------------------------------------------------------------------
    Step 2:
    BeamSearchState (B=20):
        Idx:   tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 2, 4, 0, 2, 4, 0, 0])
        Game:  tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        Layer: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        Depth: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0])
    ----------------------------------------------------------------------------------
    Step 3:
    BeamSearchState (B=20):
        Idx:   tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 2, 4, 0, 2, 4, 0, 0])
        Game:  tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        Layer: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])
        Depth: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0])
    ----------------------------------------------------------------------------------
    Step 4:
    BeamSearchState (B=20):
        Idx:   tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 2, 4, 0, 2, 4, 0, 0])
        Game:  tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        Layer: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 0])
        Depth: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0])
    ----------------------------------------------------------------------------------
    """
    
    # Core state tensors
    idx: torch.Tensor        # (B,) flat index into evaluation tensor
    game: torch.Tensor       # (B,) game index for each position  
    layer: torch.Tensor      # (B,) current layer in search tree
    depth: torch.Tensor      # (B,) current depth in search tree  # RENAMED from layer
    
    # Configuration
    G: int
    exp_f: torch.Tensor      # (D,) expansion factors [k0, k1, ..., kD-1]
    D: int                   # Number of depth levels (length of exp_f)
    L: int                   # Number of layers (D + 1), where D = len(exp_f)
    
    # Data storage tensors
    evaluations: torch.Tensor # (L, G * k0 * k1 * ... * kD-1) evaluations at each position
    moves: torch.Tensor       # (L, D, G * k0 * k1 * ... * kD-1) moves at each position
    
    # Constants
    MOVE_PAD: int = -1
    EVAL_PAD: float = 2.0
    DEBUG: bool = False  # Enable debugging output
    
    @property
    def exp_dim(self):
        """Dimension strides for flat indexing: [k0*k1*...*kD-1, k1*...*kD-1, ..., kD-1, 1]"""
        return torch.cat([
            torch.cumprod(self.exp_f.flip(0), dim=0).flip(0),
            torch.tensor([1], device=self.exp_f.device)
        ])
    
    @property 
    def game_stride(self):
        """Stride between games in flattened evaluation tensor"""
        return self.exp_f.prod().item()
    
    def __len__(self):
        """Number of positions in the current state"""
        return self.idx.size(0)
    
    @classmethod
    def initialize(cls, G, expansion_factors, debug: bool = False, device="cpu"):
        """Initialize a new BeamSearchState with given parameters"""
        exp_f = expansion_factors.clone().to(device=device, dtype=torch.long)
        D = len(exp_f)  
        L = D + 1  
        print(f"Initializing BeamSearchState: G={G}, D={D}, exp_f={exp_f.tolist()}")

        return cls(
            idx=torch.zeros(0, dtype=torch.long, device=device),
            game=torch.zeros(0, dtype=torch.long, device=device),
            layer=torch.zeros(0, dtype=torch.long, device=device), 
            depth=torch.zeros(0, dtype=torch.long, device=device),

            G=G,
            exp_f=exp_f,
            D=D,
            L=L,

            evaluations=torch.full((L, G, *exp_f), cls.EVAL_PAD, device=device).flatten(start_dim=1),  
            moves=torch.full((L, D, G, *exp_f), cls.MOVE_PAD, device=device).flatten(start_dim=2), 

            DEBUG=debug,
        )
    
    def __getitem__(self, key):
        """Index into position tensors"""
        return BeamSearchState(
            idx=self.idx[key],
            game=self.game[key], 
            layer=self.layer[key],
            depth=self.depth[key],
            G=self.G,
            exp_f=self.exp_f,
            D=self.D,
            L=self.L,
            evaluations=self.evaluations,  
            moves=self.moves,    
            DEBUG=self.DEBUG,          
        )
    
    def __repr__(self):
        """String representation of the BeamSearchState"""
        return (f"BeamSearchState (B={len(self)}):\n"
                f"  Idx:   {self.idx}\n" 
                f"  Game:  {self.game}\n"
                f"  Layer: {self.layer}\n" 
                f"  Depth: {self.depth}\n"
                )
    
    #==========================================================================
    # Position management
    #==========================================================================
    def add_new_layer(self, layer, finished_mask):
        # finished_mask: (G,) boolean mask for finished games at this layer
        new_idx = torch.zeros(self.G, dtype=torch.long, device=self.idx.device)
        new_game = torch.arange(self.G, dtype=torch.long, device=self.idx.device)
        new_layer = torch.full((self.G,), layer, dtype=torch.long, device=self.idx.device)
        new_depth = torch.full((self.G,), 0, dtype=torch.long, device=self.idx.device)

        new_idx = new_idx[~finished_mask]
        new_game = new_game[~finished_mask]
        new_layer = new_layer[~finished_mask]
        new_depth = new_depth[~finished_mask]

        self.idx = torch.cat([self.idx, new_idx])
        self.game = torch.cat([self.game, new_game])
        self.layer = torch.cat([self.layer, new_layer])
        self.depth = torch.cat([self.depth, new_depth])

    def expand(self, move_indices, ks=None):
        k = self.exp_f[self.depth] if ks is None else ks
        
        # Repeat all position attributes 
        expanded = self._repeat_interleave(k)
        expanded.depth += 1
        
        # Update indices based on selected moves
        exp_rep = expanded.exp_dim[expanded.depth]
        expanded.idx += move_indices * exp_rep
            
        return expanded
    
    def _repeat_interleave(self, repeats):
        return BeamSearchState(
            idx=self.idx.repeat_interleave(repeats),
            game=self.game.repeat_interleave(repeats),
            layer=self.layer.repeat_interleave(repeats),
            depth=self.depth.repeat_interleave(repeats),
            G=self.G,
            exp_f=self.exp_f,
            D=self.D,
            L=self.L,
            evaluations=self.evaluations,
            moves=self.moves,
            DEBUG=self.DEBUG,
        )
    
    #==========================================================================
    # Evaluation and Move management  
    #==========================================================================
    def store_final_evaluations(self, batch_mask, values):
        """Store evaluations for terminal positions"""
        flat_idx, layer = self._compute_flat_indices(batch_mask)
        self.evaluations[layer, flat_idx] = values

    def store_early_evaluations(self, batch_mask, values):  
        """
        Store evaluations for positions that terminated early (checkmate/stalemate).
        For each early-terminated position, fills all descendant leaves with the terminal value.
        
        Args:
            batch_mask: (B,) boolean mask selecting early terminated positions
            values: (N,) terminal values for the terminated positions  
        """
        assert batch_mask.sum() == values.size(0), f"Mask size {batch_mask.sum()} doesn't match values size {values.size(0)}"
        values = values.to(dtype=self.evaluations.dtype)
        if not batch_mask.any():
            return
            
        # Get indices and info for terminated positions
        flat_idx, layer = self._compute_flat_indices(batch_mask)
        depths = self.depth[batch_mask] 
        N = flat_idx.size(0)  # Number of terminated positions
        
        # Calculate expansion dimensions for each terminated layer
        exp_dim = self.exp_dim[depths]  
        exp_max = exp_dim.max().item()
        
        # Create range tensor and mask for valid descendants
        t = torch.arange(exp_max, device=values.device).expand(N, -1)  # (N, exp_max)
        mask = t < exp_dim.unsqueeze(1)  # (N, exp_max) - mask for valid descendants
        
        # Get indices for all valid descendants
        idx0, idx1 = mask.nonzero(as_tuple=True) # (N_exp,), (N_exp,)
        self.evaluations[layer[idx0], flat_idx[idx0] + idx1] = values[idx0]
        
    def store_moves(self, new_moves):
        """
        Store moves made from current positions after expansion.
        Must be called immediately after expand_positions().
        
        Args:
            new_moves: (B,) moves made from the expanded positions
        """
        assert new_moves.shape[0] == len(self), f"new_moves size {new_moves.size(0)} doesn't match current batch size {len(self)}"

        new_moves = new_moves.to(dtype=self.moves.dtype)
        B = new_moves.size(0)
        if B == 0:
            return
        
        flat_idx, layers = self._compute_flat_indices()
        
        # Vectorized expansion: for each move at layer l, fill all descendant slots
        exp_dim = self.exp_dim[self.depth]
        exp_max = exp_dim.max().item()
        
        # Create mask for valid expansions
        device = new_moves.device
        t = torch.arange(exp_max, device=device).expand(B, -1)  # (B, exp_max)
        mask = t < exp_dim.unsqueeze(1)  # (B, exp_max)
        idx0, idx1 = mask.nonzero(as_tuple=True)
        
        # Expand indices and moves according to mask
        layers = layers[idx0]
        depths = self.depth[idx0] - 1
        flat_idx = flat_idx[idx0] + idx1
        new_moves = new_moves[idx0]
        
        # Update moves tensor
        self.moves[layers, depths, flat_idx] = new_moves
    
    #=========================================================================
    # Backpropagation
    #=========================================================================
    def backpropagate(self, layer, side):
        """Backpropagate evaluations and return principal variation with target layer"""
        # Get evaluations for this stack
        evals = self.evaluations[layer].view(self.G, *self.exp_f) # (G, k0, k1, ..., kD-1)
        mask = evals == self.EVAL_PAD # Mask for invalid evals
        finished = (evals == 1).all() or (evals == -1).all()  # Check if all positions are terminal       
        if mask.all() or finished:
            print(f"Backpropagate: layer {layer}, mask = {mask.view(-1)}, finished = {finished}")
            print(f"Evals: {evals.view(-1)}")
            return None, None, None

        pv_values, pv_indices = self._minimax_backprop(evals, mask, side)

        self.eval_debug(layer, side, pv_values) # Debugging output
        
        # Get corresponding moves
        pv_moves = self._extract_pv_moves(layer, pv_indices)
        
        self.evaluations[layer] = self.EVAL_PAD  # Clear evaluations for this layer
        
        return pv_values, pv_moves, layer   # (G,), (G,D), 
    
    #==========================================================================
    # Utility methods
    #==========================================================================
    def _compute_flat_indices(self, selection=None):
        """Your existing compute_flat_indices logic"""
        if selection is None:
            flat_idx = self.idx + self.game_stride * self.game
            return flat_idx, self.layer
        
        selected_idx = self.idx[selection]
        selected_games = self.game[selection] 
        flat_idx = selected_idx + self.game_stride * selected_games
        selected_layers = self.layer[selection]
        return flat_idx, selected_layers
    
    #==========================================================================
    # Backprop helper functions
    #==========================================================================
    def _minimax_backprop(self, evals, mask, side):
        """
        Perform minimax backpropagation on game evaluations.
        
        Args:
            game_evals: (G, k0, k1, ..., kD-1) evaluation tensor
            side: bool or (G,) tensor indicating starting side for each game
            
        Returns:
            pv_values: (G,) final principal variation values
            pv_indices: (G, D) principal variation indices for each depth
        """
        D = self.exp_f.shape[0]
        G = evals.shape[0]
        
        # Handle side parameter - convert to tensor if needed
        side = side.to(dtype=torch.bool)
        assert side.shape == (G,), f"side tensor must have shape ({G},), got {side.shape}"
        
        # Shape current_side to broadcast with game_evals: (G, 1, 1, ..., 1)
        dims_to_add = evals.dim() - 2  # D - 1
        current_side = side.view(G, *([1] * dims_to_add))
        
        best_idx = []
        for d in range(D):
            # Compute both max and min for all positions
            max_vals, max_idx = masked_max(evals, mask, dim=-1)  # (G, k0, ..., kD-d-2)
            min_vals, min_idx = masked_min(evals, mask, dim=-1)  # (G, k0, ..., kD-d-2)
            
            # Select values and indices based on current_side for each game
            evals = torch.where(current_side, max_vals, min_vals)
            idx = torch.where(current_side, max_idx, min_idx)
            
            best_idx.append(idx)
            mask = mask.all(dim=-1) # (G, k0, ..., kD-d-2)
            
            # Flip sides and remove the last dimension for next iteration
            current_side = ~current_side.squeeze(-1)  # Remove one dimension and flip

        # Unravel the PV path to get indices for each depth
        pv_indices = self._unravel_pv_path(best_idx)
        
        return evals, pv_indices  # (G,), (G, D)

    def _unravel_pv_path(self, best_idx):
        """
        Given a list of best_idx as returned from minimax backprop, extract for each game
        the PV path (indices for each depth), handling multidimensional chains.
        
        Args:
            best_idx: list of tensors, length D.
              - best_idx[0]: (G, k0, ..., kD-2)
              - best_idx[1]: (G, k0, ..., kD-3)
              - ...
              - best_idx[-1]: (G,)
              
        Returns: 
            pv_idx: (G, D) tensor, so that pv_idx[i] = (i0, ..., iD-1) is the PV for game i.
        """
        D = len(best_idx)
        pv_idx = torch.zeros((self.G, D), dtype=torch.long, device=best_idx[-1].device) # (G, D)
        # print(f"best_idx: {[b.shape for b in best_idx]}")  # Debugging output
        # print(best_idx)
        for d in range(D): 
            idx_d = best_idx[-d-1] # (G, k0, ..., kd-1)
            previous = pv_idx[:, :d]  # (G, d)
            new = self._batched_index(idx_d, previous)
            pv_idx[:, d] = new.view(self.G)

        exp_dim = self.exp_dim[1:]  # (D,)
        pv_idx = (pv_idx * exp_dim).sum(-1)  # (G,)

        return pv_idx # (G,)

    def _batched_index(self, t, idx):
        """
        Batched indexing helper function.
        
        Args:
            t: tensor of shape [G, k0, ..., kD]
            idx: tensor of shape [G, D] containing indices
            
        Returns:
            Indexed values from t using idx
        """
        # t: shape [g, k0, ..., kD]
        # idx: shape [g, D]
        g, D = idx.shape
        g_idx = torch.arange(g, device=idx.device)
        return t[(g_idx, *[idx[:, d] for d in range(D)])]
    
    def _extract_pv_moves(self, layer, pv_indices):
        """
        Extract principal variation moves for a given stack and PV indices.
        
        Args:
            layer: layer to extract moves from
            pv_indices: (G, D) principal variation indices (flat indices)
            
        Returns:
            pv_moves: (G, D) moves along the principal variation
        """
        finished_moves = self.moves[layer]  # (D, G * k0 * ... * kD-1)
        finished_moves = finished_moves.view(self.D, self.G, -1)  # (D, G, k0 * ... * kD-1)
        
        g_idx = torch.arange(self.G, device=pv_indices.device)  # (G,)
        pv_moves = finished_moves[:, g_idx, pv_indices]  # (D, G)

        self.move_debug(layer, pv_moves) # Debugging output

        # Clean up processed moves
        self.moves[layer] = self.MOVE_PAD  # Clear moves for this layer
        
        return pv_moves.transpose(0, 1)  # Transpose to (G, D)
    
    def get_finished_expansion(self):
        """Get mask of positions that have reached maximum depth."""
        return (self.depth >= self.D).to(self.depth.device) # (B,)
    
    def get_finished_layer(self, current_step):
        """
        Get the stack ID that should be ready for backpropagation.
        
        Args:
            current_step: Current step/iteration number
            
        Returns:
            layer: layer ID of finished at current step
        """

        if current_step >= self.D:
            return (current_step - self.D) % self.L
        return None
    
    def eval_debug(self, layer, side, pv_values):
        if not self.DEBUG:
            return
        
        side_str = ["w" if s else "b" for s in side.bool().tolist()]
        print(f"Backpropagating layer {layer} ({side_str}) with evaluations:")
        
        evals_flat = self.evaluations[layer].view(self.G, -1)
        for g in range(self.G):
            print(f"  {evals_flat[g]}")
            print(f"  PV value: {pv_values[g]:.3f}")
        print()

    def move_debug(self, layer, pv_moves):
        if not self.DEBUG:
            return
        print(f"Finished moves for layer {layer}:")
        t = self.moves[layer].view(self.D, self.G, -1)
        t = torch.where(t != self.MOVE_PAD, t, -1)
        for g in range(self.G):
            print(t[:, g])
            print(f"PV moves: {pv_moves[:, g]}")
        print()