# ------------------------------------------------------------------
# Planned Features / To-Do List
# ------------------------------------------------------------------
# ✔️ is_game_over(): Checkmate, stalemate, or draw detection.
# ✔️ is_checkmate(), is_stalemate(), result(): Determine game termination type and winner.
# ✔️ draw by insufficient material: Handle edge cases like K vs. K, K+N vs. K, etc.
# ✔️ 50-move rule tracking: Optional half-move clock for pruning or training realism.
# ✔️ PreMove incremental update on push_move(): Avoid recomputing all pseudo-legal moves from scratch.
# ------------------------------------------------------------------
import torch
from .get_moves import GetMoves
from pytorchchess.utils import state_from_board, encode
import pytorchchess.utils.constants as const_legacy
import pytorchchess.utils.constants_new as const_new
from pytorchchess.state import GameState
from .to_chess_board import ToChessBoard
from .push_moves import PushMoves
from .stopping_condition import StoppingCondition

class BoardCache:
    def __init__(self):
        self.in_check_mask = None  # (B,) bool tensor from efficient generator
        self.no_move_mask = None   # (B,) bool tensor indicating boards with no moves
        self.legal_moves = None    # (B,) LegalMovesNew
        self.features = None       # (B, 8, 8, 21) cached feature tensor

    def select(self, idx: torch.Tensor):
        """
        Select a subset of the cache based on the provided indices.
        """
        new_cache = BoardCache()
        new_cache.in_check_mask = self.in_check_mask[idx].clone() if self.in_check_mask is not None else None
        new_cache.no_move_mask = self.no_move_mask[idx].clone() if self.no_move_mask is not None else None
        new_cache.legal_moves = self.legal_moves.select(idx).clone() if self.legal_moves is not None else None
        if self.features is not None:
            new_cache.features = self.features[idx].clone()
        return new_cache

class TorchBoard(GetMoves, ToChessBoard, PushMoves, StoppingCondition):
    def __init__(self, board_tensor: torch.Tensor, state: GameState, device = torch.device("cuda"), compute_cache: bool = False):
        assert isinstance(board_tensor, torch.Tensor), "board_tensor must be a torch.Tensor"
        assert board_tensor.shape[-2:] == (8, 8), "boards must be shape (B, 8, 8)"

        self.device = device
        const_legacy.move_constants_to(self.device)
        const_new.move_constants_to(self.device)

        if board_tensor.dim() == 2:
            board_tensor = board_tensor.unsqueeze(0)  # (1, 8, 8)

        self.board_tensor = board_tensor.to(self.device)
        self.state = state
        self.cache = BoardCache()

    @classmethod
    def from_board_list(cls, boards, device = torch.device("cuda")):
        board_tensor = encode(boards, device=device)  # (B, 8, 8)
        state = state_from_board(boards, device=device)

        return cls(board_tensor, state, device=device)
    
    def __repr__(self):
        bsz = self.board_tensor.shape[0]
        device = self.device
        # stm_str = self.state.side_to_move.tolist() if bsz < 10 else f"<{bsz} values>"
        # plys_str = self.state.plys.tolist() if bsz < 10 else f"<{bsz} values>"
        # castling_str = self.state.castling.tolist() if bsz < 10 else f"<{bsz} values>"
        cache_fields = []
        if hasattr(self, "cache") and self.cache:
            for field in ["in_check_mask", "no_move_mask", "features"]:
                value = getattr(self.cache, field, None)
                cache_fields.append(f"{field}={'yes' if value is not None else 'no'}")
            cache_str = ", ".join(cache_fields)
        else:
            cache_str = "no cache"
        return (f"TorchBoard(batch={bsz}, device={device}, "
                # f"side_to_move={stm_str}, plys={plys_str}, castling={castling_str}, "
                f"cache: {cache_str})")
    
    def concat(self, other):
        """
        Concatenate another TorchBoard to this one along the batch dimension.
        """
        assert isinstance(other, TorchBoard), "other must be an instance of TorchBoard"
        
        new_board_tensor = torch.cat((self.board_tensor, other.board_tensor), dim=0)
        new_state = self.state.cat(other.state)
        
        return TorchBoard(new_board_tensor, new_state, device=self.device)
    
    @property
    def side_to_move(self):
        return self.state.side_to_move # (B, 1) tensor with 1 for white, -1 for black

    @property
    def side(self):
        return self.state.side_to_move.view(-1) # (B,)
    
    @property
    def ep(self):
        return self.state.ep
    
    @property
    def batch_size(self):
        """
        Return the batch size of the board tensor.
        """
        return self.board_tensor.shape[0]
    
    def __len__(self):
        """
        Return the number of boards in the batch.
        """
        return self.board_tensor.shape[0]

    @property
    def in_check(self):
        if self.cache.in_check_mask is None:
            self.get_moves()
        if self.cache.in_check_mask is None:
            return torch.zeros((len(self), 1), dtype=torch.bool, device=self.device)
        return self.cache.in_check_mask.view(-1, 1)

    def to(self, device: str):
        """
        Move all board tensors to the specified device, update dtype, and clear cached check info.
        NOTE: Any constants used elsewhere should be moved to device at use time.
        """
        self.device = device
        self.board_tensor = self.board_tensor.to(device=device)
        self.state = self.state.to(device=device)

        self.cache = BoardCache()
        return self
    
    @property
    def board_flat(self):
        return self.board_tensor.view(self.board_tensor.size(0), 64)

    def feature_tensor(self):
        if self.cache.features is None:
            _, features = self.get_moves()
            if self.cache is not None:
                self.cache.features = features
            return features
        return self.cache.features
    
    # ------------------------------------------------------------------
    # Pre-moves -> Legal moves conversion
    # ------------------------------------------------------------------

    def rank_moves(self, move_pred, ks, sample=False, temp=1.0, generator=None):
        lm = self.cache.legal_moves
        if lm is None:
            lm, _ = self.get_moves()
            self.cache.legal_moves = lm
        return lm.rank_moves(move_pred, ks, sample, temp, generator)
    # ------------------------------------------------------------------
    # Clone and select
    # ------------------------------------------------------------------
    def clone(self):
        # Deep copy of board_tensor and state
        cloned_board = TorchBoard(self.board_tensor.clone(), self.state.clone(), self.device)
        cloned_board.invalidate_cache()
        return cloned_board

    def invalidate_cache(self):
        """
        Invalidate all cached derived data after the board changes.
        """
        self.cache = BoardCache()

    def select(self, idx: torch.Tensor):
        """
        Return new TorchBoard with only rows idx (like tensor slicing).
        """
        new_state = self.state[idx]
        new_board = self.board_tensor[idx].clone()
        new_torch_board = TorchBoard(new_board, new_state, self.device, compute_cache=False)
        new_torch_board.cache = self.cache.select(idx) if self.cache else BoardCache()
        return new_torch_board

    def select_without_clone(self, idx: torch.Tensor):
        """
        Return new TorchBoard with only rows idx (like tensor slicing) but WITHOUT cloning.
        This creates a view that shares memory with the original TorchBoard.
        Modifications to the returned TorchBoard will affect the original.
        """
        new_state = GameState(
            side_to_move   = self.state.side_to_move[idx],  # No clone()
            plys           = self.state.plys[idx],           # No clone()
            castling       = self.state.castling[idx],       # No clone()
            ep             = self.state.ep[idx],             # No clone()
            fifty_move_clock = self.state.fifty_move_clock[idx],  # No clone()
            previous_moves = self.state.previous_moves[idx], # No clone()
            position_history = self.state.position_history[idx] if self.state.position_history is not None else None  # No clone()
        )
        new_torch_board = TorchBoard(
            self.board_tensor[idx],  # No clone() - this is a view
            new_state, 
            self.device, 
            compute_cache=False
        )
        # Note: We don't copy cache since we're creating a view
        new_torch_board.cache = BoardCache()
        return new_torch_board

    def __getitem__(self, idx):
        """
        Implement indexing syntax board[idx] that returns a view (no cloning).
        This is more Pythonic than select_without_clone().
        """
        return self.select_without_clone(idx)
