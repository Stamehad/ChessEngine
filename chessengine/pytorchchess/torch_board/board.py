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
from .pseudo_move_gen import PseudoMoveGenerator
from .pseudo_move_gen_new import PseudoMoveGeneratorNew
from .in_check import InCheck
from pytorchchess.utils import state_from_board, encode
from pytorchchess.state import GameState, LegalMoves
from .to_chess_board import ToChessBoard
from .push_moves import PushMoves
from .feature_tensor import FeatureTensor
from .stopping_condition import StoppingCondition

class BoardCache:
    def __init__(self):
        self.check_info = None  # CheckInfo object containing in_check, pin_data, and check_data
        self.pre_moves = None   # PreMoves object containing pseudo-legal moves
        self.legal_moves = None # LegalMoves object containing filtered legal moves
        self.attack_map = None  # Attack map tensor (B, 64) indicating attacked squares for each board

    def select(self, idx: torch.Tensor):
        """
        Select a subset of the cache based on the provided indices.
        """
        new_cache = BoardCache()
        new_cache.check_info = self.check_info.select(idx).clone() if self.check_info else None
        new_cache.pre_moves = self.pre_moves.select(idx).clone() if self.pre_moves else None
        new_cache.legal_moves = self.legal_moves.select(idx).clone() if self.legal_moves else None
        new_cache.attack_map = self.attack_map[idx].clone() if self.attack_map is not None else None
        return new_cache

class TorchBoard(
        PseudoMoveGenerator, 
        PseudoMoveGeneratorNew, 
        InCheck, 
        ToChessBoard, 
        PushMoves, 
        FeatureTensor, 
        StoppingCondition
    ):
    def __init__(self, board_tensor: torch.Tensor, state: GameState, device = torch.device("cuda"), compute_cache: bool = True):
        assert isinstance(board_tensor, torch.Tensor), "board_tensor must be a torch.Tensor"
        assert board_tensor.shape[-2:] == (8, 8), "boards must be shape (B, 8, 8)"

        self.device = device

        if board_tensor.dim() == 2:
            board_tensor = board_tensor.unsqueeze(0)  # (1, 8, 8)

        self.board_tensor = board_tensor.to(self.device)
        self.state = state
        if compute_cache:
            self.cache = BoardCache()
            self.cache.check_info = self.compute_check_info()

    @classmethod
    def from_board_list(cls, boards, device = torch.device("cuda")):
        board_tensor = encode(boards, device=device)  # (B, 8, 8)
        state = state_from_board(boards, device=device)

        return cls(board_tensor, state, device=device)
    
    def __repr__(self):
        bsz = self.board_tensor.shape[0]
        device = self.device
        stm_str = self.state.side_to_move.tolist() if bsz < 10 else f"<{bsz} values>"
        plys_str = self.state.plys.tolist() if bsz < 10 else f"<{bsz} values>"
        castling_str = self.state.castling.tolist() if bsz < 10 else f"<{bsz} values>"
        cache_fields = []
        if hasattr(self, "cache") and self.cache:
            for field in ["check_info", "pre_moves", "legal_moves", "attack_map"]:
                value = getattr(self.cache, field, None)
                if value is not None:
                    cache_fields.append(f"{field}=yes")
                else:
                    cache_fields.append(f"{field}=no")
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
    
    def _get_check_info(self):
        if self.cache.check_info is None:
            self.cache.check_info = self.compute_check_info(self)
        return self.cache.check_info

    @property
    def in_check(self):
        return self._get_check_info().in_check
    
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

    def to(self, device: str):
        """
        Move all board tensors to the specified device, update dtype, and clear cached check info.
        NOTE: Any constants used elsewhere should be moved to device at use time.
        """
        self.device = device
        self.board_tensor = self.board_tensor.to(device=device)
        self.state = self.state.to(device=device)

        self.cache.check_info = None
        return self
    
    @property
    def board_flat(self):
        return self.board_tensor.view(self.board_tensor.size(0), 64)
    
    # ------------------------------------------------------------------
    # Pre-moves -> Legal moves conversion
    # ------------------------------------------------------------------

    def filter_legal_moves_from_cache(self):
        premoves = self.cache.pre_moves.clone()
        if self.cache.check_info is None:
            self.cache.check_info = self.compute_check_info()
        premoves.filter_by_pin(self.cache.check_info.pin_data)
        premoves.filter_by_check(self.cache.check_info.check_data)
        attack_map = self.kings_disallowed_squares()
        premoves.filter_king_by_attacks(attack_map)
        premoves.filter_empty()

        return LegalMoves.from_premoves(premoves, len(self)) # (B, L_max)
    
    def get_legal_moves(self, get_tensor: bool = False):
        if not hasattr(self, "cache"):
            self.cache = BoardCache()
            self.cache.check_info = self.compute_check_info()

        if self.cache.legal_moves:
            if get_tensor and self.cache.legal_moves.tensor is None:
                self.cache.legal_moves.get_tensor(self.board_flat)
            return self.cache.legal_moves

        if self.cache.pre_moves is None:
            self.get_pre_moves()
        
        self.cache.legal_moves = self.filter_legal_moves_from_cache()
        if get_tensor:
            self.cache.legal_moves.get_tensor(self.board_flat)
        return self.cache.legal_moves
    
    def get_legal_moves_new(self):
        premoves, in_check = self.get_moves() # PreMoves
        return premoves, in_check, LegalMoves.from_premoves(premoves, len(self))  # Update board_tensor and state
    
    def get_topk_legal_moves(self, move_pred, ks, sample=False, temp=1.0, generator=None):
        lm = self.get_legal_moves(get_tensor=True)  # LegalMoves
        return lm.rank_moves(move_pred, ks, sample=sample, temp=temp, generator=generator)
    
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