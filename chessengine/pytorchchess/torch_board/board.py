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
from .in_check import InCheck
from pytorchchess.utils import state_from_board, encode
from pytorchchess.state import GameState, LegalMoves
from .to_chess_board import ToChessBoard
from .push_moves import PushMoves
from .feature_tensor import FeatureTensor

class BoardCache:
    def __init__(self):
        self.check_info = None
        self.pre_moves = None
        self.legal_moves = None
        self.attack_map = None

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

class TorchBoard(PseudoMoveGenerator, InCheck, ToChessBoard, PushMoves, FeatureTensor):
    def __init__(self, board_tensor: torch.Tensor, state: GameState, device: str = "cuda", compute_cache: bool = True):
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
    def from_board_list(cls, boards, device: str = "cuda", LAYER=False):
        board_tensor = encode(boards, device=device)  # (B, 8, 8)
        state = state_from_board(boards, device=device)
        if LAYER:
            state.layer = torch.zeros(board_tensor.shape[0], dtype=torch.int64, device=device)  # (B,)

        return cls(board_tensor, state, device=device)
    
    def __repr__(self):
        bsz = self.board_tensor.shape[0]
        device = self.device
        layers = self.state.layer.unique().tolist() if self.state.layer is not None else None
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
        return (f"TorchBoard(batch={bsz}, device={device}, layers={layers}, "
                # f"side_to_move={stm_str}, plys={plys_str}, castling={castling_str}, "
                f"cache: {cache_str})")
    
    def concat(self, other):
        """
        Concatenate another TorchBoard to this one along the batch dimension.
        """
        assert isinstance(other, TorchBoard), "other must be an instance of TorchBoard"

        if isinstance(self.state.layer, torch.Tensor) and isinstance(other.state.layer, torch.Tensor):
            new_layer = torch.cat((self.state.layer, other.state.layer), dim=0)
        else:
            new_layer = None
        
        new_board_tensor = torch.cat((self.board_tensor, other.board_tensor), dim=0)
        new_state = GameState(
            side_to_move   = torch.cat((self.state.side_to_move, other.state.side_to_move), dim=0),
            plys           = torch.cat((self.state.plys, other.state.plys), dim=0),
            castling       = torch.cat((self.state.castling, other.state.castling), dim=0),
            ep             = torch.cat((self.state.ep, other.state.ep), dim=0),
            previous_moves = torch.cat((self.state.previous_moves, other.state.previous_moves), dim=0),
            layer          = new_layer
        )
        
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
        return self.state.side_to_move
    
    @property
    def ep(self):
        return self.state.ep

    def to(self, device: str):
        """
        Move all board tensors to the specified device, update dtype, and clear cached check info.
        NOTE: Any constants used elsewhere should be moved to device at use time.
        """
        self.device = device

        self.board_tensor = self.board_tensor.to(device=device)
        self.state.side_to_move = self.state.side_to_move.to(device)
        self.state.plys = self.state.plys.to(device)
        self.state.castling = self.state.castling.to(device)
        self.state.ep = self.state.ep.to(device)

        self.cache.check_info = None
        return self
    
    @property
    def board_flat(self):
        return self.board_tensor.view(self.board_tensor.size(0), 64)
    
    # @property
    # def result_info(self):
    #     return GameResult(self)
    
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

        return LegalMoves.from_premoves(premoves, self.board_flat.shape[0]) # (B, L_max)
    
    def get_legal_moves(self, get_tensor: bool = False):
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
    
    def get_topk_legal_moves(self, move_pred, ks):
        lm = self.get_legal_moves(get_tensor=True)  # LegalMoves
        return lm.rank_moves(move_pred, ks)
    
    # ------------------------------------------------------------------
    # Clone and select
    # ------------------------------------------------------------------
    def clone(self):
        # Deep copy of board_tensor and state
        new_state = GameState(
            side_to_move=self.state.side_to_move.clone(),
            plys=self.state.plys.clone(),
            castling=self.state.castling.clone(),
            ep=self.state.ep.clone(),
            previous_moves=self.state.previous_moves.clone(),
            layer=self.state.layer.clone() if self.state.layer is not None else None,
        )
        cloned_board = TorchBoard(self.board_tensor.clone(), new_state, self.device)
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
        new_state = GameState(
            side_to_move   = self.state.side_to_move[idx].clone(),
            plys           = self.state.plys[idx].clone(),
            castling       = self.state.castling[idx].clone(),
            ep             = self.state.ep[idx].clone(),
            previous_moves = self.state.previous_moves.long()[idx].clone().to(torch.uint16),
            layer          = self.state.layer[idx].clone() if self.state.layer is not None else None
        )
        new_torch_board = TorchBoard(self.board_tensor[idx].clone(), new_state, self.device, compute_cache=False)
        new_torch_board.cache = self.cache.select(idx) if self.cache else BoardCache()
        return new_torch_board

    # ------------------------------------------------------------
    #  finished-mask, result strings, select()
    # ------------------------------------------------------------
    def is_game_over(self):
        """
        Returns (B,) bool tensor: True where board is checkmate or stalemate.
        Side to move is already updated in self.state.
        """
        lm = self.get_legal_moves()          # LegalMoves
        no_moves = lm.get_terminal_boards()     # (B,) bool tensor
        in_check = self.in_check                # (B,)
        # 1 = white win, 0 = draw, -1 = black win
        result = torch.where(
            self.side_to_move[no_moves] == 1,
            torch.where(in_check[no_moves], 1, 0),  
            torch.where(in_check[no_moves], -1, 0)
        ) # (N_over,)
        
        return no_moves, result # (B,) bool, (N_over,) int tensor                         

    def result_strings(self):
        mask = self.is_game_over()
        res = []
        in_check = self.in_check.cpu()
        for done, chk, stm in zip(mask.cpu(), in_check, self.side_to_move.cpu()):
            if not done:
                res.append("*")
            else:
                if chk:
                    res.append("0-1" if stm else "1-0")  # stm==1 => white, else black
                else:
                    res.append("1/2-1/2")
        return res                                  # list length B