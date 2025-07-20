import torch
from dataclasses import dataclass
from .check_info import PinData, CheckData
from pytorchchess.utils.utils import get_check_blockers, squares_to_int
from pytorchchess.utils.constants import PROMOTION_MASK

@dataclass
class Pieces:
    sq: torch.Tensor        # (N_pieces,)
    id: torch.Tensor        # (N_pieces,) 1-12
    board: torch.Tensor     # (N_pieces,)

    def is_empty(self):
        return self.sq.numel() == 0

@dataclass
class PreMoves:
    """
    PreMoves special values:
    0: no move
    1: normal move
    5: en passant capture or double pawn push
    6: king side castling
    7: queen side castling
    8: spurious move (e.g. used for double-pawn-push check mask)
    10: promotion 
    """
    moves: torch.Tensor      # (N_moves, 64)
    sq: torch.Tensor         # (N_moves,)
    id: torch.Tensor         # (N_moves,) 1-12
    board: torch.Tensor      # (N_moves,)

    def __len__(self):
        return self.moves.shape[0]
    
    def clone(self):
        return PreMoves(
            moves=self.moves.clone(),
            sq=self.sq.clone(),
            id=self.id.clone(),
            board=self.board.clone()
        )
    
    def is_empty(self):
        return self.sq.numel() == 0

    def filter_by_pin(self, pin: PinData):

        if pin.king_sq.numel() == 0:
            return
        # Match moves to pinned pieces by (piece_sq, board)
        move_sq = self.sq.unsqueeze(1)       # (N_moves, 1)
        move_board = self.board.unsqueeze(1)       # (N_moves, 1)
        pinned_sq = pin.pinned_piece_sq.unsqueeze(0)  # (1, N_pin)
        pinned_board = pin.board.unsqueeze(0)          # (1, N_pin)

        match = (move_sq == pinned_sq) & (move_board == pinned_board)  # (N_moves, N_pin)
        move_idx, pin_idx = match.nonzero(as_tuple=True)  # (M,)

        ray_masks = pin.pin_mask()[pin_idx]    # (M, 64)
        self.moves[move_idx] *= ray_masks  # Apply ray mask to allowed moves


    def filter_by_check(self, check: CheckData):
        if check.king_sq.numel() == 0:
            return 
        
        unique_boards, masks = check.reduced_check_mask()   # (N_single,), (N_single, 64)
        move_board = self.board.unsqueeze(1)                # (N_moves, 1)
        unique_boards = unique_boards.unsqueeze(0)          # (1, N_single)

        match = move_board == unique_boards  # (N_moves, N_single)
        if not match.any():
            return  # No matches to filter

        move_idx, check_idx = match.nonzero(as_tuple=True)  # (M,), (M,)
        is_king = self.id[move_idx] % 6 == 0  # id = 6 (WK) or 12 (BK)
        self.moves[move_idx[~is_king]] *= masks[check_idx[~is_king]]     

        if check.two_pawn_push_check.any():
            # en passant -> 5, double pawn push check (masks) -> 8
            # 40 (5*8) -> hit, 8 (1*8) -> spurious move 
            spurious = self.moves == 8 
            self.moves[spurious] = 0 # (N_moves, 64)

            ep = self.moves == 40 # (N_moves, 64)
            self.moves[ep] = 5 # (N_moves, 64)

    @classmethod
    def empty(cls, device):
        return cls(
            moves=torch.empty(0, 64, dtype=torch.uint8, device=device),
            sq=torch.empty(0, dtype=torch.uint8, device=device),
            id=torch.empty(0, dtype=torch.uint8, device=device),
            board=torch.empty(0, dtype=torch.uint8, device=device)
        )
    
    def filter_empty(self):
        # Filter out empty moves
        valid = self.moves != 0
        valid = valid.any(dim=1)
        
        self.moves=self.moves[valid]
        self.sq=self.sq[valid]
        self.id=self.id[valid]
        self.board=self.board[valid]

    def concat(self, other):
        # Concatenate the PreMoves object with another PreMoves object
        self.moves = torch.cat([self.moves, other.moves])
        self.sq = torch.cat([self.sq, other.sq])
        self.id = torch.cat([self.id, other.id])
        self.board = torch.cat([self.board, other.board])

    def filter_king_by_attacks(self, enemy_attacks: torch.Tensor):
        is_king = self.id % 6 == 0
        if not is_king.any():
            return  # No king moves to filter
        king_board = self.board[is_king] # (N_kings,)
        attack_masks = enemy_attacks[king_board] # (N_kings, 64)
        self.moves[is_king] &= ~attack_masks

    def select(self, idx):
        # idx: boolean mask of length B (batch)
        keep_idx = idx.nonzero(as_tuple=True)[0]  # [B']
        is_kept = torch.isin(self.board, keep_idx)  # [N_moves]
        old_to_new = -torch.ones(idx.shape[0], dtype=torch.long, device=idx.device)
        old_to_new[keep_idx] = torch.arange(keep_idx.shape[0], device=idx.device)
        new_board = old_to_new[self.board[is_kept]]
        return PreMoves(
            moves=self.moves[is_kept],
            sq=self.sq[is_kept],
            id=self.id[is_kept],
            board=new_board,
        )