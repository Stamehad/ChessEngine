import torch
from typing import NamedTuple
from dataclasses import dataclass
from typing import Optional
from pytorchchess.utils import get_check_blockers, squares_to_int
from pytorchchess.utils.constants import PROMOTION_MASK

@dataclass(frozen=True)
class PinData:
    king_sq: torch.Tensor            # (N_pin,) square of the king (used to compute rays)
    pinned_piece_sq: torch.Tensor    # (N_pin,) index of the friendly pinned piece
    ray: torch.Tensor                # (N_pin,) 0–7
    board: torch.Tensor              # (N_pin,)
    
    def pin_mask(self):
        return get_check_blockers(self.king_sq, self.ray) # (N_pin, 64)
    
    def clone(self):
        # Return a copy of the PinData object
        return PinData(
            king_sq=self.king_sq.clone(),
            pinned_piece_sq=self.pinned_piece_sq.clone(),
            ray=self.ray.clone(),
            board=self.board.clone()
        )
    
    @classmethod
    def empty(cls, device):
        return cls(
            king_sq=torch.empty(0, device=device, dtype=torch.uint8),
            pinned_piece_sq=torch.empty(0, device=device, dtype=torch.uint8),
            ray=torch.empty(0, device=device, dtype=torch.uint8),
            board=torch.empty(0, device=device, dtype=torch.uint8)
        )
    
    def select(self, idx):
        # idx: boolean mask of length B (batch)
        keep_idx = idx.nonzero(as_tuple=True)[0]  # [B']
        is_kept = torch.isin(self.board, keep_idx).to(torch.bool)  # [N_pin]
        old_to_new = -torch.ones(idx.shape[0], dtype=torch.long, device=idx.device)
        old_to_new[keep_idx] = torch.arange(keep_idx.shape[0], device=idx.device)
        new_board = old_to_new[self.board[is_kept].long()]
        return PinData(
            king_sq=self.king_sq[is_kept],
            pinned_piece_sq=self.pinned_piece_sq[is_kept],
            ray=self.ray[is_kept],
            board=new_board,
        )

@dataclass(frozen=True)
class CheckData:
    king_sq: torch.Tensor             # (N_check,) square of the king (used to compute rays)
    attacker_sq: torch.Tensor         # (N_check,)
    attack_ray: torch.Tensor          # (N_check,) ray index 0–7, or -1 if non-slider
    board: torch.Tensor               # (N_check,)
    two_pawn_push_check: torch.Tensor # (N_check,) T/F, if the check is from a double pawn push

    def clone(self):
        # Return a copy of the CheckData object
        return CheckData(
            king_sq=self.king_sq.clone(),
            attacker_sq=self.attacker_sq.clone(),
            attack_ray=self.attack_ray.clone(),
            board=self.board.clone(),
            two_pawn_push_check=self.two_pawn_push_check.clone()
        )
    
    def check_mask(self):
        # Get mask for each attacker square and attack ray
        return get_check_blockers(
            self.king_sq, self.attack_ray, self.attacker_sq, self.two_pawn_push_check
        )  # (N_check, 64)
    
    def reduced_check_mask(self):
        # If a position has double check, chess data will contain two entries for the same position.
        # The reduced mask is a multiplication of the check masks for each entry.
        masks = get_check_blockers(
            self.king_sq, self.attack_ray, self.attacker_sq, self.two_pawn_push_check
            ) # (N_check, 64)

        # uniq_boards: (N_single,)
        # inv: (N_check,) each entry ∈ [0..N_single-1]
        uniq_boards, inv = torch.unique(self.board, return_inverse=True)   
        reduced_masks = torch.ones(len(uniq_boards), 64, dtype=masks.dtype, device=masks.device)
        reduced_masks.scatter_reduce_(0, inv.unsqueeze(-1).expand_as(masks), masks, reduce="prod")

        return uniq_boards, reduced_masks # (N_single,), (N_single, 64)
    
    def concat(self, other):
        # Concatenate the check data with another CheckData object
        return CheckData(
            king_sq=torch.cat([self.king_sq, other.king_sq]),
            attacker_sq=torch.cat([self.attacker_sq, other.attacker_sq]),
            attack_ray=torch.cat([self.attack_ray, other.attack_ray]),
            board=torch.cat([self.board, other.board]),
            two_pawn_push_check=torch.cat([self.two_pawn_push_check, other.two_pawn_push_check])

        )
    
    @classmethod
    def empty(cls, device):
        return cls(
            king_sq=torch.empty(0, dtype=torch.uint8, device=device),
            attacker_sq=torch.empty(0, dtype=torch.uint8, device=device),
            attack_ray=torch.empty(0, dtype=torch.uint8, device=device),
            board=torch.empty(0, dtype=torch.uint8, device=device),
            two_pawn_push_check=torch.empty(0, dtype=torch.bool, device=device)
        )
    
    def select(self, idx):
        # idx: boolean mask of length B (batch)
        keep_idx = idx.nonzero(as_tuple=True)[0]  # [B']
        is_kept = torch.isin(self.board, keep_idx)  # [N_check]
        old_to_new = -torch.ones(idx.shape[0], dtype=torch.long, device=idx.device)
        old_to_new[keep_idx] = torch.arange(keep_idx.shape[0], device=idx.device)
        new_board = old_to_new[self.board[is_kept].long()]
        return CheckData(
            king_sq=self.king_sq[is_kept],
            attacker_sq=self.attacker_sq[is_kept],
            attack_ray=self.attack_ray[is_kept],
            board=new_board,
            two_pawn_push_check=self.two_pawn_push_check[is_kept],
        )
    
@dataclass
class CheckInfo:
    in_check: torch.Tensor         # (B,)
    pin_data: Optional[PinData] = None
    check_data: Optional[CheckData] = None

    def select(self, idx):
        return CheckInfo(
            in_check=self.in_check[idx],
            pin_data=self.pin_data.select(idx) if self.pin_data is not None else None,
            check_data=self.check_data.select(idx) if self.check_data is not None else None,
        )
    
    def clone(self):
        return CheckInfo(
            in_check=self.in_check.clone(),
            pin_data=self.pin_data.clone() if self.pin_data is not None else None,
            check_data=self.check_data.clone() if self.check_data is not None else None
        )