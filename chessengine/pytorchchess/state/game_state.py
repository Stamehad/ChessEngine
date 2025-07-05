import torch
from dataclasses import dataclass, fields
from typing import Optional
from pytorchchess.utils.utils import move_dtype


@dataclass
class GameState:
    side_to_move: torch.Tensor  # (B,1) uint8: 1=white, 0=black 
    plys: torch.Tensor          # (B,) long
    castling: torch.Tensor      # (B,4) uint8: KQkq
    ep: torch.Tensor            # (B,) uint8
    fifty_move_clock: torch.Tensor                      # (B,) uint8: number of moves since last pawn move or capture
    previous_moves: Optional[torch.Tensor] = None       # (B,) int16: last move 
    position_history: Optional[torch.Tensor] = None     # (B, H) long Zobrist hashes of previous positions

    @property
    def side(self):
        return self.side_to_move.view(-1)  # (B,)
    
    @property
    def device(self):
        """Return the device of the tensors in this GameState"""
        if self.side_to_move is not None:
            return self.side_to_move.device
        else:
            return torch.device("cpu")

    def to(self, device: torch.device):
        return GameState(
            **{
                f.name: (
                    getattr(self, f.name).to(device)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )
    
    def clone(self):
        return GameState(
            **{f.name: (getattr(self, f.name).clone()
                if isinstance(getattr(self, f.name), torch.Tensor)
                else getattr(self, f.name))
                for f in fields(self)}
        )

    def __getitem__(self, idx):
        out = {}
        for name, t in self.__dict__.items():
            if isinstance(t, torch.Tensor):
                # If dtype can't be indexed, do workaround
                try:
                    v = t[idx].clone()
                except RuntimeError:
                    v = t.to(torch.long)[idx].clone().to(dtype=t.dtype)
                out[name] = v
            else:
                out[name] = t
        return GameState(**out)
    
    def cat(self, other):
        """
        Concatenate another GameState to this one along the batch dimension.
        Assumes all tensors have the same shape except for the first dimension.
        """
        assert isinstance(other, GameState), "Can only concatenate with another GameState"
        out = {}
        for name, t in self.__dict__.items():
            if isinstance(t, torch.Tensor):
                out[name] = torch.cat([t, getattr(other, name)], dim=0)
            else:
                out[name] = t
        return GameState(**out)
    
    def update_previous_moves(self, moves: torch.Tensor, idx: torch.Tensor):
        """Update the previous moves for the specified boards"""
        self.previous_moves[idx] = moves.to(move_dtype(self.device))
        # temp = self.previous_moves.to(torch.int32)
        # temp[idx] = moves[idx].to(torch.int32)
        # self.previous_moves = temp.to(move_dtype(self.device))

    def set_en_passant_squares(self, ep_boards: torch.Tensor, ep_squares: torch.Tensor):
        """Update en passant squares for specific boards"""
        self.ep[ep_boards] = ep_squares.to(torch.uint8)
    
    def reset_en_passant(self, idx: torch.Tensor):
        """Reset all en passant squares to invalid (64)"""
        self.ep[idx] = 64
    
    def update_castling_rights(self, castling_update: torch.Tensor, mask: torch.Tensor):
        # ks_white: torch.Tensor, qs_white: torch.Tensor, 
        #                       ks_black: torch.Tensor, qs_black: torch.Tensor):
        """Update castling rights based on move masks"""
        self.castling[mask] &= ~castling_update

    def update_after_move(self, resets_fifty_move: torch.Tensor, idx: torch.Tensor):
        """
        Update the game state after a move.
        This includes clearing position history for irreversible moves.
        """
        # Switch side to move
        self.side_to_move[idx] = 1 - self.side_to_move[idx]
        self.plys[idx] += 1
        
        # Reset fifty-move clock for captures and pawn moves
        reset_boards = idx[resets_fifty_move]
        non_reset_boards = idx[~resets_fifty_move]

        self.fifty_move_clock[reset_boards] = 0
        self.fifty_move_clock[non_reset_boards] += 1
        if resets_fifty_move.any():
            self._clear_position_history_on_irreversible_move(reset_boards)

    def _clear_position_history_on_irreversible_move(self, reset_mask: torch.Tensor):
        """
        For boards with irreversible move (reset_mask==True), zero their history.
        """
        if self.position_history is not None:
            self.position_history[reset_mask] = 0

            # After zeroing, prune any all-zero columns (padding columns)
            # Compute which columns are all zero
            if self.position_history.numel() > 0:
                col_mask = (self.position_history != 0).any(dim=0)  # (H,)
                self.position_history = self.position_history[:, col_mask] # (B, H_new)