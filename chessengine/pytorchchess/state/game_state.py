import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class GameState:
    side_to_move: torch.Tensor  # (B,1)
    plys: torch.Tensor          # (B,)
    castling: torch.Tensor      # (B,4) # KQkq
    ep: torch.Tensor            # (B,)
    previous_moves: Optional[torch.Tensor] = None  # (B,) last move
    layer: Optional[torch.Tensor] = None           # (B,) layer index for beam search
