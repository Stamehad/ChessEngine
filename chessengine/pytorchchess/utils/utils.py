import torch
import chess # type: ignore
from .constants import QUEEN_MOVES
import pytorchchess.utils.constants as c

def get_check_blockers(sq: torch.Tensor, n: torch.Tensor, sq2=None, two_pawn_push_check=None) -> torch.Tensor:
    """
    Get mask for a given square and direction.
    Args:
        sq (N,): The square index (0-63).
        n (N,): The direction index (0-7).
        sq2 (N, optional): end square index (0-63). If provided, the ray will be masked to only include squares up to sq2.
    If None, the full ray is returned.
        two_pawn_push_check (N,): if True add an en passant square to the mask 
    Returns:
        torch.Tensor: The ray as a 2D tensor (N, 8, 8).
    """
    if sq.numel() == 0:
        return torch.empty(0, 64, dtype=torch.bool, device=sq.device)
    
    mask = c.QUEEN_MOVES[sq].clone() # (N, 8, 8)
    n = n.view(-1, 1, 1) # (N, 1, 1)
    mask = (mask % 8 == n) & (mask != 0) # (N, 8, 8)

    if sq2 is not None:
        mask2 = c.QUEEN_MOVES[sq2].clone() # (N, 8, 8)
        mask2 = (mask2 % 8 == n) & (mask2 != 0) # (N, 8, 8)
        mask = (mask & ~mask2)
        
        # when n == -1 (no ray), use only sq2 as a mask
        sq2_mask = torch.nn.functional.one_hot(sq2, 64).view(-1, 8, 8) # (N, 8, 8)
        mask += ((n == -1) * sq2_mask.bool()) # (N, 8, 8)

    mask = mask.view(-1, 64).to(torch.uint8) # (N, 64)

    if two_pawn_push_check is not None and two_pawn_push_check.any():
        # find the en passant square
        white = sq > sq2 # (N,) if the attacked square is above the attaker the pawn must be white
        white = white[two_pawn_push_check] # (N_tpp,)
        ep_sq = sq2[two_pawn_push_check] + torch.where(white, -8, 8) # (N_tpp,)
        
        mask[two_pawn_push_check, ep_sq] = 8 # (N_tpp, 64) special value for en passant square
    return mask 

def squares_to_int(from_sq: torch.Tensor, to_sq: torch.Tensor, move_type: torch.Tensor) -> torch.Tensor:
        # Convert from_sq and to_sq to a 16-bit integer
        # from_sq: 0-63 (6 bits)
        # to_sq: 0-63 (6 bits)
        # move_type: 0-7 (3 bits)
        if isinstance(from_sq, int):
            from_sq = torch.tensor(from_sq, dtype=from_sq.dtype, device=from_sq.device)
        # assert (from_sq >= 0).all() and (from_sq < 64).all(), "from_sq out of range"
        # assert (to_sq >= 0).all() and (to_sq < 64).all(), "to_sq out of range"
        # assert (move_type >= 0).all() and (move_type < 8).all(), "move_type out of range"
        
        move = from_sq + to_sq * 2**6 + move_type * 2**12
        return move.to(move_dtype(from_sq.device))

def int_to_squares(move: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert a 16-bit integer to from_sq, to_sq, and move_type
        # from_sq: 0-63 (6 bits)
        # to_sq: 0-63 (6 bits)
        # move_type: 0-7 (3 bits)
        #move = move % 2**16
        from_sq = move % 2**6
        to_sq = (move // 2**6) % 2**6
        move_type = (move // 2**12) % 2**3
        return from_sq.int(), to_sq.int(), move_type.int()

def to_chess_move(encoded_move: torch.Tensor) -> chess.Move:
    from_sq, to_sq, promo_type = int_to_squares(encoded_move)
    promo_piece = None
    if promo_type in (1, 2, 3, 4):          # your promotion codes
        mapping = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}
        promo_piece = mapping[promo_type.item()]
    return chess.Move(from_sq, to_sq, promotion=promo_piece)

def move_dtype(device=None):
    """Get appropriate move dtype based on device capabilities"""
    if device is None:
        device = torch.device('cpu')
    
    if device.type == 'mps':
        return torch.int32  # MPS doesn't support uint16
    else:
        return torch.int16  # Optimal for CUDA/CPU

