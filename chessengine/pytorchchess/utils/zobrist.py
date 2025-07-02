import torch

class ZobristHasher:
    def __init__(self, device=None):
        self.device = device or "cpu"
        # Use a fixed seed for reproducibility (change if you want)
        torch.manual_seed(20240625)
        MAX_SAFE = 2**63 - 1

        # Choose appropriate dtype based on device
        if device and str(device).startswith('mps'):
            # MPS prefers int32 for index operations
            hash_dtype = torch.int32
            MAX_SAFE = 2**31 - 1  # Reduce range for int32
        else:
            hash_dtype = torch.long  # int64 for CUDA/CPU

        self.hash_dtype = hash_dtype
        self.piece_keys = torch.randint(0, MAX_SAFE, (12, 64), dtype=hash_dtype, device=self.device)
        self.castling_keys = torch.randint(0, MAX_SAFE, (4,), dtype=hash_dtype, device=self.device)
        self.ep_keys = torch.randint(0, MAX_SAFE, (64,), dtype=hash_dtype, device=self.device)
        self.side_key = torch.randint(0, MAX_SAFE, (1,), dtype=hash_dtype, device=self.device)[0]

    def hash(self, board_flat, side_to_move, castling, ep_square):
        """
        board_flat: (B, 64)  -- 0=empty, 1-6=white, 7-12=black
        side_to_move: (B, 1)   -- 1=white, 0=black
        castling: (B, 4)     -- 0/1 for [K, Q, k, q]
        ep_square: (B,)      -- 0...64 (64=none, 0-63=square index)
        """
        B = board_flat.shape[0]
        hashes = torch.zeros(B, dtype=self.hash_dtype, device=board_flat.device)

        # Mask for non-empty squares
        piece_idx = board_flat.long() - 1  # 0..11 for pieces, -1 for empty
        mask = piece_idx >= 0
        batch_idx, sq_idx = torch.where(mask)
        p_idx = piece_idx[batch_idx, sq_idx]
        hashes.index_add_(
            0, batch_idx, self.piece_keys[p_idx, sq_idx]
        )

        # Castling rights
        for i in range(4):
            mask = castling[:, i] == 1
            if mask.any():
                hashes[mask] ^= self.castling_keys[i]

        # En passant
        mask = ep_square < 64
        if mask.any():
            hashes[mask] ^= self.ep_keys[ep_square[mask].long()]

        # Side to move (1 = white)
        mask = side_to_move.squeeze() == 1
        if mask.any():
            hashes[mask] ^= self.side_key

        return hashes

# Singleton for shared use
_GLOBAL_HASHER = None

def get_hasher(device="cpu"):
    global _GLOBAL_HASHER
    if _GLOBAL_HASHER is None:
        _GLOBAL_HASHER = ZobristHasher(device)
    elif _GLOBAL_HASHER.device != device:
        # Move existing hasher to new device
        _GLOBAL_HASHER.piece_keys = _GLOBAL_HASHER.piece_keys.to(device)
        _GLOBAL_HASHER.castling_keys = _GLOBAL_HASHER.castling_keys.to(device)
        _GLOBAL_HASHER.ep_keys = _GLOBAL_HASHER.ep_keys.to(device)
        _GLOBAL_HASHER.side_key = _GLOBAL_HASHER.side_key.to(device)
        _GLOBAL_HASHER.device = device
    return _GLOBAL_HASHER