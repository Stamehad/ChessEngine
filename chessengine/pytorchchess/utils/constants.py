import torch
#from move import MoveEncoder

# Piece indices 0-11 (white 0-5, black 6-11)
EM, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK = range(13)

PIECE_NAMES = [
    "WP", "WN", "WB", "WR", "WQ", "WK",
    "BP", "BN", "BB", "BR", "BQ", "BK",
]

WHITE_RANGE = slice(1, 6)   # WP … WK
BLACK_RANGE = slice(6, 12)  # BP … BK

# Directions (square index deltas)
N, S, E, W  =  8, -8,  1, -1
NE, NW, SE, SW = 9, 7, -7, -9

ray_dict = {0: (1, -1), 1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (-1, 1), 5: (-1, 0), 6: (-1, -1), 7: (0, -1)}

def on_board(rr, ff): return 0 <= rr < 8 and 0 <= ff < 8
def set_item(x, r, f, value=1): 
    if on_board(r, f):
        x[r, f] = value

# Knight jumps, king moves, pawn captures (64-bit masks – computed once on CPU)
def _gen_shift_masks(dtype=torch.uint8):
    KNIGHT = torch.zeros(64, 8, 8, dtype=dtype)
    KING   = torch.zeros(64, 8, 8, dtype=dtype)
    PAWN_W = torch.zeros(64, 8, 8, dtype=dtype)
    PAWN_B = torch.zeros(64, 8, 8, dtype=dtype)
    PAWN_PUSH_W = torch.zeros(64, 8, 8, dtype=dtype)
    PAWN_PUSH_B = torch.zeros(64, 8, 8, dtype=dtype)
    BISHOP = torch.zeros(64, 8, 8, dtype=dtype)
    ROOK   = torch.zeros(64, 8, 8, dtype=dtype)
    QUEEN  = torch.zeros(64, 8, 8, dtype=dtype)

    # Pre-compute bit masks
    for sq in range(64):
        r, f = divmod(sq, 8)
        
        # King
        KING_MOVES = (( 1, 0), ( 0, 1), (-1, 0), ( 0,-1),
            ( 1, 1), (-1, 1), (-1,-1), ( 1,-1)
        )
        for dr, df in KING_MOVES:
            set_item(KING[sq], r + dr, f + df)

        # Knight
        KNIGHT_MOVES = (
            ( 2, 1), ( 1, 2), (-1, 2), (-2, 1),
            (-2,-1), (-1,-2), ( 1,-2), ( 2,-1)
        )
        for dr, df in KNIGHT_MOVES:
            set_item(KNIGHT[sq], r + dr, f + df)

        # Pawns
        for df in ( -1, 1 ):
            set_item(PAWN_W[sq], r + 1, f + df, 4)
            set_item(PAWN_B[sq], r - 1, f + df, 4)

        # Pawn pushes
        if sq >= 8:
            set_item(PAWN_PUSH_W[sq], r + 1, f)
        if sq < 56:
            set_item(PAWN_PUSH_B[sq], r - 1, f)

        # Two-square pushes
        if sq >= 8 and sq < 16:
            set_item(PAWN_PUSH_W[sq], r + 2, f, 2)
        if sq < 56 and sq >= 48:
            set_item(PAWN_PUSH_B[sq], r - 2, f, 2)

        # Rook
        ROOK_MOVES = (
            ( 1, 0), ( 0, 1), (-1, 0), ( 0,-1)
        )
        for j, (dr, df) in enumerate(ROOK_MOVES):
            for i in range(1, 8):
                set_item(ROOK[sq], r + dr * i, f + df * i, value=8*(i-1)+2*j+1)

        # Bishop
        BISHOP_MOVES = (
            ( 1, 1), (-1, 1), (-1,-1), ( 1,-1)
        )
        for j, (dr, df) in enumerate(BISHOP_MOVES):
            for i in range(1, 8):
                set_item(BISHOP[sq], r + dr * i, f + df * i, value=8*(i-1)+2*j+2)

        # Queen
        QUEEN = ROOK + BISHOP

    return KNIGHT, KING, PAWN_W, PAWN_B, PAWN_PUSH_W, PAWN_PUSH_B, QUEEN

KNIGHT_MOVES, KING_MOVES, PAWN_CAP_W, PAWN_CAP_B, PAWN_PUSH_W, PAWN_PUSH_B, QUEEN_MOVES = _gen_shift_masks(dtype=torch.uint8)

SHORT_RANGE_MOVES = torch.stack([
    KNIGHT_MOVES, KING_MOVES, PAWN_PUSH_W + PAWN_CAP_W, PAWN_PUSH_B + PAWN_CAP_B
    ], dim=0)  # shape (4, 64, 8, 8)
SHORT_RANGE_MOVES = SHORT_RANGE_MOVES.view(4, 64, 64)  # shape (4, 64, 64)

LONG_RANGE_MOVES = torch.stack([
    (QUEEN_MOVES % 2 == 0) * QUEEN_MOVES, (QUEEN_MOVES % 2 == 1) * QUEEN_MOVES, QUEEN_MOVES
    ], dim=0)  # shape (3, 64, 8, 8)
LONG_RANGE_MOVES = LONG_RANGE_MOVES.view(3, 64, 64)  # shape (3, 64, 64)

PROMOTION_MASK = torch.cat([
    torch.ones(1, 8, dtype=torch.uint8), 
    torch.zeros(1, 48, dtype=torch.uint8), 
    torch.ones(1, 8, dtype=torch.uint8)],
    dim=1
) # (1, 64)

king_side = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.uint8) # (8,)
queen_side = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.uint8) # (8,)
CASTLING_ZONES = torch.stack([
    torch.cat([king_side, torch.zeros(56, dtype=torch.uint8)], dim=0),
    torch.cat([queen_side, torch.zeros(56, dtype=torch.uint8)], dim=0),
    torch.cat([torch.zeros(56, dtype=torch.uint8), king_side], dim=0),
    torch.cat([torch.zeros(56, dtype=torch.uint8), queen_side], dim=0)
], dim=0).to(dtype=torch.uint8) # (4, 64)

CASTLING_ATTACK_ZONES = torch.cat([
    torch.tensor([0, 0, 1, 1, 1, 1, 1, 0], dtype=torch.uint8),
    torch.zeros(48, dtype=torch.uint8),
    torch.tensor([0, 0, 1, 1, 1, 1, 1, 0], dtype=torch.uint8)
], dim=0).to(dtype=torch.uint8) # (64,)

_CURRENT_DEVICE = torch.device("cpu")

# Function to move all constants to a specified device
def move_constants_to(device):
    global KNIGHT_MOVES, KING_MOVES, PAWN_CAP_W, PAWN_CAP_B, PAWN_PUSH_W, PAWN_PUSH_B, QUEEN_MOVES
    global SHORT_RANGE_MOVES, LONG_RANGE_MOVES, PROMOTION_MASK, CASTLING_ZONES, CASTLING_ATTACK_ZONES
    global _CURRENT_DEVICE

    if device == _CURRENT_DEVICE:
        return

    KNIGHT_MOVES = KNIGHT_MOVES.to(device)
    KING_MOVES = KING_MOVES.to(device)
    PAWN_CAP_W = PAWN_CAP_W.to(device)
    PAWN_CAP_B = PAWN_CAP_B.to(device)
    PAWN_PUSH_W = PAWN_PUSH_W.to(device)
    PAWN_PUSH_B = PAWN_PUSH_B.to(device)
    QUEEN_MOVES = QUEEN_MOVES.to(device)

    SHORT_RANGE_MOVES = SHORT_RANGE_MOVES.to(device)
    LONG_RANGE_MOVES = LONG_RANGE_MOVES.to(device)
    PROMOTION_MASK = PROMOTION_MASK.to(device)
    CASTLING_ZONES = CASTLING_ZONES.to(device)
    CASTLING_ATTACK_ZONES = CASTLING_ATTACK_ZONES.to(device)

    _CURRENT_DEVICE = device
