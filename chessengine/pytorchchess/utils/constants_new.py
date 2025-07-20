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
    """
    KING:       [[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 4, 5, 6, 0, 0, 0],
                 [0, 0, 3, 0, 7, 0, 0, 0],
                 [0, 0, 2, 1, 8, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]

    QUEEN:      [[ 0,  0,  0, 25,  0,  0,  0, 26],
                 [24,  0,  0, 17,  0,  0, 18,  0],
                 [ 0, 16,  0,  9,  0, 10,  0,  0],
                 [ 0,  0,  8,  1,  2,  0,  0,  0],
                 [23, 15,  7,  0,  3, 11, 19, 27],
                 [ 0,  0,  6,  5,  4,  0,  0,  0],
                 [ 0, 14,  0, 13,  0, 12,  0,  0],
                 [22,  0,  0, 21,  0,  0, 20,  0]])

    ROOK: odd part of QUEEN
    BISHOP: even part of QUEEN
    """
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
        KING_MOVES = ((0,1), (1,1), (1,0), (1,-1),
            (0,-1), (-1,-1), (-1,0), (-1,1)
        )
        for j, (dr, df) in enumerate(KING_MOVES):
            tau_j = j if j != 0 else 8
            set_item(KING[sq], r + dr, f + df, tau_j)

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

        # Queen
        QUEEN_MOVES = ((0,1), (1,1), (1,0), (1,-1),
            (0,-1), (-1,-1), (-1,0), (-1,1)
        )
        QUEEN[sq, r, f] = 128  # Set the center square to 64
        for j, (dr, df) in enumerate(QUEEN_MOVES):
            tau_j = j if j != 0 else 8
            for i in range(1, 8):
                set_item(QUEEN[sq], r + dr * i, f + df * i, value=8*(i-1)+tau_j)

        
        #ROOK[sq, r, f] = 128  # Set the center square to 64 for rook
        BISHOP[sq, r, f] = 128  # Set the center square to 64 for bishop

    ROOK += (QUEEN % 2 == 0) * QUEEN
    BISHOP += (QUEEN % 2 == 1) * QUEEN

    return KNIGHT, KING, PAWN_W, PAWN_B, PAWN_PUSH_W, PAWN_PUSH_B, BISHOP, ROOK, QUEEN

KNIGHT_MOVES, KING_MOVES, PAWN_CAP_W, PAWN_CAP_B, PAWN_PUSH_W, PAWN_PUSH_B, BISHOP, ROOK, QUEEN_MOVES = _gen_shift_masks(dtype=torch.uint8)

MOVES = torch.stack([
    PAWN_CAP_W + PAWN_PUSH_W, KNIGHT_MOVES, BISHOP, ROOK, QUEEN_MOVES, KING_MOVES, PAWN_CAP_B + PAWN_PUSH_B, KNIGHT_MOVES, BISHOP, ROOK, QUEEN_MOVES, KING_MOVES
], dim=0)  # (12, 64, 8, 8)
MOVES = MOVES.view(12, 64, 64)


# SHORT_RANGE_MOVES = torch.stack([
#     KNIGHT_MOVES, KING_MOVES, PAWN_PUSH_W + PAWN_CAP_W, PAWN_PUSH_B + PAWN_CAP_B
#     ], dim=0)  # shape (4, 64, 8, 8)
# SHORT_RANGE_MOVES = SHORT_RANGE_MOVES.view(4, 64, 64)  # shape (4, 64, 64)

# LONG_RANGE_MOVES = torch.stack([
#     (QUEEN_MOVES % 2 == 0) * QUEEN_MOVES, (QUEEN_MOVES % 2 == 1) * QUEEN_MOVES, QUEEN_MOVES
#     ], dim=0)  # shape (3, 64, 8, 8)
# LONG_RANGE_MOVES = LONG_RANGE_MOVES.view(3, 64, 64)  # shape (3, 64, 64)

PROMOTION_MASK = torch.cat([
    torch.ones(1, 8, dtype=torch.uint8), 
    torch.zeros(1, 48, dtype=torch.uint8), 
    torch.ones(1, 8, dtype=torch.uint8)],
    dim=1
) # (1, 64)

king_side  = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0], dtype=torch.bool) # (8,)
queen_side = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool) # (8,)
CASTLING_ZONES = torch.stack([
    torch.cat([king_side, torch.zeros(56, dtype=torch.bool)], dim=0),
    torch.cat([queen_side, torch.zeros(56, dtype=torch.bool)], dim=0),
    torch.cat([torch.zeros(56, dtype=torch.bool), king_side], dim=0),
    torch.cat([torch.zeros(56, dtype=torch.bool), queen_side], dim=0)
], dim=0).to(dtype=torch.bool) # (4, 64)

king_side_attack  = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.bool) # (8,)
queen_side_attack = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool) # (8,)
CASTLING_ATTACK_ZONES = torch.stack([
    torch.cat([king_side_attack, torch.zeros(56, dtype=torch.bool)], dim=0),
    torch.cat([queen_side_attack, torch.zeros(56, dtype=torch.bool)], dim=0),
    torch.cat([torch.zeros(56, dtype=torch.bool), king_side_attack], dim=0),
    torch.cat([torch.zeros(56, dtype=torch.bool), queen_side_attack], dim=0)
], dim=0).to(dtype=torch.bool) # (4, 64)

KING_TO = torch.tensor([6, 2, 62, 58], dtype=torch.long)  # King to square after castling (white: 6, black: 58)

# CASTLING_ATTACK_ZONES = torch.cat([
#     torch.tensor([0, 0, 1, 1, 1, 1, 1, 0], dtype=torch.uint8),
#     torch.zeros(48, dtype=torch.uint8),
#     torch.tensor([0, 0, 1, 1, 1, 1, 1, 0], dtype=torch.uint8)
# ], dim=0).to(dtype=torch.uint8) # (64,)

# Function to move all constants to a specified device
def move_constants_to(device):
    global KNIGHT_MOVES, KING_MOVES, PAWN_CAP_W, PAWN_CAP_B, PAWN_PUSH_W, PAWN_PUSH_B, QUEEN_MOVES
    global SHORT_RANGE_MOVES, LONG_RANGE_MOVES, PROMOTION_MASK, CASTLING_ZONES, CASTLING_ATTACK_ZONES

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