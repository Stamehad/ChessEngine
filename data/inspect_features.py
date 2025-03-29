import torch
import random
import chess
import chess.svg
from IPython.display import SVG, display, Markdown
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

# Then use it
if args.seed:
    SEED = args.seed
else:
    SEED = 42  # or any integer you like

random.seed(SEED)

print(f"Using sample index with seed {SEED}")

##################################################
# LOAD TENSORS
##################################################

# Load dataset
positions = torch.load("positions_short3.pt", weights_only=True)
print(f"✅ Loaded {len(positions)} positions from positions_short3.pt")

# Pick a random sample or use the specified index
if args.index:
    sample = positions[args.index]
else:
    sample = random.choice(positions)
board_tensor = sample["board"]
eval_score = sample["eval"]
move_target = sample["move_target"]
threat_target = sample["threat_target"]
# threat_mask = sample["threat_mask"]
check = sample["check"]
king_square = sample["king_square"]
# move_weight = sample["move_weight"]

##################################################
# TEST FEATURES DIMENSIONS
##################################################

# Check dimensions
assert board_tensor.shape == (8, 8, 20), f"Expected (8, 8, 20), got {board_tensor.shape}"
assert eval_score.shape == (1,), f"Expected (1,), got {eval_score.shape}"
assert move_target.shape == (64,), f"Expected (64,), got {move_target.shape}"
assert threat_target.shape == (64,), f"Expected (64,), got {threat_target.shape}"
# assert threat_mask.shape == (64,), f"Expected (64,), got {threat_mask.shape}"
# assert move_weight.shape == (1,), f"Expected (1,), got {move_weight.shape}"
assert isinstance(king_square.item(), int) and 0 <= king_square.item() < 64, f"king_square must be in [0, 63], got {king_square}"
assert check.shape == (1,) and check.item() in {0, 1}, f"check must be 0 or 1, got {check}"

##################################################
# CONVERT TENSORS TO CHESS BOARD
##################################################

def tensor_to_board(tensor):
    board = chess.Board.empty()
    board.clear_board()
    piece_map = {1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK, 5: chess.QUEEN, 6: chess.KING}

    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        for i in range(1, 7):
            if tensor[row, col, i]:
                board.set_piece_at(square, chess.Piece(piece_map[i], chess.WHITE))
        for i in range(7, 13):
            if tensor[row, col, i]:
                board.set_piece_at(square, chess.Piece(piece_map[i - 6], chess.BLACK))

    board.turn = bool(tensor[0, 0, 13])  # side to move flag
    return board

def display_board(board, highlights):
    # Combine all highlighted squares into a single list
    highlight_squares = set()
    for sq_list in highlights.values():
        highlight_squares.update(sq_list)
    svg = chess.svg.board(board=board, squares=list(highlight_squares), size=400)
    display(SVG(svg))

##################################################
# SHOW FEATURES
##################################################

# Reconstruct board from board_tensor
board = tensor_to_board(board_tensor)

def show_feature(name, sq_dict):
    display(Markdown(f"### {name}"))
    display_board(board, sq_dict)

# 1. Side to move
side = "White" if board_tensor[0, 0, 13] else "Black"
display(Markdown(f"### ♟️ Side to Move: **{side}**"))

# 2. King in check
in_check = bool(board_tensor[0, 0, 14])
display(Markdown(f"### ⚠️ King in Check: **{in_check}**"))

# 3. Castling rights
white_castle = bool(board_tensor[0, 0, 18])
black_castle = bool(board_tensor[0, 0, 19])
display(Markdown(f"### 🏰 Castling Rights — White: **{white_castle}**, Black: **{black_castle}**"))

# 4. Show raw board
display(Markdown("### 🧩 Raw Board"))
display_board(board, {})

# 5. Threatened pieces (channel 15)
threatened = []
for square in chess.SQUARES:
    row, col = divmod(square, 8)
    if board_tensor[row, col, 15]:
        threatened.append(square)
show_feature("🔥 Threatened Pieces", {"cross": threatened})

# 6. Threatening pieces (channel 16)
threatening = []
for square in chess.SQUARES:
    row, col = divmod(square, 8)
    if board_tensor[row, col, 16]:
        threatening.append(square)
show_feature("🎯 Threatening Pieces", {"cross": threatening})

# 7. Legal moves (channel 17)
legal_squares = []
for square in chess.SQUARES:
    row, col = divmod(square, 8)
    if board_tensor[row, col, 17]:
        legal_squares.append(square)
show_feature("🟦 Pieces With Legal Moves", {"cross": legal_squares})