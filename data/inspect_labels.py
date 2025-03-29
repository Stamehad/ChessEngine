import torch
import random
import chess
import chess.svg
from IPython.display import SVG, display, Markdown
import argparse

##################################################################################################
##################################################################################################
# Each training batch consists of a dictionary with the following entries:
# | Key              | Shape      | Description                                                  |
# |------------------|------------|--------------------------------------------------------------|
# | `board`          | (8, 8, 20) | The raw input tensor for a position                          |
# | `eval`           | (1,)       | Scalar win probability target (1=white win, 0=black win)     |
# | `move_target`    | (64,)      | Labels for changed squares (0=empty, 1–6=piece type)         |
# | `king_square`    | (1,)       | Index [0–63] of opponent king square                         |
# | `check`          | (1,)       | Whether opponent king is in check (0 or 1)                   |
# | `threat_target`  | (64,)      | Labels for newly threatened opponent pieces (0 or 1)         |
# | `move_weight`    | (1,)       | Optional per-sample weighting for move prediction loss       |
# | `terminal_flag`  | (1,)       | Indicates the game state (active, stalemate, checkmate)     |
##################################################################################################
##################################################################################################

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
check = sample["check"]
king_square = sample["king_square"]
terminal_flag = sample["terminal_flag"]

##################################################
# TEST FEATURES DIMENSIONS
##################################################

# Check dimensions
assert board_tensor.shape == (8, 8, 20), f"Expected (8, 8, 20), got {board_tensor.shape}"
assert eval_score.shape == (1,), f"Expected (1,), got {eval_score.shape}"
assert move_target.shape == (64,), f"Expected (64,), got {move_target.shape}"
assert threat_target.shape == (64,), f"Expected (64,), got {threat_target.shape}"
assert isinstance(king_square.item(), int) and 0 <= king_square.item() < 64, f"king_square must be in [0, 63], got {king_square}"
assert check.shape == (1,) and check.item() in {0, 1}, f"check must be 0 or 1, got {check}"
assert terminal_flag.shape == (1,), f"Expected (1,), got {terminal_flag.shape}"

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

def square_name(index):
    return chess.square_name(index)

##################################################
# SHOW LABELS
##################################################

# Reconstruct board from board_tensor
board = tensor_to_board(board_tensor)

# 1. Evaluation
outcome_map = {1.0: "White Win", 0.5: "Draw", 0.0: "Black Win"}
label_value = eval_score.item()
label_str = outcome_map.get(label_value, f"Unknown ({label_value})")
display(Markdown(f"### 🧮 Evaluation Target: **{label_str}**"))

# 2. King square
king_idx = king_square.item()
king_str = square_name(king_idx).upper()
display(Markdown(f"### 👑 Opponent King on **{king_str}**"))

# 3. In check
check_status = bool(check.item())
display(Markdown(f"### ⚠️ Opponent King in Check: **{check_status}**"))

# Side to move
side = "White" if board_tensor[0, 0, 13] else "Black"
display(Markdown(f"### ♟️ Side to Move: **{side}**"))

# 4. Terminal state
terminal_state_map = {0: "Active Game", 1: "Stalemate", 2: "Checkmate"}
terminal_state = terminal_flag.item()
terminal_str = terminal_state_map.get(terminal_state, "Unknown")
display(Markdown(f"### 🚩 Game State: **{terminal_str}**"))

# 5. Raw board
display(Markdown("### ♟️ Board Position"))
display_board(board, {})

# 6. Move targets (text)
move_strs = []
for idx in range(64):
    val = move_target[idx].item()
    if val >= 0 and val <= 6:
        move_strs.append(f"{square_name(idx).upper()} - {val}")
display(Markdown("### 🏹 Move Target Labels"))
display(Markdown(", ".join(move_strs)))

# 7. Move targets (board)
move_squares = [i for i in range(64) if move_target[i] >= 0 and move_target[i] <= 6]
display_board(board, {"cross": move_squares})

# 8. Threat target (board)
threat_squares = [i for i in range(64) if threat_target[i] == 1]
display(Markdown("### 🎯 Threat Target Labels"))
display_board(board, {"cross": threat_squares})