import pygame
import chess
import chess.svg
import sys
import numpy as np
from typing import List, Tuple, Optional, Any # Use Any for the custom LegalMove format initially
import torch
from dotenv import load_dotenv
import os
import yaml

from chessengine.preprocessing.position_parsing import encode_board, generate_legal_move_tensor
from chessengine.model.engine_pl import ChessLightningModule
from chessengine.chess_rl.mcts.mcts import BATCHED_MCTS
from chessengine.chess_rl.beam_search.beam import BEAM
from chessengine.model.prediction import predict3, mcts_predict

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BOARD_SIZE = 512 # Size of the chessboard in pixels
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_WIDTH = SCREEN_WIDTH - BOARD_SIZE
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 180
MARGIN = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 100) # Yellow with transparency
BUTTON_COLOR = (100, 100, 200)
BUTTON_TEXT_COLOR = WHITE
TEXT_COLOR = BLACK

# Piece Assets Path (Make sure this directory exists and contains piece images)
# Images should be named like wP.png, bP.png, wN.png, bN.png, etc.
ASSET_PATH = "chessengine/GameEnv/pieces/" # CHANGE IF NEEDED

# --- Type Hint for Custom Model Move Format ---
# Replace 'Any' with the actual type/structure if known
LegalMove = Any

# --- Model Interface (Stubbed - REPLACE WITH YOUR ACTUAL MODEL LOGIC) ---

# Assume model and necessary components are loaded elsewhere
# e.g., model = load_my_transformer_model()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# get the checkpoint path from the file checkpoint.txt

##################################
# LOAD CHECKPOINT AND CONFIGS
##################################

load_dotenv()
CHECKPOINT_PATH = os.getenv("BASE_MODEL")

# load config from the file engine_config.yaml
with open("engine_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

rl_config = {
            'mcts':{"num_simulations": 50, "cpuct": 1.0, "temperature_start": 1.0,},
            'beam_search': {"depth": 5, "width": 5},
        }
USE_MCTS = True # Set to True to use MCTS, False for greedy policy
USE_BEAM = False # Set to True to use Beam Search, False for greedy policy

##################################
# GET MODEL
##################################
        
model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
model.eval() # Set model to evaluation mode
model.to(device)

##################################
# MCTS
##################################
if USE_MCTS:
    mcts = BATCHED_MCTS(model, rl_config)
elif USE_BEAM:
    beam = BEAM(model, rl_config)

def prepare_features(board: chess.Board) -> Tuple[np.ndarray, List[LegalMove]]:
    """
    Given a python-chess Board object, returns:
    - A board tensor of shape (8, 8, 21) to be passed to the model.
    - A list of legal moves in a custom format for model compatibility.
    """
    print("--- Preparing Features ---")
    # --- !!! REPLACE WITH YOUR FEATURE EXTRACTION !!! ---
    # Example dummy tensor (replace with your actual logic)
    # This dummy tensor doesn't represent a real position.
    board_tensor = encode_board(board) # (8, 8, 21) uint8 torch tensor
    # For the stub, we'll just use the standard chess.Move objects
    # If your model needs a different format, adapt this.
    legal_moves, _ = generate_legal_move_tensor(board, None) # (64, L) int8 torch tensor
    print(f"Board FEN: {board.fen()}")
    print(f"Legal Moves (for model): {legal_moves.shape[-1]} moves")
    # --- / REPLACE ---
    return board_tensor, legal_moves

def evaluate_position(board: chess.Board) -> Tuple[float, List[chess.Move]]:
    """
    Returns:
    - Result probability distribution (3, ) for black win, draw and white win
    - A list of top 3 move suggestions (as python-chess Move objects), sorted by model preference
    """
    print("--- Evaluating Position ---")
    if USE_MCTS:
        move1, move2, move3, p1, p2, p3, prob_eval = mcts_predict(model, board, mcts ,device) # Get eval score from model
    else:
        move1, move2, move3, p1, p2, p3, prob_eval = predict3(model, board ,device) # Get eval score from model

    return prob_eval, [move1, move2, move3], [p1, p2, p3]

def get_best_move(board: chess.Board) -> Optional[chess.Move]:
    """
    Returns the best move (as a python-chess Move object)
    """
    print("--- Getting Best Move ---")
    if USE_MCTS:
        m1, m2, m3, p1, p2, p3, _ = mcts_predict(model, board, mcts ,device)
    else:
        m1, m2, m3, p1, p2, p3, _ = predict3(model, board, device) # Get eval score from model
    return [m1, m2, m3], [p1, p2, p3]

# --- GUI Class ---
class ChessGUI:
    def __init__(self, start_fen: Optional[str] = None):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Chess Engine Interface")

        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_medium = pygame.font.SysFont("monospace", 20)
        #self.font_large = pygame.font.SysFont("sans", 24)
        self.font_large = pygame.font.SysFont("arialunicode", 32)

        self.board = chess.Board(fen=start_fen) if start_fen else chess.Board()
        self.piece_images = self._load_piece_images()
        self.selected_square: Optional[chess.Square] = None
        self.highlighted_squares: List[chess.Square] = []
        self.player_color = chess.WHITE # Player plays as White by default
        self.is_player_turn = (self.board.turn == self.player_color)
        self.game_over = False
        self.status_message = ""
        self.last_eval_score: Optional[float] = None
        self.suggested_moves: List[chess.Move] = []
        self.probs: List[float] = []
        self.move_history_san: List[str] = []
        self.engine_status_message = ""

        # GUI Element Rects
        self.board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
        self.suggest_button_rect = pygame.Rect(BOARD_SIZE + MARGIN, MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.eval_button_rect = pygame.Rect(BOARD_SIZE + MARGIN, MARGIN + BUTTON_HEIGHT + MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
        # Add more buttons if needed (e.g., New Game, Flip Board)

    def _load_piece_images(self):
        images = {}
        pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
        for piece_code in pieces:
            try:
                path = f"{ASSET_PATH}{piece_code}.png"
                image = pygame.image.load(path)
                # Scale image to square size. Use convert_alpha for transparency.
                images[piece_code] = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE)).convert_alpha()
            except pygame.error as e:
                print(f"Error loading image {path}: {e}")
                # Create a placeholder surface if image loading fails
                placeholder = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                placeholder.fill((128, 128, 128, 100)) # Semi-transparent grey
                pygame.draw.rect(placeholder, (255,0,0), placeholder.get_rect(), 2) # Red border
                images[piece_code] = placeholder
                
        return images

    def _get_piece_code(self, piece: chess.Piece) -> Optional[str]:
        """Gets the code (e.g., 'wP', 'bN') for a chess.Piece."""
        if piece is None:
            return None
        color = 'w' if piece.color == chess.WHITE else 'b'
        ptype = piece.symbol().upper()
        return f"{color}{ptype}"

    def _square_to_coords(self, square: chess.Square) -> Tuple[int, int]:
        """Converts a chess square index (0-63) to pixel coordinates (top-left)."""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE # Pygame y-coordinates are top-down
        return x, y

    def _coords_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        """Converts pixel coordinates to a chess square index."""
        if not self.board_rect.collidepoint(x, y):
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None

    def _draw_board(self):
        for r in range(8):
            for f in range(8):
                color = LIGHT_SQUARE if (r + f) % 2 == 0 else DARK_SQUARE
                rect = pygame.Rect(f * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_code = self._get_piece_code(piece)
                if piece_code in self.piece_images:
                    x, y = self._square_to_coords(square)
                    self.screen.blit(self.piece_images[piece_code], (x, y))

    def _draw_highlights(self):
        # Highlight selected square
        if self.selected_square is not None:
            x, y = self._square_to_coords(self.selected_square)
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT_COLOR)
            self.screen.blit(highlight_surface, (x, y))

            # Highlight legal moves for the selected piece
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.color == self.board.turn: # Only show moves for current turn's piece
                 for move in self.board.legal_moves:
                     if move.from_square == self.selected_square:
                         tx, ty = self._square_to_coords(move.to_square)
                         # Draw a circle for potential move destinations
                         center_x = tx + SQUARE_SIZE // 2
                         center_y = ty + SQUARE_SIZE // 2
                         pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, (center_x, center_y), SQUARE_SIZE // 6)

    def _draw_gui_elements(self):
        # Info Panel Background
        panel_rect = pygame.Rect(BOARD_SIZE, 0, INFO_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, (200, 200, 200), panel_rect) # Light grey background

        # Buttons
        pygame.draw.rect(self.screen, BUTTON_COLOR, self.suggest_button_rect)
        pygame.draw.rect(self.screen, BUTTON_COLOR, self.eval_button_rect)

        suggest_text = self.font_medium.render("Suggest Moves", True, BUTTON_TEXT_COLOR)
        eval_text = self.font_medium.render("Evaluate", True, BUTTON_TEXT_COLOR)
        self.screen.blit(suggest_text, (self.suggest_button_rect.x + 10, self.suggest_button_rect.y + 10))
        self.screen.blit(eval_text, (self.eval_button_rect.x + 10, self.eval_button_rect.y + 10))

        # Status Message Area
        y_offset = self.eval_button_rect.bottom + MARGIN * 2

        # Turn Indicator
        turn_text = "White's Turn" if self.board.turn == chess.WHITE else "Black's Turn"
        turn_render = self.font_medium.render(turn_text, True, TEXT_COLOR)
        self.screen.blit(turn_render, (BOARD_SIZE + MARGIN, y_offset))
        y_offset += 30

        # Evaluation Display
        eval_str = f"W {100*self.Pw:.1f}% D {100*self.Pd:.1f}% B {100*self.Pb:.1f}%" if self.last_eval_score is not None else "Eval: N/A"
        eval_render = self.font_medium.render(eval_str, True, TEXT_COLOR)
        self.screen.blit(eval_render, (BOARD_SIZE + MARGIN, y_offset))
        y_offset += 30

        # Suggested Moves Display
        suggest_title = self.font_medium.render("Suggestions:", True, TEXT_COLOR)
        self.screen.blit(suggest_title, (BOARD_SIZE + MARGIN, y_offset))
        y_offset += 25
        for i, move in enumerate(self.suggested_moves):
            move_san = self.board.san(move) # Get Standard Algebraic Notation
            move_render = self.font_small.render(f"{i+1}. {move_san}", True, TEXT_COLOR)
            self.screen.blit(move_render, (BOARD_SIZE + MARGIN + 10, y_offset))
            y_offset += 20
            if i >= 2: break # Show top 3

        y_offset += MARGIN # Spacer

        # Engine Move Info
        if self.engine_status_message:
            lines = self.engine_status_message.split("\n")
            for line in lines:
                msg_render = self.font_small.render(line, True, TEXT_COLOR)
                self.screen.blit(msg_render, (BOARD_SIZE + MARGIN, y_offset))
                y_offset += 20

        # Move History
        history_title = self.font_medium.render("Move History:", True, TEXT_COLOR)
        history_rect = pygame.Rect(BOARD_SIZE + MARGIN, y_offset, INFO_PANEL_WIDTH - 2 * MARGIN, SCREEN_HEIGHT - y_offset - MARGIN)
        pygame.draw.rect(self.screen, (220, 220, 220), history_rect) # Slightly different background
        self.screen.blit(history_title, (history_rect.x + 5, history_rect.y + 5))
        y_offset = history_rect.y + 30
        
        # Display last N moves that fit in the box
        max_history_items = (history_rect.height - 35) // 20 
        start_index = max(0, len(self.move_history_san) - max_history_items)
        for i in range(start_index, len(self.move_history_san)):
            move_num = (i // 2) + 1
            prefix = f"{move_num}. " if i % 2 == 0 else ""
            history_text = f"{prefix}{self.move_history_san[i]}"
            history_render = self.font_small.render(history_text, True, TEXT_COLOR)
            self.screen.blit(history_render, (history_rect.x + 10, y_offset))
            y_offset += 20


        # Game Over Message
        if self.game_over:
            result = self.board.result()
            outcome = ""
            if self.board.is_checkmate():
                outcome = f"Checkmate! {'Black' if self.board.turn == chess.WHITE else 'White'} wins."
            elif self.board.is_stalemate():
                outcome = "Stalemate! Draw."
            elif self.board.is_insufficient_material():
                outcome = "Draw by insufficient material."
            elif self.board.is_seventyfive_moves():
                 outcome = "Draw by 75-move rule."
            elif self.board.is_fivefold_repetition():
                outcome = "Draw by fivefold repetition."
            # Add other draw conditions if needed
            else:
                outcome = f"Game Over: {result}"

            end_text = self.font_large.render(outcome, True, (255, 0, 0))
            text_rect = end_text.get_rect(center=(BOARD_SIZE / 2, BOARD_SIZE / 2))
            # Add a background to make text more visible
            pygame.draw.rect(self.screen, (200, 200, 200, 200), text_rect.inflate(10, 10))
            self.screen.blit(end_text, text_rect)

    def _update_display(self):
        """Draws all elements onto the screen."""
        self.screen.fill(WHITE) # Clear screen
        self._draw_board()
        self._draw_highlights()
        self._draw_pieces()
        self._draw_gui_elements()
        pygame.display.flip() # Update the full display

    def _handle_click(self, pos):
        """Handles mouse clicks on the board or buttons."""
        x, y = pos

        # Check button clicks first
        if self.suggest_button_rect.collidepoint(x, y):
            self._request_suggestions()
            return
        if self.eval_button_rect.collidepoint(x, y):
            self._request_evaluation()
            return

        # Check board click
        if self.game_over or not self.is_player_turn:
             # Ignore clicks if game is over or not player's turn
             self.selected_square = None
             return
             
        clicked_square = self._coords_to_square(x, y)

        if clicked_square is None: # Click outside board
            self.selected_square = None
            return

        if self.selected_square is None:
            # First click - select a square
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.player_color:
                self.selected_square = clicked_square
                # Highlight potential moves (will be drawn in _draw_highlights)
                self.highlighted_squares = [m.to_square for m in self.board.legal_moves if m.from_square == clicked_square]
            else:
                self.selected_square = None
                self.highlighted_squares = []
        else:
            # Second click - attempt to make a move
            move = None
            # Check for standard move
            try:
                move = self.board.find_move(self.selected_square, clicked_square)
            except ValueError: # Not a legal move (standard or promotion)
                 # Check for promotion - needs UI element, simplifying for now
                 # Try creating promotion moves manually if the target is a promotion square
                 piece = self.board.piece_at(self.selected_square)
                 if piece and piece.piece_type == chess.PAWN:
                     is_promo_rank = (piece.color == chess.WHITE and chess.square_rank(clicked_square) == 7) or \
                                     (piece.color == chess.BLACK and chess.square_rank(clicked_square) == 0)
                     if is_promo_rank:
                          # Simplification: Always promote to Queen if move is otherwise legal
                          promo_move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
                          if promo_move in self.board.legal_moves:
                              move = promo_move
                          else: # Maybe other promo pieces? For now, fail if Queen promo isn't legal
                              print(f"Promotion move {promo_move.uci()} not legal, check logic.")
                              move = None # Reset move if specific promotion isn't found/legal

            if move in self.board.legal_moves:
                self._make_move(move)
                # Player's move is done, now it's potentially engine's turn
                self.is_player_turn = False
                self.selected_square = None
                self.highlighted_squares = []
                # Trigger engine move *after* updating display for player move
            else:
                # Invalid move - deselect
                print(f"Illegal move attempt: {chess.square_name(self.selected_square)}{chess.square_name(clicked_square)}")
                self.selected_square = None
                self.highlighted_squares = []
                
        # After handling click, clear suggestions unless a button was pressed
        # self.suggested_moves = [] # Optional: clear suggestions after any board click

    def _make_move(self, move: chess.Move):
        """Pushes a move onto the board and updates history."""
        if not self.game_over:
            san_move = self.board.san(move)
            self.board.push(move)
            self.move_history_san.append(san_move)
            self.is_player_turn = (self.board.turn == self.player_color)
            self.game_over = self.board.is_game_over()
            self.last_eval_score = None # Clear eval after move
            self.suggested_moves = []   # Clear suggestions after move
            print(f"Move made: {san_move}")
            if self.game_over:
                print(f"Game Over! Result: {self.board.result()}")

    def _request_evaluation(self):
        """Calls the engine to evaluate the current position."""
        if self.game_over: return
        print("Requesting evaluation...")
        prob_eval, _, _ = evaluate_position(self.board) # Get eval score from model
        self.Pb = prob_eval[0].item()
        self.Pd = prob_eval[1].item()
        self.Pw = prob_eval[2].item()
        self.last_eval_score = self.Pw - self.Pb # White's perspective
        print(f"Evaluation: Black Win: {self.Pb:.2f}, Draw: {self.Pd:.2f}, White Win: {self.Pw:.2f}")

    def _request_suggestions(self):
        """Calls the engine to suggest top moves."""
        if self.game_over: return
        print("Requesting suggestions...")
        _, self.suggested_moves, self.probs  = evaluate_position(self.board) 
        for m, p in zip(self.suggested_moves, self.probs):
            if m is not None:
                print(f"Move: {m.uci()}, Probability: {p:.2f}")
        #print(f"Suggestions received: {[m.uci() for m in self.suggested_moves]}")

    def _engine_thinks_and_moves(self):
        """Handles the engine's turn."""
        if not self.game_over and not self.is_player_turn:
            print("Engine's turn...")
            self._update_display() # Update display to show it's engine's turn before thinking
            pygame.time.wait(100) # Small delay to allow display update

            moves, probs = get_best_move(self.board) # Get best move from model
            best_move = moves[0] if moves else None
            
            # Ensure we have valid moves for the engine
            if best_move is None:
                 print("Error: Engine has no legal moves according to prepare_features.")
                 # This might happen if prepare_features doesn't return standard moves
                 # Or if the game state is inconsistent. Handle appropriately.
                 self.game_over = True # Or handle based on game rules
                 return

            if best_move and best_move in self.board.legal_moves:
                self.engine_status_message = f"Engine played: {self.board.san(best_move)} ({probs[0]*100:.1f}%)"
                for i, (m, p) in enumerate(zip(moves[1:], probs[1:]), start=2):
                    if m is not None:
                        self.engine_status_message += f"\n{i}. {m.uci()} ({p*100:.1f}%)"
                self._make_move(best_move)
            elif best_move:
                 print(f"Error: Engine suggested an illegal move: {best_move.uci()}. Legal moves: {[m.uci() for m in self.board.legal_moves]}")
                 # Fallback: Maybe pick the first legal move from python-chess?
                 if self.board.legal_moves:
                     fallback_move = next(iter(self.board.legal_moves)) # Get first legal move
                     print(f"Engine playing fallback move: {self.board.san(fallback_move)}")
                     self._make_move(fallback_move)
                 else: # No legal moves at all -> game must be over
                      self.game_over = True
                      print("Engine has no legal moves (checked by python-chess). Game likely ended.")
            else:
                print("Engine failed to produce a move.")
                # This could indicate an issue in the model or that the game ended.
                if not self.board.legal_moves:
                    self.game_over = True
                    print("Engine has no legal moves (checked by python-chess). Game likely ended.")
                # Handle the case where model returns None but moves exist? Error or fallback?

            # Player's turn again
            self.is_player_turn = (self.board.turn == self.player_color)

    def run(self):
        """Main game loop."""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left mouse button
                        self._handle_click(event.pos)

            # If it's the engine's turn, let it move
            if not self.is_player_turn and not self.game_over:
                self._update_display() # Show board state before engine moves
                pygame.time.wait(100) # Short pause to see the player's move effect
                self._engine_thinks_and_moves()

            # Update display continuously
            self._update_display()
            clock.tick(30) # Limit frame rate

        pygame.quit()
        sys.exit()

# --- Main Execution ---
if __name__ == "__main__":
    # Example: Start from a custom FEN position
    # start_fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2" # Sicilian Defense start
    start_fen = None # Start from the default position
    
    game = ChessGUI(start_fen=start_fen)
    game.run()