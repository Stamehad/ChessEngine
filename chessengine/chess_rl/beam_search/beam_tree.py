import time
import torch
import chess
from typing import List, Tuple
from chessengine.preprocessing.position_parsing import generate_legal_move_tensor
from chessengine.model.utils import pad_and_stack
from chessengine.model.prediction import get_eval_prob, get_move_probs, get_batch_legal_moves
from chessengine.preprocessing.position_parsing import encode_board
from chessengine.model.utils import compute_topk_coverage, pad_and_stack

def time_method(func):
    """Decorator to optionally time a method based on instance timing_enabled."""
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "timing_enabled", True):  # Default to True if attribute missing
            return func(self, *args, **kwargs)
        
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        elapsed = end - start
        
        # Register time in BeamTree
        if func.__name__ == "_evaluate":
            batch_size = self.layers[-1].num_positions
            self.eval_time[batch_size] = elapsed
        elif func.__name__ == "_expand_layer":
            batch_size = self.layers[-1].num_positions
            self.expand_time[batch_size] = elapsed
        
        return result
    return wrapper

class BeamPosition:
    def __init__(self, board, parent=None, move=None):
        self.board = board              # chess.Board
        self.color = board.turn         # Ture = white, False = black
        self.parent = parent            # int (index in previous depth layer)
        self.move = move                # chess.Move (move leading to this position)
        self.children = []              # List[BeamPosition] (children of this position)
        self.eval = None                # torch.Tensor (3,) after evaluation
        self.chosen_moves = None        # List[chess.Move] topk predicted moves
        self.chosen_moves_prob = None   # torch.Tensor (topk,) move probabilities
        self.is_expanded = False        # Flag to know if expanded yet
        self.best_child = None          # BeamPosition (best child from this position)
        self.is_terminal = board.is_game_over()         # Flag to know if terminal position

    def get_prediction(self, moves: List[chess.Move], ps: torch.Tensor, eval: torch.Tensor):
        """
        Assign model outputs to this position.
        Args:
            moves: List of top-k moves
            eval: Evaluation tensor (3,)
        """
        self.chosen_moves = moves
        self.eval = eval
        self.chosen_moves_prob = ps

    def expand(self) -> List['BeamPosition']:
        """
        Expand the current position by selecting top-k moves.
        Returns list of new BeamPosition objects.
        """
        if self.is_terminal:
            return []
        if self.chosen_moves is None:
            raise ValueError("Cannot expand without chosen moves.")

        for move_idx, move in enumerate(self.chosen_moves):
            new_board = self.board.copy()
            new_board.push(move)
            child = BeamPosition(new_board, parent=self, move=move)
            self.children.append(child)

        # Find moves that lead to checkmate. If any exists, all other moves can be ignored.
        decisive_positions = [child for child in self.children if child.board.is_checkmate()]
        if decisive_positions:
            self.children = decisive_positions[:1] # Only need to keep one

        return self.children

    def clear_board(self):
        """Optionally free board memory if no longer needed."""
        self.board = None
        self.chosen_moves = None
        self.chosen_moves_prob = None
    
    def backpropagate(self): 
        """Backpropagate evaluations from children to this position. Returns best move."""
        children_with_evals = [child for child in self.children if child.eval is not None]
        evals = torch.stack([child.eval for child in children_with_evals], dim=0)  # (k, 3)
        if not children_with_evals:
            return  # No evaluations to backpropagate
        
        scalar_eval = evals[:, 2] - evals[:, 0]  # (k,) white's perpective
        idx = torch.argmax(scalar_eval) if self.color else torch.argmin(scalar_eval)
        self.eval = evals[idx]  # (3,)
        self.best_child = children_with_evals[idx]  # best child

    def prob_from_root(self):
        """Return probability of this position from root."""
        if self.parent is None:
            return 1.0
        else:
            return (
                [p for m, p in zip(self.parent.chosen_moves, self.parent.chosen_moves_prob) if m == self.move][0] *  
                self.parent.prob_from_root()
            )

class BeamLayer:
    def __init__(self, positions: List[BeamPosition], layer_num=0):
        self.positions = positions  # List[BeamPosition]
        self.layer_num = layer_num   # int (depth level in the tree)
        self.num_positions = len(positions)  # Number of positions in this layer
        self.tree = None  # Reference to the parent BeamTree
        self.coverage = None  # Coverage of top-k moves for this layer

    def get_boards(self, active_only=False):
        """Return a list of all boards for batch prediction."""
        if active_only:
            return [p.board for p in self.positions if p.board is not None and not p.is_terminal]
        return [p.board for p in self.positions if p.board is not None]
    
    def get_active_positions(self):
        """Return a list of all active positions."""
        return [p for p in self.positions if p.board is not None and not p.is_terminal]
    
    def all_terminal(self):
        """Check if all positions in this layer are terminal."""
        return all(p.is_terminal for p in self.positions)
    
    def set_tree(self, tree):
        """Set the parent tree reference."""
        self.tree = tree
    
    def prepare_batch(self, device="cuda"):
        """Prepare batch of inputs for model prediction."""
        active_positions = self.get_active_positions()
        boards = [p.board for p in active_positions]
        if not boards:
            return

        # Prepare batch
        x_batch = [encode_board(b) for b in boards]          # List[Tensor (8,8,21)]
        x_batch = torch.stack(x_batch).float().to(device)    # (B, 8, 8, 21)

        legal_moves_tensor, legal_move_lists = get_batch_legal_moves(boards) # (B, 64, L), [List[chess.Move]]
        legal_moves_tensor = legal_moves_tensor.long().to(device)

        return x_batch, legal_moves_tensor, legal_move_lists, active_positions
        
    def predict_moves(self, model, device="cuda", out_device="cpu", topk=5):
        """
        Predict top-k moves and evaluations for the current layer.
        Distribute only top-k moves instead of all legal moves.
        """
        batch = self.prepare_batch(device=device)
        if batch is None:
            self.set_terminal_evals()
            return
        
        x_batch, legal_moves_tensor, legal_move_lists, active_positions = batch
        L = legal_moves_tensor.shape[-1] # max number of legal moves
        topk = min(topk, L)
        # Run model
        with torch.no_grad():
            model.eval()
            model.to(device)
            x_out, move_pred = model(x_batch)  # (B, 65, H), (B, 64, 7)
            eval_probs = get_eval_prob(model, x_out, device=out_device)  # (B, 3)
            move_probs = get_move_probs(move_pred, legal_moves_tensor, device=out_device)  # (B, L)
        
        self.coverage = compute_topk_coverage(move_probs, max_k=10)  # (B, topk)

        _, move_idx = torch.topk(move_probs, topk, dim=-1)  # (B, topk)
        self.assign_predictions(legal_move_lists, move_probs, move_idx, eval_probs, active_positions)

    def assign_predictions(self, legal_move_lists, move_probs, move_idx, eval_probs, active_positions):
        """Assign model outputs back to corresponding BeamPosition."""
        #active_positions = [p for p in self.positions if p.board is not None and not p.is_terminal]
        assert len(active_positions) == len(legal_move_lists), "Mismatch in number of positions and legal move lists."
        
        for idx, lm, ps, eval, pos in zip(move_idx, legal_move_lists, move_probs, eval_probs, active_positions):
            selected_idx = [j for j in idx.tolist() if j < len(lm)]
            moves = [lm[j] for j in selected_idx] 
            ps = ps[selected_idx]  # (topk,)
            pos.get_prediction(moves, ps, eval) # top-k moves, eval (3,)
        
        # Set evaluations for terminal positions
        self.set_terminal_evals()

    def set_terminal_evals(self):
        """Set evaluations for terminal positions."""
        terminal_positions = [p for p in self.positions if p.is_terminal]
        for pos in terminal_positions:
            result = pos.board.result()
            if result == "1-0":
                pos.eval = torch.tensor([0.0, 0.0, 2.0]) # the unrealistic value of 2.0 to numerically dominate over other values
            elif result == "0-1":
                pos.eval = torch.tensor([2.0, 0.0, 0.0]) # same here
            else:
                pos.eval = torch.tensor([0.0, 1.0, 0.0])
        
    def expand_layer(self, layer_num, clear_boards=True) -> 'BeamLayer':
        """Expand all positions to next depth level."""
        new_positions = []
        for pos in self.positions:
            if pos.children or pos.is_terminal:
                continue
            children = pos.expand()
            new_positions.extend(children)
            if layer_num != 1 and clear_boards:
                pos.clear_board()  # Free memory if desired
        return BeamLayer(new_positions, layer_num=layer_num)
    
    def backpropagate(self):
        for pos in self.positions:
            if pos.children:
                pos.backpropagate()

class BeamTree:
    def __init__(self, model, device="cuda", topk_schedule=None, timing_enabled=False, clear_boards=False):
        self.model = model
        self.device = device
        self.topk_schedule = topk_schedule
        self.timing_enabled = timing_enabled
        self.clear_boards = clear_boards
        self.layers = None
        self.eval_time = {}
        self.expand_time = {}
        self.best_moves_list = None  # Best moves from root to leaf
        self.roots_eval = None  # Evaluation of the root positions

    def setup(self, root_boards):
        """Setup the tree with one or multiple root boards."""
        if not isinstance(root_boards, list):
            root_boards = [root_boards]
        root_positions = [BeamPosition(board) for board in root_boards]
        self.layers = [BeamLayer(root_positions, layer_num=0)]
        self.layers[0].set_tree(self)

    def all_terminal(self):
        """Check if all positions in the tree are terminal."""
        return self.layers[-1].all_terminal()

    @time_method
    def _evaluate(self, k=5):
        """Expand the tree by one depth layer."""
        current_layer = self.layers[-1]
        
        # Predict moves and evaluations
        current_layer.predict_moves(self.model, device=self.device, topk=k)

    @time_method
    def _expand_layer(self):
        layer_num = len(self.layers)
        current_layer = self.layers[-1]

        # Expand positions into a new layer
        next_layer = current_layer.expand_layer(layer_num=layer_num, clear_boards=self.clear_boards)
        next_layer.set_tree(self)  # <-- tell layer who owns it

        # Optionally clear boards from current layer (memory optimization)
        #current_layer.clear_boards()

        # Append new layer to the tree
        self.layers.append(next_layer)

    def expand_to_depth(self, depth=5, k=5):
        """Expand tree to a fixed depth."""
        if self.topk_schedule is not None:
            depth = len(self.topk_schedule)
        for d in range(depth):
            if self.all_terminal():
                break
            
            if self.topk_schedule is not None:
                k = self.topk_schedule[d]

            #print(f"Expanding layer {len(self.layers)} with top-{k}...")
            self._evaluate(k=k)
            self._expand_layer()

        self._evaluate(k=k)

    def backpropagate(self):
        """Propagate evaluations from leaves up to root."""
        for layer in reversed(self.layers[:-1]):
            layer.backpropagate()

        # best moves from each root separately
        self.best_moves_list = []
        self.roots_eval = []
        for root in self.layers[0].positions:
            if root.eval is None:
                continue  # skip if somehow root never evaluated
            self.roots_eval.append(root.eval)
            moves = []
            node = root
            while node.best_child is not None:
                node = node.best_child
                moves.append(node.move)
            self.best_moves_list.append(moves)

    def get_best_moves(self):
        """Return list of best move sequences from all roots."""
        if self.best_moves_list is None:
            raise ValueError("Tree has not been expanded or backpropagated.")
        return self.best_moves_list 

    def get_all_positions(self):
        """Return a flat list of all BeamPositions in the tree."""
        return [pos for layer in self.layers for pos in layer.positions]
    
    def get_coverage_data(self):
        """Return coverage data for all layers."""
        coverage_data = [layer.coverage for layer in self.layers if layer.coverage is not None]
        if not coverage_data:
            raise ValueError("No coverage data available.")
        return pad_and_stack(coverage_data, BATCH_DIM=True, pad_value=1.0)  # (B, topk)