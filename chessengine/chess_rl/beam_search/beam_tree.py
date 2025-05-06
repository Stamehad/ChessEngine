import time
import torch
import chess
from typing import List, Tuple
from chessengine.preprocessing.position_parsing import generate_legal_move_tensor
from chessengine.model.utils import pad_and_stack
from chessengine.model.prediction import get_eval_prob, get_move_probs, get_batch_legal_moves
from chessengine.preprocessing.position_parsing import encode_board
from chessengine.model.utils import compute_topk_coverage, pad_and_stack

import psutil, os
_proc = psutil.Process(os.getpid())

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
    def __init__(self, board, parent=None, move=None, depth=0, MAX_PLIES=300):
        self.board = board              # chess.Board
        self.color = board.turn         # Ture = white, False = black
        self.parent = parent            # int (index in previous depth layer)
        self.move = move                # chess.Move (move leading to this position)
        self.children = []              # List[BeamPosition] (children of this position)
        self.eval = None                # torch.Tensor (3,) after evaluation
        self.chosen_moves = None        # List[chess.Move] topk predicted moves
        self.chosen_moves_prob = None   # torch.Tensor (topk,) move probabilities
        self.best_child = None          # BeamPosition (best child from this position)
        self.is_terminal = board.is_game_over(claim_draw=True)         # Flag to know if terminal position, take technical draws when possible
        self.depth = depth              # int, depth in the tree
        self.descendants_pending = 0   # number of unfinished sub‑trees below me
        self._backprop_done = False    # flag so back‑prop only runs once
        self.MAX_PLIES = MAX_PLIES

    @property
    def need_eval(self):
        return (self.eval is None) and (not self.is_terminal)
    
    def game_stop(self):
        return self.board.fullmove_number * 2 >= self.MAX_PLIES or self.is_terminal or self.board is None

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

    def get_game_result(self):
        result = self.board.result()
        if result == "1-0":
            self.eval = torch.tensor([0.0, 0.0, 2.0]) # the unrealistic value of 2.0 to numerically dominate over other values
        elif result == "0-1":
            self.eval = torch.tensor([2.0, 0.0, 0.0]) # same here
        else:
            self.eval = torch.tensor([0.0, 1.0, 0.0])

    def expand(self) -> List['BeamPosition']:
        """
        Expand the current position by selecting top-k moves.
        Returns list of new BeamPosition objects.
        """
        if self.children:
            return self.children
        if self.is_terminal:
            return []
        if self.chosen_moves is None:
            raise ValueError("Cannot expand without chosen moves.")

        for move in self.chosen_moves:
            new_board = self.board.copy()
            new_board.push(move)
            child = BeamPosition(new_board, parent=self, move=move, depth=self.depth+1)
            self.children.append(child)

        # Find moves that lead to checkmate. If any exists, all other moves can be ignored.
        decisive_positions = [child for child in self.children if child.board.is_checkmate()]
        if decisive_positions:
            self.children = decisive_positions[:1] # Only need to keep one
        # after appending all children
        self.descendants_pending = len(self.children)
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

    # ──────────────────────────────────────────────────────────
    #  Back‑prop cascade helpers
    # ──────────────────────────────────────────────────────────
    def _child_finished(self):
        """Called by a child when its entire sub‑tree has back‑propagated."""
        self.descendants_pending -= 1
        if self.descendants_pending == 0:
            self._try_backpropagate()

    def _try_backpropagate(self):
        """Run back‑prop once, then notify my parent."""
        if self._backprop_done:
            return
        # All children are finished; compute my eval from theirs
        if self.children:
            self.backpropagate()
            # ----- memory cleanup: break sub‑tree links -----
            for ch in self.children:
                ch.parent = None      # break upward cycle
            self.children = []        # drop sibling refs
        self._backprop_done = True
        if self.parent:
            self.parent._child_finished()

    def finish_leaf(self):
        """
        Call this exactly once on every leaf node (terminal or depth‑max)
        after it has its eval.  Triggers the upward cascade.
        """
        if not self._backprop_done:          # sanity
            self._backprop_done = True
            if self.parent:
                self.parent._child_finished()

    def prob_from_root(self):
        """Return probability of this position from root."""
        if self.parent is None:
            return 1.0
        else:
            return (
                [p for m, p in zip(self.parent.chosen_moves, self.parent.chosen_moves_prob) if m == self.move][0] *  
                self.parent.prob_from_root()
            )
        
    # ──────────────────────────────────────────────────────────
    #  Utility helpers used by TrainingBeamTree
    # ──────────────────────────────────────────────────────────
    def extract_pv(self, max_len=3):
        """Return the first `max_len` moves of the principal variation."""
        pv = []
        node = self
        while node.best_child is not None and len(pv) < max_len:
            node = node.best_child
            pv.append(node.move)
        return pv

    def clean_up_pv(self):
        """Remove board and PV chain to allow GC of the finished mini‑tree."""
        node = self
        while node:
            node.board = None
            nxt = node.best_child
            node.best_child = None
            node = nxt


class BeamLayer:
    def __init__(self, positions: List[BeamPosition], layer_num=0, topk_schedule={}):
        self.positions = positions  # List[BeamPosition]
        self.layer_num = layer_num   # int (depth level in the tree)
        self.num_positions = len(positions)  # Number of positions in this layer
        self.coverage = None  # Coverage of top-k moves for this layer
        self.topk_schedule = topk_schedule  # Top-k schedule for this layer

    def concat(self, other: 'BeamLayer'):
        """Concatenate another layer into this one."""
        self.positions.extend(other.positions)
        self.num_positions = len(self.positions)
        
    def get_boards(self, active_only=False):
        """Return a list of all boards for batch prediction."""
        if active_only:
            return [p.board for p in self.positions if p.board is not None and not p.is_terminal]
        return [p.board for p in self.positions if p.board is not None]
    
    def get_active_positions(self):
        """Return a list of all active positions."""
        return [p for p in self.positions if not p.game_stop()]
    
    def to_expand(self):
        """Return a list of positions that need to be expanded."""
        return [p for p in self.positions if not p._backprop_done]
    
    def all_terminal(self):
        """Check if all positions in this layer are terminal."""
        return all(p.is_terminal for p in self.positions)
    
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
            x_out, move_pred = model(x_batch)  # (B, 65, H), (B, 64, 7)
            eval_probs = get_eval_prob(model, x_out, device=out_device)  # (B, 3)
            move_probs = get_move_probs(move_pred, legal_moves_tensor, device=out_device)  # (B, L)
        
        #self.coverage = compute_topk_coverage(move_probs, max_k=10)  # (B, topk)

        _, move_idx = torch.topk(move_probs, topk, dim=-1)  # (B, topk)
        self.assign_predictions(legal_move_lists, move_probs, move_idx, eval_probs, active_positions)

    def assign_predictions(self, legal_move_lists, move_probs, move_idx, eval_probs, active_positions):
        """Assign model outputs back to corresponding BeamPosition."""
        assert len(active_positions) == len(legal_move_lists), "Mismatch in number of positions and legal move lists."
        
        for idx, lm, ps, eval, pos in zip(move_idx, legal_move_lists, move_probs, eval_probs, active_positions):
            topk = self.topk_schedule.get(pos.depth, 5)  # Get top-k for this layer
            idx = idx[:topk]  # (topk,)
            selected_idx = [j for j in idx.tolist() if j < len(lm)]
            moves = [lm[j] for j in selected_idx] 
            ps = ps[selected_idx]  # (topk,)
            pos.get_prediction(moves, ps, eval) # top-k moves, eval (3,)
            # If this position is a leaf (terminal or depth‑max), start cascade
            if pos.depth >= len(self.topk_schedule):  # depth_max is layer_num here
                pos.finish_leaf()
        # Set evaluations for terminal positions
        self.set_terminal_evals()

    def set_terminal_evals(self):
        """Set evaluations for terminal positions."""
        terminal_positions = [p for p in self.positions if p.game_stop()]
        for pos in terminal_positions:
            pos.get_game_result()
            pos.finish_leaf()  # Start backpropagation cascade
        
    def expand_layer(self, layer_num, clear_boards=True) -> 'BeamLayer':
        """Expand all positions to next depth level."""
        new_positions = []
        for pos in self.to_expand():
            children = pos.expand()
            new_positions.extend(children)
            if pos.depth > 0 and clear_boards:
                pos.clear_board()  # Free memory if desired
        return BeamLayer(new_positions, layer_num=layer_num, topk_schedule=self.topk_schedule)
    
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
        self.layers = [BeamLayer(root_positions, layer_num=0, topk_schedule=self.topk_schedule)]

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

class TrainingBeamTree:
    def __init__(self, model, device="cuda", topk_schedule=None, pv_skip=3):
        self.model = model
        self.device = device
        self.topk_schedule = topk_schedule or {0: 8, 1: 5, 2: 3, 3: 2, 4: 1}
        self.pv_skip = pv_skip
        self.current_layer = None
        self.packet_queue = []
        self.depth_counter = 0
        self.active_roots = []          # list of current root BeamPositions
        self._gid_counter = 0            # unique game ids
        self._trajectories = {}          # gid → list[(fen, move_uci)]
        self.training_records = []       # final (fen, move, outcome)

    def setup(self, root_boards, n_packets=8):
        """Initialize the packets from root boards."""
        packets = [[] for _ in range(n_packets)]
        for idx, board in enumerate(root_boards):
            packets[idx % n_packets].append(board)

        def _make_root(board):
            root = BeamPosition(board, depth=0)
            root.game_id = self._gid_counter
            self._gid_counter += 1
            self._trajectories[root.game_id] = []
            return root

        self.packet_queue = [
            BeamLayer([_make_root(b) for b in packet],
                      layer_num=i,
                      topk_schedule=self.topk_schedule)
            for i, packet in enumerate(packets)
        ]

        self.current_layer = self.packet_queue.pop(0)

        # register all initial roots
        self.active_roots = [pos for layer in self.packet_queue + [self.current_layer]
                             for pos in layer.positions]

    def play_until_done(self):
        """
        Self-play loop:
          • evaluate + expand frontier
          • when a root finishes its sub-tree:
              - store ⟨fen, move, outcome⟩ for all positions in the trajectory
              - follow principal variation `pv_skip` moves
              - reseed as a new root unless game ended
        """
        topk_max = max(self.topk_schedule.values())
        i = 0  
        while self.active_roots:
            # 1) evaluate front‑tier
            self.current_layer.predict_moves(self.model, device=self.device, topk=topk_max)

            # 2) expand
            self.current_layer = self.current_layer.expand_layer(layer_num=self.current_layer.layer_num + 1)

            # 3) harvest finished roots and reseed packet_queue
            finished = [r for r in self.active_roots if r._backprop_done]
            #print(f"Layer {self.current_layer.layer_num} expand {num1} -> {num2} positions. Finished: {num_finished} roots. ")
            new_roots = []
            for root in finished:
                gid = getattr(root, "game_id", None)
                best_mv = root.best_child.move if root.best_child else None
                fen = root.board.fen()
                self._trajectories[gid].append((fen, best_mv.uci() if best_mv else None))

                # (b) advance PV
                pv_moves = root.extract_pv(max_len=self.pv_skip)
                new_board = root.board.copy(stack=False)
                for mv in pv_moves:
                    new_board.push(mv)
                    if new_board.is_game_over(claim_draw=True):
                        break
                ply_limit_hit = new_board.fullmove_number * 2 >= root.MAX_PLIES
                if new_board.is_game_over(claim_draw=True) or ply_limit_hit:
                    # game ended → determine outcome and flush trajectory
                    result = new_board.result()
                    outcome = 2 if result == "1-0" else 0 if result == "0-1" else 1
                    for rec_fen, rec_mv in self._trajectories[gid]:
                        self.training_records.append((rec_fen, rec_mv, outcome))
                    del self._trajectories[gid]
                else:
                    # reseed continuation root
                    new_root = BeamPosition(new_board, depth=0)
                    new_root.game_id = gid
                    self.active_roots.append(new_root)
                    new_roots.append(new_root)

                # -- drop all references inside the old mini‑tree --
                root.clean_up_pv()
                root.children = []
                root.parent = None

                self.active_roots.remove(root)
            
            if new_roots:
                self.packet_queue.append(
                        BeamLayer(new_roots,
                                    layer_num=0,
                                    topk_schedule=self.topk_schedule))
            # 4) concat next packet if available
            if self.packet_queue:
                self.current_layer.concat(self.packet_queue.pop(0))

            if i % 100 == 0:                      # every 10 layers
                rss = _proc.memory_info().rss / 1024 / 1024
                print(f"[mem] layer {self.current_layer.layer_num:4d} "
                    f"RSS={rss:6.1f} MB   active={len(self.active_roots):3d}")

            i += 1