import torch
from tqdm import tqdm
from pytorchchess import TorchBoard
from .beam_search import BeamSearchState
from .position_queue import PositionQueue
from ..utils.profiler import profiler, auto_profile_class

def move_to_notation(moves):
    """
    Convert move tensor to human-readable notation.
    
    Args:
        moves: (G, M) tensor of move integers
        
    Returns:
        notation: (G, M) list of lists with f"{from_sq} -> {to_sq}" strings
        move_type: (G, M) tensor of move types
    """
    moves = moves.to(torch.int64)  # (G, M)
    from_sq = moves % 64
    to_sq = (moves // 64) % 64
    move_type = moves // 4096

    # change to standard chess notation
    file_map = 'abcdefgh'
    rank_map = '12345678'
    
    def idx_to_coord(idx):
        f, r = idx % 8, idx // 8
        return f"{file_map[f]}{rank_map[r]}"

    # Convert each position index to coordinate notation
    # Process each game and each move within that game
    notation = []
    for game_idx in range(moves.shape[0]):  # For each game
        game_moves = []
        for move_idx in range(moves.shape[1]):  # For each move in the game
            from_coord = idx_to_coord(from_sq[game_idx, move_idx].item())
            to_coord = idx_to_coord(to_sq[game_idx, move_idx].item())
            move_notation = f"{from_coord} -> {to_coord}"
            game_moves.append(move_notation)
        notation.append(game_moves)

    return notation, move_type

class BeamSearchEngine:
    """High-level beam search engine that manages the full search cycle"""
    
    def __init__(
            self, 
            model, 
            expansion_factors, 
            device="cpu", 
            pv_depth=3, 
            verbose=False, 
            profile=False,
            seed=None
        ):
        """
        Args:
            model: Chess neural network model
            expansion_factors: (D,) expansion factors for each depth
            device: Device for computation
            pv_depth: Number of PV moves to actually play (e.g., 3)
        """
        self.model = model
        self.expansion_factors = expansion_factors.clone().to(device=device)
        self.device = device
        self.pv_depth = pv_depth
        self.L = len(expansion_factors)
        
        # Use PositionQueue instead of raw dictionary
        self.position_queue = None
        self.num_games = None
        self.beam_state = None
        self.beam_boards = None
        self.step = 0
        
        # Create cycling iterator for continuous pipeline
        self.layer_cycle = None
        self.VERBOSE = verbose  # Enable verbose output for debugging
        
        # Enable/disable profiling
        if profile:
            profiler.enable()
            # Apply profiling decorators automatically
            self._setup_profiling()
        else:
            profiler.disable()

        if seed is not None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)
        else:
            self.generator = None
    
    def _setup_profiling(self):
        """Centralized profiling setup - all timing config in one place"""
        profile_config = {
            # Model operations
            '_model_forward_pass': 'model.forward',
            
            # TorchBoard operations  
            '_board_get_legal_moves': 'board.legal_moves',
            '_board_feature_extraction': 'board.features',
            '_board_is_game_over': 'board.game_over',
            '_board_push_moves': 'board.push_moves',
            '_board_select_positions': 'board.selection',
            '_board_get_topk': 'board.get_topk',
            '_board_apply_moves_to_roots': 'board.apply_moves',   
            '_board_add_new_layer': 'board.add_layer',            
            
            # BeamSearchState operations
            '_beam_expand_positions': 'beam.expansion',
            '_beam_store_early_terminated_evaluations': 'beam.store_early_term',
            '_beam_store_evaluations': 'beam.store_eval',
            '_beam_backpropagate': 'beam.backprop',
            '_beam_add_stack': 'beam.add_stack',
            '_beam_store_moves': 'beam.store_moves',
            '_beam_get_finished_stack': 'beam.get_finished',
            '_beam_finished_expansion': 'beam.finished_exp',
            
            # # Coordinator methods
            # '_terminal_check': 'coord.terminal',
            # '_expand_positions_by_component': 'coord.expansion',
            # '_beam_handle_completed_stacks': 'coord.completed_stacks',
            # '_continue_pipeline': 'coord.pipeline',
            
            # # High-level coordination
            # 'step_search': 'engine.step',
        }
        
        auto_profile_class(self.__class__, profile_config)
    
    def initialize(self, initial_boards):
        """Initialize with starting positions"""
        self.num_games = initial_boards.batch_size // (self.L + 1)
        
        # Create position queue
        self.position_queue = PositionQueue.from_board_list(
            initial_boards, 
            self.num_games, 
            self.device
        )
        
        # Create cycling iterator
        self.layer_cycle = self.position_queue.cycle_iterator(start_layer=0)
        
        print(f"Initialized: {self.position_queue}")
        print(f"Layer stats: {self.position_queue.get_layer_stats()}")
        
        # Initialize beam search
        self._start_new_beam_batch()
        
    def _start_new_beam_batch(self):
        """Start beam search with current layer positions"""
        self.beam_state = BeamSearchState.create(
            self.num_games, 
            self.expansion_factors, 
            device=self.device
        )
        
        # Get current layer positions for beam search
        current_layer = self.step % (self.L + 1)
        self.beam_boards = self.position_queue.get_layer(current_layer, clone=True)
        
    def step_search(self):
        """Perform one step of beam search"""
        if self.VERBOSE:
            print(f"\nBeam Search Step {self.step + 1}")
            print(f"Processing layer: {self.step % (self.L + 1)}")
            print(f"Beam positions: {self.beam_state.idx.shape[0]}")
        

        self._board_get_legal_moves()
        self._terminal_check()
        
        # if self.beam_state.idx.shape[0] == 0:
        #     return False
            
        x = self._board_feature_extraction()
        evaluations = self._model_forward_pass(x)
        
        evaluations = self._handle_finished_expansion(evaluations)
        
        # Pure BeamSearchState operations
        self._beam_handle_completed_stacks()
        
        # Mixed operations - separated by component
        if self.beam_state.idx.shape[0] > 0:
            self._expand_positions_by_component(evaluations)
        
        # Pure BeamSearchState + TorchBoard operations
        self._continue_pipeline()
        self.step += 1
    
    # ============= MODEL OPERATIONS =============
    def _model_forward_pass(self, x):
        """Pure model forward pass"""
        with torch.no_grad():
            x_out, move_pred = self.model(x)
            x_out = x_out.detach()
            move_pred = move_pred.detach()

            eval_logits = self.model.loss_module.prob_eval_loss.head(x_out)
            prob_eval = torch.nn.functional.softmax(eval_logits, dim=-1)
            scalar_eval = prob_eval[:, 0] - prob_eval[:, 2]
        return scalar_eval, move_pred
    
    # ============= TORCHBOARD OPERATIONS =============
    def _board_get_legal_moves(self):
        """Pure TorchBoard legal move computation"""
        self.beam_boards.get_legal_moves(get_tensor=True)
        if self.step > 88 and self.step < 95:
            lm = self.beam_boards.get_legal_moves(get_tensor=True)
            print("-" * 40)
            print(f"Step {self.step}: Legal moves tensor shape: {lm.shape}")
            print(f"Legal moves tensor: ")
            print(lm.encoded)
            print("-" * 40)
    def _board_feature_extraction(self):
        """Pure model feature extraction"""
        return self.beam_boards.feature_tensor().float()
    
    def _board_is_game_over(self):
        return self.beam_boards.is_game_over(
            max_plys=300,
            enable_fifty_move_rule=True,
            enable_insufficient_material=True,
            enable_threefold_repetition=True,
        )
    
    def _board_push_moves(self, move_data):
        """Pure TorchBoard move application"""
        moves, board_idx, _, _ = move_data
        return self.beam_boards.push(moves, board_idx)
    
    def _board_select_positions(self, mask):
        """Pure TorchBoard position selection"""
        return self.beam_boards.select(mask)
    
    def _board_get_topk(self, evaluations):
        _, move_pred = evaluations
        move_data = self.beam_boards.get_topk_legal_moves(
            move_pred, 
            ks=self.expansion_factors[self.beam_state.depth],
            sample=True,
            temp=3.0,
            generator=self.generator
        )
        return move_data # (new_moves, board_idx, move_indices, ks)
    
    def _board_apply_moves_to_roots(self, pv_data):
        pv_values, pv_moves, target_layer = pv_data
        if pv_values is not None:
            moves_to_apply = pv_moves[:, :self.pv_depth]
            #print(f"Applying {moves_to_apply} PV moves to layer {target_layer}")
            self.position_queue.apply_moves_to_layer(target_layer, moves_to_apply)
            if self.step % (self.L + 1) == 1:
                print(f"Applied PV moves to layer {target_layer}: {moves_to_apply}") if self.step > 80 else None
                notation, _ = move_to_notation(moves_to_apply)
                plys = self.position_queue.get_layer(target_layer).state.plys
                print(f"Current plys: {plys}") if self.step > 80 else None
                print(f"Move notation: {notation}") if self.step > 80 else None
                print() if self.step > 80 else None
                
                
            # terminal, result = self.position_queue[target_layer].is_game_over(
            #     max_plys=300,
            #     enable_fifty_move_rule=True,
            #     enable_insufficient_material=True,
            #     enable_threefold_repetition=True,
            # )
            # if terminal.any():
            #     print(f"Layer {target_layer} positions game over: {terminal}")
            #     print(f"Results: {result}")
        else:
            print("No PV values to apply - skipping move application")

    def _board_add_new_layer(self, next_layer):
        next_positions = self.position_queue.get_layer(next_layer, clone=True)
        self.beam_boards = self.beam_boards.concat(next_positions)
    
    # ============= BEAM SEARCH STATE OPERATIONS =============
    def _beam_expand_positions(self, move_data):
        """Pure BeamSearchState expansion"""
        _, _, move_indices, ks = move_data
        return self.beam_state.expand_positions(move_indices, ks)
    
    def _beam_store_early_terminated_evaluations(self, dead_positions, results):
        """Pure BeamSearchState early termination evaluation storage"""
        self.beam_state.store_early_terminated_evaluations(dead_positions, results)
        if self.step > 80:
            evals = self.beam_state.evaluations.flatten(start_dim=1)
            eval_stack = self.beam_state.stack
            print(f"Evaluations shape: {evals}")
            print(f"Evaluation stack: {eval_stack}")
    
    def _beam_store_evaluations(self, mask, evaluations):
        """Pure BeamSearchState evaluation storage"""
        self.beam_state.store_terminal_evaluations(mask, evaluations)
    
    def _beam_backpropagate(self, finished_stack):
        """Pure BeamSearchState backpropagation"""
        return self.beam_state.backpropagate(finished_stack) # (pv_values, pv_moves, target_layer)
    
    def _beam_add_stack(self, step):
        """Pure BeamSearchState stack addition"""
        return self.beam_state.add_new_stack(step)
    
    def _beam_store_moves(self, move_data):
        new_moves, _, _, _ = move_data
        if self.step > 80:
            idx = self.beam_state.idx
            game = self.beam_state.game
            stack = self.beam_state.stack
            print(f"Step {self.step}")
            print(f"idx = {idx}")
            print(f"stack = {stack}")
            print(f"New moves: {new_moves}")
        return self.beam_state.store_moves(new_moves)
    
    def _beam_get_finished_stack(self):
        """Pure BeamSearchState finished stack retrieval"""
        return self.beam_state.get_finished_stack(self.step)
    
    def _beam_finished_expansion(self):
        """Pure BeamSearchState finished expansion check"""
        return self.beam_state.get_finished_expansion()
    
    # ============= COORDINATED OPERATIONS =============
    def _terminal_check(self):
        """Pure TorchBoard terminal position detection"""
        dead_positions, results = self._board_is_game_over()

        print(f"Dead positions: {dead_positions}") if self.step > 80 else None
        print(f"Results: {results}") if self.step > 80 else None
        
        if dead_positions.any():
            self._beam_store_early_terminated_evaluations(dead_positions, results)
            # Pure board operations
            self.beam_boards = self.beam_boards.select(~dead_positions)
            self.beam_state = self.beam_state[~dead_positions]

    def _expand_positions_by_component(self, evaluations):
        """Expansion separated by component"""
        
        move_data = self._board_get_topk(evaluations) 
        self.beam_state = self._beam_expand_positions(move_data)
        self.beam_boards = self._board_push_moves(move_data)
        self._beam_store_moves(move_data)
    
    def _beam_handle_completed_stacks(self):
        """Handle PV extraction - pure beam operations"""
        finished_stack = self._beam_get_finished_stack()
        if finished_stack is not None:
            print(f"Step {self.step}: Found finished stack {finished_stack}") if self.step > 80 else None
            pv_data = self._beam_backpropagate(finished_stack)
            self._board_apply_moves_to_roots(pv_data)

        else:
            print(f"Step {self.step}: No finished stack") if self.step > self.L else None
    
    def _continue_pipeline(self):
        """Pipeline continuation - mixed operations"""
        self.beam_state = self._beam_add_stack(self.step + 1)
        next_layer = (self.step + 1) % (self.L + 1)
        self._board_add_new_layer(next_layer)
    
    def _handle_finished_expansion(self, evaluations):
        """Handle finished expansion - now mostly coordination"""
        scalar_eval, move_pred = evaluations
        finished_expansion = self._beam_finished_expansion()
        
        if finished_expansion.any():
            # BeamSearchState operation
            self._beam_store_evaluations(finished_expansion, scalar_eval[finished_expansion])
            
            # TorchBoard operations  
            self.beam_boards = self.beam_boards.select(~finished_expansion)
            self.beam_state = self.beam_state[~finished_expansion]
            
            if self.beam_state.idx.shape[0] > 0:
                move_pred = move_pred[~finished_expansion]
                scalar_eval = scalar_eval[~finished_expansion]
                
        return scalar_eval, move_pred
    
    def run_full_search(self, max_iterations=50):
        """Run complete beam search until PV moves are found"""
        profiler.reset()
        iterator = range(max_iterations) if self.VERBOSE else tqdm(range(max_iterations), desc="Search", unit="iter")
        for iteration in iterator:
            self.step_search()
            
            # Terminal check
            all_positions = self.position_queue.get_all_positions()
            terminal, result = all_positions.is_game_over(
                max_plys=300, enable_fifty_move_rule=True,
                enable_insufficient_material=True, enable_threefold_repetition=True
            )
            
            if terminal.all():
                print(f"All positions terminal after {iteration + 1} iterations")
                profiler.print_summary()
                return True
                
        profiler.print_summary()
        return False
        
    def get_current_root_positions(self):
        """Get the current root positions (layer 0) for analysis"""
        return self.position_queue[0]
        
    def get_all_positions(self):
        """Get all positions in the queue"""
        return self.position_queue.get_all_positions()

