import torch
import chess 
from pytorchchess import TorchBoard

class PositionQueue:
    """
    Manages G * (D+1) chess positions organized into D+1 layers of G positions each.
    Supports cycling through layers and applying moves to specific layers.
    """
    
    def __init__(self, layers_dict, finished_dict, all_finished_dict, num_games, device):
        """
        Args:
            layers_dict: {layer_idx: TorchBoard} - each TorchBoard has G positions
            num_games: Number of games G
            device: Device for computations
        """
        self.layers = layers_dict       # {0: TorchBoard, 1: TorchBoard, ..., D: TorchBoard}
        self.finished = finished_dict   # {0: finished_mask, 1: finished_mask, ..., D: finished_mask}
        self.all_finished = all_finished_dict  # {0: all_finished_bool, ..., D: all_finished_bool}
        self.num_games = num_games
        self.device = device
        self.num_layers = len(layers_dict)
        
    @classmethod
    def from_board_list(cls, boards, num_games, device):
        """
        Create PositionQueue from list of G * (D+1) chess boards.
        
        Args:
            boards: TorchBoard or List of Chess.Board objects (length G * (D+1))
            num_games: Number of games G
            device: Device for computations
        """
        assert isinstance(boards, TorchBoard) or all(isinstance(b, chess.Board) for b in boards), \
            "boards must be a TorchBoard or a list of Chess.Board objects"
        assert len(boards) % num_games == 0, "Total positions must be divisible by number of games"
        assert type(num_games) is int and num_games > 0, "num_games must be a positive integer"
        
        if not isinstance(boards, TorchBoard):
            # Convert list of Chess.Board to TorchBoard
            boards = TorchBoard.from_board_list(boards, device=device)
        
        #total_positions = boards.batch_size
        total_positions = len(boards)  # Total number of positions
        num_layers = total_positions // num_games
        
        layers_dict = {}
        finished_dict = {}
        all_finished_dict = {}
        for layer in range(num_layers):
            start_idx = layer * num_games
            end_idx = start_idx + num_games
            layers_dict[layer] = boards[start_idx:end_idx]
            finished_dict[layer] = torch.zeros(num_games, dtype=torch.bool, device=device)
            all_finished_dict[layer] = False
            
        return cls(layers_dict, finished_dict, all_finished_dict, num_games, device)
    
    def __getitem__(self, layer_idx):
        """Get TorchBoard for specific layer (returns view, not clone)"""
        return self.layers[layer_idx]
    
    def __setitem__(self, layer_idx, torch_board):
        """Set TorchBoard for specific layer"""
        assert len(torch_board) == self.num_games, f"Expected {self.num_games} positions, got {len(torch_board)}"
        self.layers[layer_idx] = torch_board
    
    def __iter__(self):
        """Iterate through layers in order"""
        for layer_idx in range(self.num_layers):
            yield layer_idx, self.layers[layer_idx]
    
    def __len__(self):
        """Number of layers"""
        return self.num_layers
    
    def cycle_iterator(self, start_layer=0):
        """
        Create infinite iterator that cycles through layers.
        Useful for the continuous pipeline.
        """
        layer = start_layer
        while True:
            yield layer, self.layers[layer]
            layer = (layer + 1) % self.num_layers
    
    def get_layer(self, layer_idx, clone=False):
        """Return the TorchBoard for a specific layer."""
        board = self.layers[layer_idx]
        return board.clone() if clone else board

    def get_active_layer(self, layer_idx, clone=False):
        """Return unfinished positions and the finished mask for a layer."""
        board = self.layers[layer_idx]
        mask = self.finished[layer_idx]
        if clone:
            board = board.clone()
            mask = mask.clone()
        return board[~mask], mask

    def update_finished(self, layers, games):
        """
        Mark specific games as finished at given layer.
        
        Args:
            layers: (N,) tensor of layer indices (must be the same)
            games: (N,) tensor of game indices to mark as finished
        """
        if games.numel() == 0:
            return

        lay = layers.unique()
        assert lay.numel() == 1, "All layer indices must be the same"
        lay = int(lay.item())
        finished_mask = self.finished[lay]
        finished_mask[games] = True
        self.finished[lay] = finished_mask

        self.all_finished[lay] = finished_mask.all().item()
    
    def apply_moves_to_layer(self, layer_idx, moves_sequence):
        """
        Apply sequence of moves to specific layer.
        
        Args:
            layer_idx: Target layer index
            moves_sequence: (G, M) tensor where M is number of moves to apply
        """
        target_positions = self.layers[layer_idx]
        
        # Apply moves sequentially
        for move_depth in range(moves_sequence.shape[1]):
            moves_at_depth = moves_sequence[:, move_depth]  # (G,)
            batch_indices = torch.arange(self.num_games, device=self.device)
            
            # Push moves - creates new TorchBoard
            target_positions = target_positions.push(moves_at_depth, batch_indices)
        
        self.layers[layer_idx] = target_positions
    
    def get_all_positions(self):
        """Concatenate all layers into single TorchBoard for analysis"""
        all_layers = [self.layers[i] for i in range(self.num_layers)]
        result = all_layers[0]
        for layer_tb in all_layers[1:]:
            result = result.concat(layer_tb)
        return result
    
    def get_total_positions(self):
        """Get total number of positions across all layers"""
        return self.num_games * self.num_layers
    
    def get_layer_stats(self):
        """Get statistics about each layer for debugging"""
        stats = {}
        for layer_idx, torch_board in self:
            stats[layer_idx] = {
                'positions': torch_board.board_tensor.shape[0],
                'device': torch_board.device,
                'has_cache': hasattr(torch_board, 'cache') and torch_board.cache is not None
            }
        return stats
    
    def all_game_over(self):
        """
        Check if all root positions in all layers are game over.
        """
        all_finished = all(self.all_finished[layer] for layer in range(self.num_layers))
        return all_finished
        # for layer_idx in range(self.num_layers):
        #     positions = self.layers[layer_idx]
        #     terminal, _ = positions.is_game_over()
        #     if not terminal.all():
        #         return False
        # return True
    
    def __repr__(self):
        layer_sizes = [self.layers[i].board_tensor.shape[0] for i in range(self.num_layers)]
        return f"PositionQueue(layers={self.num_layers}, games={self.num_games}, layer_sizes={layer_sizes})"
