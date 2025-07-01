import torch
from typing import Optional, Tuple

class StoppingCondition:
    """
    Mixin class for TorchBoard to handle various game termination conditions.
    Follows the same pattern as InCheck, PseudoMoveGenerator, etc.
    """
    
    def is_game_over(self, 
                     max_plys: Optional[int] = None,
                     enable_fifty_move_rule: bool = True,
                     enable_insufficient_material: bool = True,
                     enable_threefold_repetition: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check for various game termination conditions.
        
        Args:
            max_plys: Maximum number of plys before forcing termination (None = no limit)
            enable_fifty_move_rule: Apply 50-move rule for draws
            enable_insufficient_material: Check for insufficient material draws
            enable_threefold_repetition: Check for threefold repetition (expensive)
        
        Returns:
            is_terminal: (B,) bool tensor indicating terminal positions
            results: (N_terminal,) int tensor with results (1=white win, 0=draw, -1=black win)
        """
        B = self.board_tensor.shape[0]
        device = self.device
        
        # Initialize tracking
        is_terminal = torch.zeros(B, dtype=torch.bool, device=device)
        all_results = []
        
        # 1. Standard chess termination (checkmate/stalemate)
        chess_terminal, chess_results = self._check_standard_termination()
        if chess_terminal.any():
            is_terminal[chess_terminal] = True
            all_results.extend(chess_results.tolist())
        
        # 2. Maximum ply limit
        if max_plys is not None:
            ply_terminal, ply_results = self._check_ply_limit(max_plys, is_terminal)
            if ply_terminal.any():
                is_terminal[ply_terminal] = True
                all_results.extend(ply_results.tolist())
        
        # 3. Fifty-move rule
        if enable_fifty_move_rule:
            fifty_terminal, fifty_results = self._check_fifty_move_rule(is_terminal)
            if fifty_terminal.any():
                is_terminal[fifty_terminal] = True
                all_results.extend(fifty_results.tolist())
        
        # 4. Insufficient material
        if enable_insufficient_material:
            material_terminal, material_results = self._check_insufficient_material(is_terminal)
            if material_terminal.any():
                is_terminal[material_terminal] = True
                all_results.extend(material_results.tolist())
        
        # 5. Threefold repetition (expensive)
        if enable_threefold_repetition:
            repetition_terminal, repetition_results = self._check_threefold_repetition(is_terminal)
            if repetition_terminal.any():
                is_terminal[repetition_terminal] = True
                all_results.extend(repetition_results.tolist())
        
        # Convert results to tensor
        if all_results:
            result_tensor = torch.tensor(all_results, dtype=torch.long, device=device)
        else:
            result_tensor = torch.empty(0, dtype=torch.long, device=device)
        
        return is_terminal, result_tensor
    
    def _check_standard_termination(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for checkmate and stalemate."""
        lm = self.get_legal_moves()
        no_moves = lm.get_terminal_boards()  # (B,) bool tensor
        
        if not no_moves.any():
            return no_moves, torch.empty(0, dtype=torch.long, device=self.device)
        
        in_check = self.in_check  # (B,)
        
        # Calculate results for terminal positions
        # 1 = white win, 0 = draw, -1 = black win
        results = torch.where(
            self.side_to_move[no_moves].view(-1) == 1,
            torch.where(in_check[no_moves], -1, 0),   # White to move: checkmate=black wins, stalemate=draw
            torch.where(in_check[no_moves], 1, 0)   # Black to move: checkmate=white wins, stalemate=draw
        )
        
        return no_moves, results
    
    def _check_ply_limit(self, max_plys: int, already_terminal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for games exceeding maximum ply count."""
        exceeded_plys = self.state.plys >= max_plys
        newly_terminal = exceeded_plys & ~already_terminal
        
        if newly_terminal.any():
            # Force draw for exceeded games
            num_exceeded = newly_terminal.sum().item()
            results = torch.zeros(num_exceeded, dtype=torch.long, device=self.device)  # 0 = draw
            return newly_terminal, results
        
        return newly_terminal, torch.empty(0, dtype=torch.long, device=self.device)
    
    def _check_fifty_move_rule(self, already_terminal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for fifty-move rule draws."""
        fifty_move_draw = self.state.fifty_move_clock >= 100  # 50 moves = 100 half-moves
        newly_terminal = fifty_move_draw & ~already_terminal
        
        if newly_terminal.any():
            num_fifty = newly_terminal.sum().item()
            results = torch.zeros(num_fifty, dtype=torch.long, device=self.device)  # 0 = draw
            return newly_terminal, results
        
        return newly_terminal, torch.empty(0, dtype=torch.long, device=self.device)
    
    def _check_insufficient_material(self, already_terminal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for insufficient material draws."""
        # Get piece counts for each position
        insufficient = self._has_insufficient_material()
        newly_terminal = insufficient & ~already_terminal
        
        if newly_terminal.any():
            num_insufficient = newly_terminal.sum().item()
            results = torch.zeros(num_insufficient, dtype=torch.long, device=self.device)  # 0 = draw
            return newly_terminal, results
        
        return newly_terminal, torch.empty(0, dtype=torch.long, device=self.device)
    
    def _has_insufficient_material(self) -> torch.Tensor:
        """
        Check for insufficient material to deliver checkmate.
        Returns (B,) bool tensor indicating positions with insufficient material.
        """
        B = len(self)
        device = self.device
        
        # Count pieces for each side (excluding kings and pawns)
        white_pieces, black_pieces, same_color_bishops = self._count_pieces()
        
        # Insufficient material cases:
        # 1. K vs K
        # 2. K+B vs K
        # 3. K+N vs K  
        # 4. K+B vs K+B (same color bishops)
        
        insufficient = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Case 1: K vs K (no pieces for either side)
        case1 = (white_pieces['total'] == 1) & (black_pieces['total'] == 1)
        insufficient |= case1
        
        # Case 2: K+B vs K or K vs K+B
        case2a = (white_pieces['total'] == 2) & (white_pieces['bishops'] == 1) & (black_pieces['total'] == 1)
        case2b = (black_pieces['total'] == 2) & (black_pieces['bishops'] == 1) & (white_pieces['total'] == 1)
        insufficient |= case2a | case2b
        
        # Case 3: K+N vs K or K vs K+N
        case3a = (white_pieces['total'] == 2) & (white_pieces['knights'] == 1) & (black_pieces['total'] == 1)
        case3b = (black_pieces['total'] == 2) & (black_pieces['knights'] == 1) & (white_pieces['total'] == 1)
        insufficient |= case3a | case3b
        
        # Case 4: K+B vs K+B with same color bishops would require more complex analysis
        case4 = (white_pieces['total'] == 2) & (black_pieces['total'] == 2) & same_color_bishops
        insufficient |= case4
        
        return insufficient
    
    def _count_pieces(self) -> Tuple[dict, dict]:
        """
        Count pieces for each side (excluding kings and pawns).
        Returns dictionaries with piece counts for white and black.
        """
        B = self.board_tensor.shape[0]
        device = self.device
        
        # Assuming board encoding where:
        # Positive values = white pieces, negative = black pieces
        # This would need to be adapted to your specific encoding scheme
        
        # Placeholder implementation - adapt to your board encoding
        white_pieces = {
            'total': self.occ_white().sum(dim=-1),  # Total white pieces
            'knights': (self.board_flat == 2).sum(dim=-1),
            'bishops': (self.board_flat == 3).sum(dim=-1),
            'rooks': (self.board_flat == 4).sum(dim=-1),
            'queens': (self.board_flat == 5).sum(dim=-1),
        }
        
        black_pieces = {
            'total': self.occ_black().sum(dim=-1),  # Total black pieces
            'knights': (self.board_flat == 8).sum(dim=-1),
            'bishops': (self.board_flat == 9).sum(dim=-1),
            'rooks': (self.board_flat == 10).sum(dim=-1),
            'queens': (self.board_flat == 11).sum(dim=-1),
        }
        
        light_squares = torch.arange(64, device=device)
        light_squares = (light_squares % 8 + light_squares // 8) % 2 == 0  # True for light squares
        # Check for bishops on same color squares
        wb = self.board_flat == 3 # (B, 64)
        bb = self.board_flat == 9 # (B, 64)
        one_each = (wb.sum(dim=-1) == 1) & (bb.sum(dim=-1) == 1) # (B,) 
        bishops = (wb | bb)  # (B, 64)
        same_color = bishops[:, light_squares].sum(dim=-1) == 2  # (B,)
        same_color |= bishops[:, ~light_squares].sum(dim=-1) == 2  # (B,)
        same_color_bishops = one_each & same_color

        return white_pieces, black_pieces, same_color_bishops
    
    def _compute_position_hash(self) -> torch.Tensor:
        """
        Compute Zobrist hash for threefold repetition detection.
        Returns: (B,) uint64 tensor of position hashes.
        """
        from chessengine.pytorchchess.utils.zobrist import get_hasher
        hasher = get_hasher(device=self.device)
        # Ensure these attributes exist with the right shapes/types:
        # - self.board_flat: (B, 64), 0=empty, 1-6=white, 7-12=black
        # - self.side_to_move: (B,), 1=white, 0=black
        # - self.state.castling: (B, 4)
        # - self.state.ep: (B,) 0..7 or -1
        return hasher.hash(
            self.board_flat,
            self.side_to_move,
            self.state.castling,
            self.state.ep
        )
    def _initialize_position_history(self):
        B = len(self)
        device = self.device
        # Start with empty history (B, 0)
        self.position_history = torch.empty((B, 0), dtype=torch.long, device=device)

    def _add_positions_to_history(self, position_hashes: torch.Tensor) -> torch.Tensor:
        """
        Concatenate new position hashes, check repetition count per board.
        position_hashes: (B,)
        Returns: (B,) repetition count for current hash (including this time)
        """
        # Step 1: If not initialized, create history
        if not hasattr(self, 'position_history') or self.position_history is None:
            self._initialize_position_history()

        # Step 2: Map accidental 0 hashes to 1 (to safely use 0 as pad)
        hashes = position_hashes.clone()
        hashes[hashes == 0] = 1

        # Step 3: Append new hash as a new column (B, 1)
        self.position_history = torch.cat([self.position_history, hashes.unsqueeze(1)], dim=1)  # (B, H+1)

        # Step 4: Compute repetition count for each board
        repetition_counts = (self.position_history == hashes.unsqueeze(1)).sum(dim=1)  # (B,)

        return repetition_counts

    def _check_threefold_repetition(self, already_terminal: torch.Tensor):
        """
        Check for threefold repetition draws.
        """
        B = self.board_tensor.shape[0]
        device = self.device

        # Compute current position hashes
        current_hashes = self._compute_position_hash()  # (B,)

        # Add to history and get repetition counts
        repetition_counts = self._add_positions_to_history(current_hashes)  # (B,)

        # Check for threefold repetition (>= 3 occurrences)
        repetition_draw = repetition_counts >= 3
        newly_terminal = repetition_draw & ~already_terminal

        if newly_terminal.any():
            num_repetition = newly_terminal.sum().item()
            results = torch.zeros(num_repetition, dtype=torch.long, device=device)  # 0 = draw
            return newly_terminal, results

        return newly_terminal, torch.empty(0, dtype=torch.long, device=device)
    
    def result_strings(self, 
                      max_plys: Optional[int] = None,
                      enable_fifty_move_rule: bool = True,
                      enable_insufficient_material: bool = True) -> list:
        """
        Get human-readable result strings for each position.
        
        Returns:
            List of strings: ["1-0", "0-1", "1/2-1/2", "*"] for each position
        """
        terminal_mask, results = self.is_game_over(
            max_plys=max_plys,
            enable_fifty_move_rule=enable_fifty_move_rule,
            enable_insufficient_material=enable_insufficient_material,
            enable_threefold_repetition=False  # Too expensive for routine use
        )
        
        result_strings = []
        result_idx = 0
        
        for i in range(self.board_tensor.shape[0]):
            if terminal_mask[i]:
                result_value = results[result_idx].item()
                if result_value == 1:
                    result_strings.append("1-0")  # White wins
                elif result_value == -1:
                    result_strings.append("0-1")  # Black wins
                else:
                    result_strings.append("1/2-1/2")  # Draw
                result_idx += 1
            else:
                result_strings.append("*")  # Game ongoing
        
        return result_strings
    
    def print_memory_overview(self, normalize_to_b=100):
        """Print comprehensive memory usage overview"""
        current_b = self.board_tensor.shape[0]
        scale_factor = normalize_to_b / current_b
        
        print(f"TorchBoard Memory Overview (normalized to B={normalize_to_b}):")
        print("=" * 60)
        
        # Core TorchBoard data
        board_mem = current_b * 64 * 1 * scale_factor  # uint8
        side_mem = current_b * 1 * 1 * scale_factor
        plys_mem = current_b * 8 * scale_factor  # int64
        castling_mem = current_b * 4 * 1 * scale_factor  # uint8
        ep_mem = current_b * 1 * 1 * scale_factor
        fifty_mem = current_b * 1 * 1 * scale_factor
        
        core_total = board_mem + side_mem + plys_mem + castling_mem + ep_mem + fifty_mem
        
        print(f"Core Board Data:")
        print(f"  board_tensor:     {board_mem:8.1f} bytes ({board_mem/1024:.1f} KB)")
        print(f"  side_to_move:     {side_mem:8.1f} bytes")
        print(f"  plys:             {plys_mem:8.1f} bytes") 
        print(f"  castling:         {castling_mem:8.1f} bytes")
        print(f"  ep:               {ep_mem:8.1f} bytes")
        print(f"  fifty_move_clock: {fifty_mem:8.1f} bytes")
        print(f"  CORE TOTAL:       {core_total:8.1f} bytes ({core_total/1024:.1f} KB)")
        print()
        
        # Position history (if present)
        if hasattr(self, 'position_hashes') and self.position_hashes is not None:
            avg_positions = 5  # Your realistic estimate
            hash_mem = current_b * avg_positions * 4 * scale_factor  # int32
            count_mem = current_b * avg_positions * 1 * scale_factor  # uint8
            history_total = hash_mem + count_mem
            
            print(f"Position History (~{avg_positions} pos/game avg):")
            print(f"  position_hashes:  {hash_mem:8.1f} bytes ({hash_mem/1024:.1f} KB)")
            print(f"  position_counts:  {count_mem:8.1f} bytes")
            print(f"  HISTORY TOTAL:    {history_total:8.1f} bytes ({history_total/1024:.1f} KB)")
            print(f"  History overhead: {(history_total/core_total)*100:.1f}% of core data")
            print()
        
        # Summary
        total_mem = core_total
        if hasattr(self, 'position_hashes') and self.position_hashes is not None:
            total_mem += history_total
        
        print(f"GRAND TOTAL:        {total_mem:8.1f} bytes ({total_mem/1024:.1f} KB)")