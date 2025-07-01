import torch
from pytorchchess.utils import int_to_squares
from pytorchchess.state.game_state import GameState

class PushMoves:
    """
    Class to handle push moves in a chess game.
    """

    def push(self, moves: torch.Tensor, board_idx: torch.Tensor = None):
        # Phase 1: Always expand
        expanded_board = self._expand_for_moves(moves, board_idx)
        expanded_board._check_move_validity(moves) # Check if moves are valid
        
        # Phase 2: Apply only valid moves
        PAD_MOVE = 2**15
        valid_mask = moves != PAD_MOVE
        expanded_board._apply_moves_masked(moves, valid_mask)
        expanded_board.invalidate_cache()  # Invalidate cache after applying moves
    
        return expanded_board
    
    def _check_move_validity(self, moves: torch.Tensor):
        """Check if moves are in legal moves"""
        lm = self.get_legal_moves() # (LegalMoves)
        all_valid = lm.check_moves(moves)
        if not all_valid:
            raise ValueError("Invalid moves detected in push operation. Moves must be legal.")

    
    def _expand_for_moves(self, moves: torch.Tensor, board_idx: torch.Tensor = None):
        """Phase 1: Create expanded board copies for all moves"""
        device = self.device
        if board_idx is None:
            board_idx = torch.arange(moves.size(0), device=device)
        
        board_flat = self.board_flat[board_idx].clone()
        new_state = self.state[board_idx].clone()  # Ensure proper cloning
        
        return self.__class__(
            board_tensor=board_flat.view(-1, 8, 8),
            state=new_state,
            device=device,
            compute_cache=False  # No need to compute cache here
        )
    
    def _compute_move_context(self, from_sq: torch.Tensor, to_sq: torch.Tensor, 
                              move_type: torch.Tensor, mask: torch.Tensor):
        """Get pieces involved in the moves"""
        idx = torch.where(mask)[0]
        piece_moved = self.board_flat[idx, from_sq[idx]]
        captured_piece = self.board_flat[idx, to_sq[idx]]

        is_capture = (captured_piece != 0)
        is_pawn_move = (piece_moved % 6 == 1)
        resets_fifty_move = is_capture | is_pawn_move  # Reset fifty-move clock if capture or pawn move
        
        return {
            'from': from_sq,                        # (B_new,)
            'to': to_sq,                            # (B_new,)
            'type': move_type,                      # (B_new,) 
            'piece': piece_moved,                   # (B_new,)
            'captured': captured_piece,             # (B_new,)
            'idx': idx,                             # (B_new,)
            'mask': mask,                           # (B_new,)
            'resets_fifty_move': resets_fifty_move, # (B_new,)
        }
    
    def _apply_moves_masked(self, moves: torch.Tensor, valid_mask: torch.Tensor):
        """Phase 2: Apply moves in-place with masking"""
        if not valid_mask.any():
            return  # No valid moves to apply
            
        # Extract move components
        from_sq, to_sq, move_type = int_to_squares(moves) # (B_new,)
        data = self._compute_move_context(from_sq, to_sq, move_type, valid_mask)
        
        # Apply each move type with proper masking
        self._apply_basic_moves(data)
        self._apply_special_moves(data)
        self._update_game_state(data, moves)
    
    def _apply_basic_moves(self, data):
        """Apply basic piece movement (from->to) with masking"""
        idx, mask = data['idx'], data['mask']
        if not mask.any():
            return
        
        from_sq, to_sq, piece_moved = data['from'], data['to'], data['piece']
        self.board_flat[idx, from_sq[idx]] = 0
        self.board_flat[idx, to_sq[idx]] = piece_moved
    
    def _apply_special_moves(self, data):
        """Apply special moves (en passant, castling, promotion) with masking"""
        self._apply_en_passant(data)
        self._apply_promotions(data)
        self._apply_castling(data)
    
    def _apply_en_passant(self, data):
        """Handle en passant moves and double pawn pushes"""
        # Reset en passant for all valid moves (not just type 5)
        mask, from_sq, to_sq, move_type = data['mask'], data['from'], data['to'], data['type']
        
        self.state.reset_en_passant(mask)
        type5_mask = mask & (move_type == 5)
        if not type5_mask.any():
            return
        
        # Handle double pawn pushes
        pawn_push_mask = type5_mask & (torch.abs(from_sq - to_sq) == 16)
        if pawn_push_mask.any():
            valid_boards = torch.where(pawn_push_mask)[0]
            ep_squares = (from_sq[pawn_push_mask] + to_sq[pawn_push_mask]) // 2
            self.state.set_en_passant_squares(valid_boards, ep_squares)
        
        # Handle en passant captures
        ep_capture_mask = type5_mask & ~pawn_push_mask
        if ep_capture_mask.any():
            valid_boards = torch.where(ep_capture_mask)[0]
            ep_tos = to_sq[ep_capture_mask]
            ep_dirs = torch.where(self.side[valid_boards] == 1, -8, 8)
            ep_caps = ep_tos + ep_dirs
            self.board_flat[valid_boards, ep_caps] = 0
    
    def _apply_promotions(self, data):
        """Handle pawn promotions"""
        mask, to_sq, move_type = data['mask'], data['to'], data['type']
        
        promotion_mask = mask & (move_type >= 1) & (move_type <= 4) # (B_new,)
        if not promotion_mask.any():
            return
            
        valid_boards = torch.where(promotion_mask)[0] # (VB,)
        promotion_pieces = torch.where(
            self.side[valid_boards] == 1,
            6 - move_type[promotion_mask],   # White pieces
            12 - move_type[promotion_mask]   # Black pieces
        ) # (VB,)
        self.board_flat[valid_boards, to_sq[promotion_mask]] = promotion_pieces.to(torch.uint8)
    
    def _apply_castling(self, data):
        """Handle castling moves"""
        mask, from_sq, to_sq, move_type = data['mask'], data['from'], data['to'], data['type']
        
        castling_mask = mask & ((move_type == 6) | (move_type == 7))
        if not castling_mask.any():
            return
            
        valid_boards = torch.where(castling_mask)[0]
        side = self.side[valid_boards]
        sign = torch.sign(to_sq[castling_mask] - from_sq[castling_mask])
        shift = torch.where(sign == 1, 3, 4).to(torch.uint8)
        rook = torch.where(side == 1, 4, 10).to(torch.uint8)
        
        rook_from = from_sq[castling_mask] + sign * shift
        rook_to = to_sq[castling_mask] - sign
        
        self.board_flat[valid_boards, rook_from] = 0
        self.board_flat[valid_boards, rook_to] = rook
    
    def _update_game_state(self, data, moves):
        """Update game state (castling rights, fifty-move rule, etc.)"""
        mask, piece_moved, from_sq, to_sq, resets_fifty_move = (
            data['mask'], data['piece'], data['from'], data['to'], data['resets_fifty_move']
        )
        
        if not mask.any():
            return
            
        self.state.update_previous_moves(moves, mask)
        self._update_castling_rights(piece_moved, from_sq, to_sq, mask)
        self.state.update_after_move(resets_fifty_move, mask)

    def _update_castling_rights(self, piece_moved: torch.Tensor, 
                               from_sq: torch.Tensor, to_sq: torch.Tensor, mask: torch.Tensor):
        """Update castling rights based on moves"""
        if not mask.any(): # or not self.state.castling.any():
            return
        
        # Determine which pieces moved
        white_king_move = (piece_moved == 6)
        black_king_move = (piece_moved == 12)
        ks_white_rook_move = ((piece_moved == 4) & (from_sq == 7)) | (to_sq == 7)
        qs_white_rook_move = ((piece_moved == 4) & (from_sq == 0)) | (to_sq == 0)
        ks_black_rook_move = ((piece_moved == 10) & (from_sq == 63)) | (to_sq == 63)
        qs_black_rook_move = ((piece_moved == 10) & (from_sq == 56)) | (to_sq == 56)
        
        ks_white = white_king_move | ks_white_rook_move
        qs_white = white_king_move | qs_white_rook_move
        ks_black = black_king_move | ks_black_rook_move
        qs_black = black_king_move | qs_black_rook_move

        #castling_update = torch.stack([ks_white, qs_white, ks_black, qs_black], dim=1) # (N, 4)
        # print(f"self.state.castling = {self.state.castling}, castling_update = {castling_update}")
        #self.state.update_castling_rights(castling_update, mask)
        # print(f"Updated castling rights: {self.state.castling}")
        self.state.castling[mask, 0] &= ~ks_white[mask]
        self.state.castling[mask, 1] &= ~qs_white[mask] 
        self.state.castling[mask, 2] &= ~ks_black[mask]
        self.state.castling[mask, 3] &= ~qs_black[mask]

    def push2(self, moves: torch.Tensor, board_idx: torch.Tensor = None): # -> "TorchBoard":
        device = self.device
        from_sq, to_sq, move_type = int_to_squares(moves)   # (N,)
        if board_idx is None:
            board_idx = torch.arange(moves.size(0), device=device).long()

        # --- expand board & state rows in the exact input order -------------
        board_flat_orig = self.board_flat                    # (B, 64)
        board_flat = board_flat_orig[board_idx].clone()      # (N, 64)

        new_state = self.state[board_idx]
        #new_state.update_previous_moves(moves)
        new_state.previous_moves.long()[board_idx] = moves.clone()  # (N, L_max)
        new_state.previous_moves = new_state.previous_moves.to(torch.uint16)  # (N, L_max)

        N = moves.size(0)  # new batch size
        local_idx = torch.arange(N, device=device)           # 0..N-1

        piece_moved = board_flat[local_idx, from_sq]  # (N,)
        captured_piece = board_flat[local_idx, to_sq]  # (N,)

        is_capture = (captured_piece != 0)  # (N,)
        is_pawn_move = (piece_moved % 6 == 1)  # (N,) 
        resets_fifty_move = is_capture | is_pawn_move  # (N,)

        # Clear from_sq and place piece at to_sq
        board_flat[local_idx, from_sq] = 0
        board_flat[local_idx, to_sq] = piece_moved

        # Handle en passant and double pawn push (move_type == 5)
        type5_mask = (move_type == 5)
        new_state.ep[:] = 64  # reset all en passant squares
        if type5_mask.any():
            pawn_push_mask = type5_mask & (torch.abs(from_sq - to_sq) == 16) 
            if pawn_push_mask.any():
                # Handle double pawn push to set new en passant target
                ep_boards = local_idx[pawn_push_mask]
                ep_squares = (from_sq[pawn_push_mask] + to_sq[pawn_push_mask]) // 2
                new_state.ep[ep_boards] = ep_squares.to(torch.uint8)
                #new_state.set_en_passant_squares(ep_boards, ep_squares)

            ep_capture_mask = type5_mask & ~pawn_push_mask
            if ep_capture_mask.any():
                ep_boards = local_idx[ep_capture_mask]
                ep_tos = to_sq[ep_capture_mask]
                ep_dirs = torch.where(new_state.side[ep_boards] == 1, -8, 8)
                ep_caps = ep_tos + ep_dirs
                board_flat[ep_boards, ep_caps] = 0

        # Handle promotion (move_type == 1, 2, 3, 4 -> Q, R, B, N = 5, 4, 3, 2)
        promotion_mask = (move_type >= 1) & (move_type <= 4)
        if promotion_mask.any():
            promotion_boards = local_idx[promotion_mask]
            promotion_pieces = torch.where(new_state.side[promotion_boards] == 1,
                                           6 - move_type[promotion_mask],
                                           12 - move_type[promotion_mask]
                                           )
            board_flat[promotion_boards, to_sq[promotion_mask]] = promotion_pieces.to(torch.uint8)

        # Handle castling (move_type == 6, 7)
        castling_mask = (move_type == 6) | (move_type == 7) 
        if castling_mask.any():
            castling_boards = local_idx[castling_mask] 
            side = new_state.side[castling_boards]  
            sign = torch.sign(to_sq[castling_mask] - from_sq[castling_mask])  # ks = 1, qs = -1
            shift = torch.where(sign == 1, 3, 4).to(torch.uint8)              # rook relative to king in ks/qs
            rook = torch.where(side == 1, 4, 10).to(torch.uint8)     # white = 4, black = 10

            rook_from = from_sq[castling_mask] + sign * shift  
            rook_to = to_sq[castling_mask] - sign

            # Shift rook to castling square
            board_flat[castling_boards, rook_from] = 0
            board_flat[castling_boards, rook_to] = rook

        # Update castling rights with king or rook moves
        if new_state.castling.any():
            #-------king moves----------------------
            white_king_move = (piece_moved == 6) 
            black_king_move = (piece_moved == 12)
            #-------rook moves or captured----------
            ks_white_rook_move = ((piece_moved == 4) & (from_sq == 7)) | (to_sq == 7)
            qs_white_rook_move = ((piece_moved == 4) & (from_sq == 0)) | (to_sq == 0)
            ks_black_rook_move = ((piece_moved == 10) & (from_sq == 63)) | (to_sq == 63)
            qs_black_rook_move = ((piece_moved == 10) & (from_sq == 56)) | (to_sq == 56)
            
            ks_white = white_king_move | ks_white_rook_move
            qs_white = white_king_move | qs_white_rook_move
            ks_black = black_king_move | ks_black_rook_move
            qs_black = black_king_move | qs_black_rook_move

            new_state.castling[local_idx, 0] &= ~ks_white
            new_state.castling[local_idx, 1] &= ~qs_white
            new_state.castling[local_idx, 2] &= ~ks_black
            new_state.castling[local_idx, 3] &= ~qs_black

            #new_state.update_castling_rights(ks_white, qs_white, ks_black, qs_black)

        # new_state.update_after_move(resets_fifty_move)
        # Switch side to move
        new_state.side_to_move = 1 - new_state.side_to_move
        new_state.plys += 1
        new_state.fifty_move_clock[resets_fifty_move] = 0
        new_state.fifty_move_clock[resets_fifty_move] += 1
        if resets_fifty_move.any():
            new_state._clear_position_history_on_irreversible_move(resets_fifty_move)

        new_board_tensor = board_flat.view(-1, 8, 8)
        #new_board = TorchBoard(new_board_tensor, state, device)
        new_board = self.__class__(
            board_tensor=new_board_tensor,
            state=new_state,
            device=device
        )
        #new_board.invalidate_cache()
        return new_board