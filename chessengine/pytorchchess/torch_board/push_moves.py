import torch
from pytorchchess.utils import int_to_squares
from pytorchchess.state.game_state import GameState

class PushMoves:
    """
    Class to handle push moves in a chess game.
    """

    def push(self, moves: torch.Tensor, board_idx: torch.Tensor = None): # -> "TorchBoard":
        device = self.device
        from_sq, to_sq, move_type = int_to_squares(moves)   # (N,)
        if board_idx is None:
            board_idx = torch.arange(moves.size(0), device=device).long()

        # --- expand board & state rows in the exact input order -------------
        board_flat_orig = self.board_flat                    # (B, 64)
        board_flat = board_flat_orig[board_idx].clone()      # (N, 64)

        state = GameState(
            side_to_move   = self.state.side_to_move[board_idx].clone(),
            plys           = self.state.plys[board_idx].clone(),
            castling       = self.state.castling[board_idx].clone(),
            ep             = self.state.ep[board_idx].clone(),
            previous_moves = moves[board_idx].clone().to(torch.uint16),
            layer          = self.state.layer[board_idx].clone() if self.state.layer is not None else None
        )

        N = moves.size(0)  # new batch size
        local_idx = torch.arange(N, device=device)           # 0..N-1

        piece_moved = board_flat[local_idx, from_sq]  # (N,)

        # Clear from_sq and place piece at to_sq
        board_flat[local_idx, from_sq] = 0
        board_flat[local_idx, to_sq] = piece_moved

        # Handle en passant and double pawn push (move_type == 5)
        type5_mask = (move_type == 5)
        state.ep[local_idx] = 64  # reset to invalid square
        if type5_mask.any():
            pawn_push_mask = type5_mask & (torch.abs(from_sq - to_sq) == 16) 
            if pawn_push_mask.any():
                # Handle double pawn push to set new en passant target
                ep_boards = local_idx[pawn_push_mask]
                ep_squares = (from_sq[pawn_push_mask] + to_sq[pawn_push_mask]) // 2
                state.ep[ep_boards] = ep_squares.to(torch.uint8)

            ep_capture_mask = type5_mask & ~pawn_push_mask
            if ep_capture_mask.any():
                ep_boards = local_idx[ep_capture_mask]
                ep_tos = to_sq[ep_capture_mask]
                ep_dirs = torch.where(state.side_to_move[ep_boards] == 1, -8, 8)
                ep_caps = ep_tos + ep_dirs
                board_flat[ep_boards, ep_caps] = 0

        # Handle promotion (move_type == 1, 2, 3, 4 -> Q, R, B, N = 5, 4, 3, 2)
        promotion_mask = (move_type >= 1) & (move_type <= 4)
        if promotion_mask.any():
            promotion_boards = local_idx[promotion_mask]
            promotion_pieces = torch.where(state.side_to_move[promotion_boards] == 1,
                                           6 - move_type[promotion_mask],
                                           12 - move_type[promotion_mask]
                                           )
            board_flat[promotion_boards, to_sq[promotion_mask]] = promotion_pieces.to(torch.uint8)

        # Handle castling (move_type == 6, 7)
        castling_mask = (move_type == 6) | (move_type == 7) 
        if castling_mask.any():
            castling_boards = local_idx[castling_mask] 
            side = state.side_to_move[castling_boards]  
            sign = torch.sign(to_sq[castling_mask] - from_sq[castling_mask])  # ks = 1, qs = -1
            shift = torch.where(sign == 1, 3, 4).to(torch.uint8)              # rook relative to king in ks/qs
            rook = torch.where(side.view(-1) == 1, 4, 10).to(torch.uint8)     # white = 4, black = 10

            rook_from = from_sq[castling_mask] + sign * shift  
            rook_to = to_sq[castling_mask] - sign

            # Shift rook to castling square
            board_flat[castling_boards, rook_from] = 0
            board_flat[castling_boards, rook_to] = rook

        # Update castling rights with king or rook moves
        if state.castling.any():
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

            state.castling[local_idx, 0] = state.castling[local_idx, 0] & ~ks_white
            state.castling[local_idx, 1] = state.castling[local_idx, 1] & ~qs_white
            state.castling[local_idx, 2] = state.castling[local_idx, 2] & ~ks_black
            state.castling[local_idx, 3] = state.castling[local_idx, 3] & ~qs_black

        # Switch side to move
        state.side_to_move[local_idx] = 1 - state.side_to_move[local_idx]
        state.plys[local_idx] += 1
        if state.layer is not None:
            state.layer[local_idx] += 1

        new_board_tensor = board_flat.view(-1, 8, 8)
        #new_board = TorchBoard(new_board_tensor, state, device)
        new_board = self.__class__(
            board_tensor=new_board_tensor,
            state=state,
            device=device
        )
        #new_board.invalidate_cache()
        return new_board
