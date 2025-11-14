import torch
import torch.nn.functional as F
from typing import NamedTuple
from dataclasses import dataclass
from typing import Optional
from pytorchchess.utils.utils import get_check_blockers, squares_to_int, int_to_squares, move_dtype
from pytorchchess.utils.constants import PROMOTION_MASK  
from .premoves import PreMoves
from model.utils import masked_one_hot

@dataclass
class LegalMoves:
    """Encodes and decodes chess moves to a 16-bit integer format.

    Layout:
    bits 0-5   (6 bits): from_sq (0-63)
    bits 6-11  (6 bits): to_sq (0-63)
    bits 12-14 (3 bits): move type (0-7)
    bits 15    (1 bit): padding, if 1 then move is invalid and used for padding

    Move types:
        0 = normal
        1 = promote to Q
        2 = promote to R
        3 = promote to B
        4 = promote to N
        5 = en passant and two squares pawn push
        6 = castle kingside
        7 = castle queenside
    """
    encoded: torch.Tensor  # (B, L_max)
    mask: torch.Tensor     # (B, L_max)
    tensor: Optional[torch.Tensor] = None  # (B, 64, L_max), optional
    one_hot: Optional[torch.Tensor] = None # (B, 64, L_max, 7), optional

    @property
    def shape(self):
        """Returns the shape of the encoded legal moves."""
        return self.encoded.shape

    def select(self, idx):
        return LegalMoves(
            encoded=self.encoded[idx].clone(),
            mask=self.mask[idx].clone(),
            tensor=self.tensor[idx].clone() if self.tensor is not None else None,
            one_hot=self.one_hot[idx].clone() if self.one_hot is not None else None,
        ) 
    
    def clone(self):
        """Returns a copy of the LegalMoves object."""
        return LegalMoves(
            encoded=self.encoded.clone(),
            mask=self.mask.clone(),
            tensor=self.tensor.clone() if self.tensor is not None else None,
            one_hot=self.one_hot.clone() if self.one_hot is not None else None,
        )
    
    @classmethod
    def empty(cls, device, batch_size: int = 0):
        """Returns an empty LegalMoves object."""
        return cls(
            encoded=torch.zeros((batch_size, 0), device=device, dtype=move_dtype(device)),
            mask=torch.zeros((batch_size, 0), device=device, dtype=torch.bool),
            tensor=None,
            one_hot=None
        )
    
    def get_terminal_boards(self):
        """Returns a mask of terminal boards (no legal moves available)."""
        return self.mask.sum(dim=-1) == 0
    
    @classmethod
    def from_premoves(cls, premoves: PreMoves, batch_size: int):
        if premoves.is_empty():
            return cls.empty(premoves.sq.device, batch_size)
        
        move_idx, to_sq = premoves.moves.nonzero(as_tuple=True) # (N_moves,), (N_moves,)
        move_type = premoves.moves[move_idx, to_sq] # (N_moves,)
        from_sq = premoves.sq[move_idx]  # (N_moves,)
        move_board = premoves.board[move_idx] # (N_moves,)
        # moves with 1 are regular -> type 0
        move_type = move_type.masked_fill(move_type == 1, 0) # (N_moves,)
    
        #-------------------------------------
        # promotion moves
        #-------------------------------------
        promo_mask = move_type == 10 # (N_promo,)
        promo_from_sq = from_sq[promo_mask] # (N_promo,)
        promo_to_sq = to_sq[promo_mask] # (N_promo,) 
        promo_moves = torch.cat([
                squares_to_int(promo_from_sq, promo_to_sq, 1),
                squares_to_int(promo_from_sq, promo_to_sq, 2),
                squares_to_int(promo_from_sq, promo_to_sq, 3),
                squares_to_int(promo_from_sq, promo_to_sq, 4)
            ], dim=0) # (N_promo*4,)

        promo_board = move_board[promo_mask] # (N_promo,)
        promo_board = torch.cat([promo_board, promo_board, promo_board, promo_board], dim=0) # (N_promo*4,)
        #--------------------------------------
        # Non-promotion moves
        #--------------------------------------
        regular = ~promo_mask # (N_moves,) 
        to_sq = to_sq[regular] # (N_regular,)
        from_sq = from_sq[regular]  # (N_regular,)
        move_type = move_type[regular] # (N_regular,)

        all_moves = [squares_to_int(from_sq, to_sq, move_type)] # (N_regular,)
        all_boards = [move_board[regular]] # (N_regular,)

        all_moves.append(promo_moves) # (N_regular + N_promo*4,)
        all_boards.append(promo_board) # (N_regular + N_promo*4,)

        all_moves = torch.cat(all_moves, dim=0).long() # (N_moves,)
        all_boards = torch.cat(all_boards, dim=0) # (N_moves,) 
        
        #------------------------------------------------
        # Arrange moves by board (B, L_max) with padding
        #------------------------------------------------
        device  = all_moves.device
        dtype   = move_dtype(device)

        # 1. sort by board ----------------------------------------------------------
        order        = all_boards.argsort(stable=True)
        all_moves    = all_moves[order]
        all_boards   = all_boards[order]

        # 2. counts and maximum length ---------------------------------------------
        counts       = torch.bincount(all_boards, minlength=batch_size)        # (B,)
        L_max        = counts.max().item()

        # 3. prefix-sum gives start-offset of each board’s block -------------------
        starts       = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                counts.cumsum(0)[:-1]])                       # (B,)

        # 4. col = global_pos – starts[board_id] -----------------------------------
        global_pos   = torch.arange(all_moves.size(0), device=device)
        cols         = global_pos - starts[all_boards]                          # (N_moves,)

        # 5. scatter into a padded (B, L_max) tensor -------------------------------
        legal_moves  = torch.full((batch_size, L_max), -1,
                                dtype=dtype, device=device)
        legal_moves[all_boards, cols] = all_moves.to(dtype)

        mask         = legal_moves != -1
        return cls(encoded=legal_moves, mask=mask)

        # counts = torch.bincount(all_boards) 
        # boards = torch.arange(batch_size, dtype=all_boards.dtype, device=all_boards.device)
        # L_max = counts.max()
        # l_idx = torch.arange(1, L_max+1, dtype=boards.dtype, device=boards.device)
        # l_idx = l_idx.view(1, -1, 1)

        # s = (all_boards.view(1, -1) == boards.view(-1, 1))
        # s = s.cumsum(dim=1) * s
        # s = s[:, None, :] == l_idx
        
        # try:
        #     legal_moves = s.long() @ all_moves
        # except:
        #     # s.shape = (B, L_max, N_moves), all_moves.shape = (N_moves,)
        #     legal_moves = s.long() * all_moves
        #     legal_moves = legal_moves.sum(dim=-1)  # (B, L_max)
        # legal_moves = legal_moves.masked_fill(s.sum(dim=-1) == 0, -1)  # padding
        # legal_moves = legal_moves.to(move_dtype(legal_moves.device))

        # mask = (s.sum(dim=-1) > 0)  # (B, L_max)
        
        # return cls(encoded=legal_moves, mask=mask)
    
    def moves_to_standard_format(self):
        # Convert the encoded moves to standard format

        moves = self.encoded.to(torch.int64) # (B, L_max)
        from_sq = moves % 64
        to_sq = (moves // 64) % 64
        move_type = moves // 4096

        # change to standard chess notation
        file_map = 'abcdefgh'
        rank_map = '12345678'
        def idx_to_coord(idx):
            f, r = idx % 8, idx // 8
            return f"{file_map[f]}{rank_map[r]}"

        from_notation = [[idx_to_coord(idx.item()) for idx,m in zip(row, ms) if m != -1] for row, ms in zip(from_sq, moves)]
        to_notation = [[idx_to_coord(idx.item()) for idx,m in zip(row, ms) if m != -1] for row,ms in zip(to_sq, moves)]
        move_type = [[m for m in ms if m != -1] for ms in move_type]

        return from_notation, to_notation, move_type
    
    def is_padded(self):
        # returns a mask of the padded moves (moves with bit 15 set regardless of the rest)
        return self.encoded == -1
    
    def get_tensor(self, board_flat: torch.Tensor):
        """Converts the legal moves to a tensor representation.
        Returns:
            torch.Tensor: A tensor of shape (B, 64, L_max) where each slice along the second dimension
            represents a legal move from a square to another square, with the piece type encoded.
        """
        lm, lm_mask = self.encoded, self.mask                 # (B, L_max), (B, L_max)

        from_sq, to_sq, move_type = int_to_squares(lm)    # (B, L_max)
        piece = board_flat.gather(1, from_sq.long())        # (B, L_max)
        piece += - 6 * (piece > 6)                        # color independent piece type

        # promotions (1,2,3,4) -> (5,4,3,2)
        promotion_mask = (move_type >= 1) & (move_type <= 4)
        if promotion_mask.any():
            piece = piece + (5 - move_type) * promotion_mask

        piece = piece.unsqueeze(-1) # (B, L_max, 1)

        lm_tensor = F.one_hot(from_sq.long(), num_classes=64) * 7     # (B, L_max, 64) 
        lm_tensor += F.one_hot(to_sq.long(), num_classes=64) * piece  # (B, L_max, 64)
        
        # en passant capture 
        ep_mask = (move_type == 5) * ~ (torch.abs(from_sq - to_sq) == 16) # (B, L_max) avoid double pawn push
        if ep_mask.any():
            sign = torch.sign(to_sq[ep_mask] - from_sq[ep_mask]) # white/black captures = 1/-1
            captured_sq = to_sq[ep_mask] - 8 * sign
            assert torch.all((captured_sq >= 0) & (captured_sq < 64)), f"Invalid en passant square {captured_sq}"
            lm_tensor[ep_mask] += F.one_hot(captured_sq.long(), num_classes=64) * 7

        # castling
        castling_mask = (move_type == 6) | (move_type == 7)
        if castling_mask.any():
            sign = torch.sign(to_sq[castling_mask] - from_sq[castling_mask]) # ks = 1, qs = -1
            shift = torch.where(sign == 1, 3, -4)
            rook_from = from_sq[castling_mask] + shift
            rook_to = to_sq[castling_mask] - sign 
            lm_tensor[castling_mask] += F.one_hot(rook_from.long(), num_classes=64) * 7
            lm_tensor[castling_mask] += F.one_hot(rook_to.long(), num_classes=64) * 4

        lm_tensor = lm_tensor * lm_mask.unsqueeze(-1)                 # (B, L_max, 64)
        lm_tensor = lm_tensor.masked_fill(lm_tensor == 0, -100)       # (B, L_max, 64)
        lm_tensor = lm_tensor.masked_fill(lm_tensor == 7, 0)          # (B, L_max, 64)
        lm_tensor = lm_tensor.permute(0, 2, 1)                        # (B, 64, L_max)

        self.tensor = lm_tensor.to(torch.int8)
        self.one_hot = masked_one_hot(lm_tensor.long(), num_classes=7, mask_value=-100).to(torch.int8)  # (B, 64, L_max, 7)

    def get_logits(self, move_pred: torch.Tensor):
        """Converts the legal moves to logits for the model prediction.
        
        Args:
            move_pred: A tensor of shape (B, 64, 7) representing the model's predicted probabilities for each legal move.
        
        Returns:
            move_logits: A tensor of shape (B, L_max) with logits for each legal move.
        """    
        assert move_pred.shape[0] == self.tensor.shape[0], f"move_pred batch size ({move_pred.shape[0]}) must match legal moves batch size ({self.tensor.shape[0]})"

        one_hot = self.one_hot.to(dtype=move_pred.dtype)  # (B, 64, L_max, 7)
        move_pred = move_pred.unsqueeze(2)  # (B, 64, 1, 7)
        move_logits = (move_pred * one_hot).sum(dim=-1)  # (B, 64, L)

        # Average over changed squares
        
        mask = (self.tensor != -100)  # (B, 64, L)
        move_logits = move_logits.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, L)
        move_logits = move_logits.masked_fill(~self.mask, float('-inf'))
        return move_logits
    
    def get_topk(self, logits, ks):
        """
        logits: (B, L_max) Tensor of move logits
        ks: (B,) LongTensor of top-k values for each board

        Returns:
            selected_moves: (N_k,) LongTensor of selected moves
            b_idx:          (N_k,) LongTensor of board indices corresponding to selected moves
            m_idx:          (N_k,) LongTensor of move indices corresponding to selected moves
        """
        assert logits.shape == self.mask.shape, f"Logits ({logits.shape}) and legal moves ({self.mask.shape}) must have the same shape"
        assert ks.shape[0] == self.mask.shape[0], "ks must match the batch size of legal moves"

        moves_per_board = self.mask.sum(dim=-1)
        ks = ks.clamp(max=moves_per_board)  # Ensure k does not exceed available moves

        t = torch.arange(ks.max(), device=self.encoded.device)     # (max_k,)
        t = t.expand(self.encoded.shape[0], -1)                    # (B, max_k)
        t_mask = t < ks.unsqueeze(1)                               # (B, max_k)
        b_idx, m_idx = t_mask.nonzero(as_tuple=True)               # (N_k,), (N_k,)

        logits = logits.masked_fill(~self.mask, float('-inf'))  # Apply mask to logits
        _, topk_indices = torch.topk(logits, ks.max(), dim=-1)  # (B, max_k)
        bs = torch.arange(topk_indices.shape[0], device=topk_indices.device).view(-1, 1)
        if self.encoded.device == torch.device('cpu'):
            top_lm = self.encoded.long()[bs, topk_indices]  # (B, max_k)
        else:
            top_lm = self.encoded[bs, topk_indices]  # (B, max_k)

        #----------------------------------------
        # Flat tensor for selected moves
        #----------------------------------------
        selected_moves = top_lm[b_idx, m_idx]  # (N_k,)

        return selected_moves, b_idx, m_idx # (N_k,), (N_k,), (N_k,)    
    
    def sample_k(self, logits, ks, temp=1.0, generator=None):
        """
        Sample k moves from the logits.
        
        Args:
            move_logits: (B, L_max) Tensor of move logits
            ks: (B,) LongTensor of top-k values for each board
        
        Returns:
            selected_moves: (N_k,) LongTensor of selected moves
            b_idx:          (N_k,) LongTensor of board indices corresponding to selected moves
            m_idx:          (N_k,) LongTensor of move indices corresponding to selected moves
        """
        assert logits.shape == self.mask.shape, f"Logits ({logits.shape}) and legal moves ({self.mask.shape}) must have the same shape"
        assert ks.shape[0] == self.mask.shape[0], "ks must match the batch size of legal moves"
        assert temp > 0, "Temperature tau must be greater than 0"

        moves_per_board = self.mask.sum(dim=-1)
        ks = ks.clamp(max=moves_per_board)

        t = torch.arange(ks.max(), device=self.encoded.device)     # (max_k,)
        t = t.expand(self.encoded.shape[0], -1)                    # (B, max_k)
        t_mask = t < ks.unsqueeze(1)                               # (B, max_k)
        b_idx, m_idx = t_mask.nonzero(as_tuple=True)               # (N_k,), (N_k,)

        logits = logits.masked_fill(~self.mask, float('-inf'))  # Apply mask to logits
        # Sample from the logits
        sampled_indices = torch.multinomial(F.softmax(temp * logits, dim=-1), num_samples=ks.max(), replacement=False, generator=generator)  # (B, max_k)   
        bs = torch.arange(sampled_indices.shape[0], device=sampled_indices.device).view(-1, 1)
        try:
            top_lm = self.encoded[bs, sampled_indices]  # (B, max_k)
        except RuntimeError:
            # If the device is CPU, we need to convert to long
            top_lm = self.encoded.long()[bs, sampled_indices]  # (B, max_k)

        #----------------------------------------
        # Flat tensor for selected moves
        #----------------------------------------
        selected_moves = top_lm[b_idx, m_idx]  # (N_k,) 
        return selected_moves, b_idx, m_idx, ks
    
    def rank_moves(self, move_pred, ks, sample=False, temp=1.0, generator=None):
        move_logits = self.get_logits(move_pred)
        if sample:
            return self.sample_k(move_logits, ks, temp, generator)
        else:
            return self.get_topk(move_logits, ks)
        
    def check_moves(self, moves):
        """
        Check if the given moves are valid legal moves or padding.
        
        Args:
            moves: A tensor of shape (N,) representing the moves to check.
        
        Returns:
            True if the moves are valid legal moves, False otherwise.
        """
        assert moves.dim() == 1, f"Moves must be a 1D tensor, got {moves.dim()}D"
        assert moves.shape[0] == self.encoded.shape[0], f"Moves shape ({moves.shape}) must match legal moves shape ({self.encoded.shape})"
        moves = moves.to(self.encoded.dtype)
        
        # Check if the moves are in the legal moves encoded tensor
        is_valid = self.encoded.eq(moves.view(-1, 1)).any(dim=1)
        # Check if the moves are padding (-1)
        is_padding = moves.eq(-1)

        all_valid = (is_valid | is_padding).all(dim=0)  # Check if all moves are valid or padding
        return all_valid
