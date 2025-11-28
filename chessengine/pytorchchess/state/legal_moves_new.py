import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LegalMovesNew:
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
    encoded: torch.Tensor           # (B, L_max)
    sq_changes: torch.Tensor        # (B, L_max, 4) from_sq, to_sq + ep_sq and rook_sq for castling
    label_changes: torch.Tensor     # (B, L_max, 4) one-hot encoding of piece changes
    mask: torch.Tensor              # (B, L_max)

    @property
    def shape(self):
        """Returns the shape of the encoded legal moves."""
        return self.encoded.shape

    def select(self, idx):
        return LegalMovesNew(
            encoded=self.encoded[idx].clone(),
            mask=self.mask[idx].clone(),
            sq_changes=self.sq_changes[idx].clone(),
            label_changes=self.label_changes[idx].clone(),
        ) 
    
    def clone(self):
        """Returns a copy of the LegalMovesNew object."""
        return LegalMovesNew(
            encoded=self.encoded.clone(),
            mask=self.mask.clone(),
            sq_changes=self.sq_changes.clone(),
            label_changes=self.label_changes.clone(),
        )
    
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
    
    def get_logits(self, move_pred: torch.Tensor):
        """Converts the legal moves to logits for the model prediction.
        
        Args:
            move_pred: A tensor of shape (B, 64, 7) representing the model's predicted probabilities for each legal move.
        
        Returns:
            move_logits: A tensor of shape (B, L_max) with logits for each legal move.
        """    
        assert move_pred.shape[0] == self.encoded.shape[0], f"move_pred batch size ({move_pred.shape[0]}) must match legal moves batch size ({self.tensor.shape[0]})"

        B, L, S = self.sq_changes.shape
        b_idx = torch.arange(B, device=move_pred.device).view(-1, 1, 1).expand(-1, L, S)  # (B, L, S)

        sq = self.sq_changes.clamp_min(0).long()                    # (B, L, 4)
        lab = self.label_changes.clamp_min(0)                       # (B, L, 4)

        valid = (self.sq_changes >= 0) & self.mask.unsqueeze(-1)    # (B, L, 4)

        vals = move_pred[b_idx, sq, lab]                            # (B, L, 4)
        vals = vals * valid

        counts = valid.sum(dim=-1).clamp(min=1)                     # (B, L)
        move_logits = vals.sum(dim=-1) / counts                     # (B, L)

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

        device = logits.device
        B, L = logits.shape

        ks = ks.to(device)
        assert ks.shape[0] == B, "ks must have shape (B,)"

        # Number of legal moves per board
        moves_per_board = self.mask.sum(dim=-1)            # (B,)
        ks = torch.minimum(ks, moves_per_board)            # elementwise cap
        max_k = int(ks.max().item())

        if max_k == 0:
            # No moves requested anywhere – return empty tensors
            empty_moves = self.encoded.new_empty(0, dtype=self.encoded.dtype)
            empty_idx = ks.new_empty(0, dtype=torch.long)
            return empty_moves, empty_idx, empty_idx

    
        logits = logits.masked_fill(~self.mask, float('-inf'))  # (B, L)
        _, topk_idx = torch.topk(logits, max_k, dim=-1)         # (B, max_k)

        t = torch.arange(max_k, device=device).expand(B, max_k)   # (B, max_k)
        t_mask = t < ks.unsqueeze(1)                              # (B, max_k)
        b_idx, m_idx = t_mask.nonzero(as_tuple=True)              # (N_k,), (N_k,)

        bs = torch.arange(B, device=device).view(B, 1)            # (B, 1)
        top_lm = self.encoded[bs, topk_idx]                       # (B, max_k)

        selected_moves = top_lm[b_idx, m_idx]                     # (N_k,)

        return selected_moves, b_idx, m_idx
    
    def sample_k(self, logits, ks, temp=1.0, generator=None):
        """
        Sample k moves from the logits.

        Args:
            logits: (B, L_max) Tensor of move logits
            ks: (B,) LongTensor of top-k values for each board

        Returns:
            selected_moves: (N_k,) LongTensor of selected moves
            b_idx:          (N_k,) LongTensor of board indices corresponding to selected moves
            m_idx:          (N_k,) LongTensor of move indices corresponding to selected moves
            ks:             (B,) LongTensor of (possibly clamped) sample sizes per board
        """
        assert logits.shape == self.mask.shape, f"Logits ({logits.shape}) and legal moves ({self.mask.shape}) must have the same shape"
        assert ks.shape[0] == self.mask.shape[0], "ks must match the batch size of legal moves"
        assert temp > 0, "Temperature tau must be greater than 0"

        device = logits.device
        B, L = logits.shape

        ks = ks.to(device)
        assert ks.shape[0] == B, "ks must have shape (B,)"

        moves_per_board = self.mask.sum(dim=-1)                    # (B,)
        ks = torch.minimum(ks, moves_per_board)                    # elementwise cap
        max_k = int(ks.max().item())

        if max_k == 0:
            # No moves requested / available – return empty tensors
            empty_moves = self.encoded.new_empty(0, dtype=self.encoded.dtype)
            empty_idx = ks.new_empty(0, dtype=torch.long)
            return empty_moves, empty_idx, empty_idx, ks

        logits = logits.masked_fill(~self.mask, float('-inf'))     # (B, L)

        # For boards with no legal moves, avoid NaNs in softmax by zeroing the row
        no_moves = moves_per_board == 0
        if no_moves.any():
            logits = logits.clone()
            logits[no_moves] = 0.0

        # Sample indices according to softmax over temp * logits
        probs = F.softmax(temp * logits, dim=-1)                   # (B, L)
        sampled_indices = torch.multinomial(
            probs,
            num_samples=max_k,
            replacement=False,
            generator=generator,
        )                                                           # (B, max_k)

        t = torch.arange(max_k, device=device).expand(B, max_k)    # (B, max_k)
        t_mask = t < ks.unsqueeze(1)                               # (B, max_k)
        b_idx, m_idx = t_mask.nonzero(as_tuple=True)               # (N_k,), (N_k,)

        bs = torch.arange(B, device=device).view(B, 1)             # (B, 1)
        top_lm = self.encoded[bs, sampled_indices]                 # (B, max_k)

        selected_moves = top_lm[b_idx, m_idx]                      # (N_k,)

        return selected_moves, b_idx, m_idx, ks
    
    def rank_moves(self, move_pred, ks, sample=False, temp=1.0, generator=None):
        self.generate_one_hot_()
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