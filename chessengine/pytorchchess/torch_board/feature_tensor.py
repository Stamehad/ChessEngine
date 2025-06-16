import torch
import torch.nn.functional as F
from pytorchchess.utils import int_to_squares
import pytorchchess.utils.constants as c
from pytorchchess.utils.constants import PAWN_CAP_W, PAWN_CAP_B
from pytorchchess.state.premoves import PreMoves

class FeatureTensor:
    def feature_tensor(self):
        """
        Returns (B, 21, 8, 8) uint8.
        """
        B = self.board_tensor.size(0)
        if B == 0:
            return torch.zeros(0, 21, 8, 8, device=self.device, dtype=torch.uint8)
        x = torch.zeros(B, 21, 8, 8, device=self.device, dtype=torch.uint8)

        # planes 1-12
        x0 = F.one_hot(self.board_tensor.long(), num_classes=13).to(torch.uint8) # (B, 8, 8, 13)
        #x[:, :13] = x0.permute(0, 3, 1, 2)  # (B, 13, 8, 8)

        # side-to-move channel 13
        stm = self.side_to_move.view(B, 1, 1).to(torch.uint8)
        x[:, 13].fill_(0)
        x[:, 13] += stm

        # in-check 14
        chk = self.in_check.view(B, 1, 1)
        x[:, 14].fill_(0)
        x[:, 14] += chk

        # threat planes: need attack maps
        attack_map = self.cache.attack_map
        friendly_att = (attack_map == 1) | (attack_map == 3)  # (B, 64)
        enemy_att = (attack_map > 1)                          # (B, 64)
        
        friendly_att &= (self.board_flat != 0)  # (B, 64)
        enemy_att &= (self.board_flat != 0)    # (B, 64)

        friendly_att &= ~self.friendly_occ()  # (B, 64)
        enemy_att &= ~self.enemy_occ()        # (B, 64)
        
        threatened = (friendly_att | enemy_att).view(-1, 8, 8)
        x[:, 15] = threatened.to(torch.uint8)

        friendly, enemy = self.premoves_both_sides()
        friendly = self.filter_pawn_push(friendly, ep=False)
        enemy = self.filter_pawn_push(enemy, ep=True)

        att = (friendly.moves.bool() & self.enemy_occ()[friendly.board]).any(dim=-1)   # (N_f,) 
        fr_sq = friendly.sq[att]                                                       # (N_a,)
        fr_board = friendly.board[att]                                                 # (N_a,)
 
        att = (enemy.moves.bool() & self.friendly_occ()[enemy.board]).any(dim=-1)      # (N_e,)
        en_sq = enemy.sq[att]                                                          # (N_a,)
        en_board = enemy.board[att]                                                    # (N_a,)

        threatening = torch.zeros_like(self.board_flat)  # (B, 64)
        threatening[fr_board, fr_sq] = 1
        threatening[en_board, en_sq] = 1

        x[:, 16] = threatening.view(B,8,8).to(torch.uint8)


        # legal-move plane 17
        legal = self.get_legal_moves()            # LegalMoves
        from_sq = legal.encoded.long() % 64              # (B, L_max)
        mask = legal.mask.unsqueeze(-1)           # (B, L_max, 1)
              
        has_moves = F.one_hot(from_sq, num_classes=64).to(torch.uint8)  # (B, L_max, 64)
        has_moves = (has_moves * mask).sum(dim=1)                       # (B, 64)
        x[:, 17] += has_moves.clamp(max=1).view(-1, 8, 8)               # (B, 64)

        # castling planes 18,19
        x[:, 18].fill_(0)
        x[:, 19].fill_(0)
        w_castling = self.state.castling[:, 0] | self.state.castling[:, 1]
        b_castling = self.state.castling[:, 2] | self.state.castling[:, 3]
        w_castling = w_castling.view(B, 1, 1).to(torch.uint8)
        b_castling = b_castling.view(B, 1, 1).to(torch.uint8)
        x[:, 18] += w_castling  # white either side
        x[:, 19] += b_castling  # black

        # en-passant target 20
        ep_flat = torch.zeros_like(self.board_flat) # (B, 64)
        valid_ep = self.state.ep < 64               # (B,)
        if valid_ep.any():
            sq = self.state.ep[valid_ep]  
            ep_flat[valid_ep, sq.int()] = 1
        x[:, 20] = ep_flat.view(B, 8, 8).to(torch.uint8)

        x = x.permute(0, 2, 3, 1)  # (B, 8, 8, 21)
        x[:, :, :, :13] = x0  # (B, 8, 8, 13)

        return x
    
    def get_legal_move_tensor(self):
        
        lm = self.get_legal_moves()                       # LegalMoves
        lm, lm_mask = lm.encoded, lm.mask                 # (B, L_max), (B, L_max)

        from_sq, to_sq, move_type = int_to_squares(lm)    # (B, L_max)
        piece = self.board_flat.gather(1, from_sq.long())        # (B, L_max)
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
            assert (captured_sq >= 0) & (captured_sq < 64), f"Invalid en passant square {captured_sq}"
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

        return lm_tensor.permute(0, 2, 1).to(torch.int8)  # (B, 64, L_max)
    
    def filter_pawn_push(self, moves: PreMoves, ep: bool=False) -> PreMoves:
        """
        Filter out non-capture pawn moves.
        """
        is_pawn = (moves.id == 1) | (moves.id == 7)  # (N_moves,) Pawn piece IDs
        sq = moves.sq[is_pawn]  # (N_p,)
        capture_mask = (c.PAWN_CAP_W + c.PAWN_CAP_B).clamp(max=1).view(64, 64)  # (64, 64)

        capture_mask = capture_mask[sq]                         # (N_p, 64)
        pre_moves = moves.moves.clone()                         # (N_moves, 64)
        pre_moves[is_pawn] = pre_moves[is_pawn] * capture_mask  # (N_p, 64)

        if ep:
            pre_moves = pre_moves.masked_fill(pre_moves == 5, 0)  # (N_p, 64)

        return PreMoves(
            moves=pre_moves,
            sq=moves.sq.clone(),
            board=moves.board.clone(),
            id=moves.id.clone()
        )
