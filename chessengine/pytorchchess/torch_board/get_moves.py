import torch
import torch.nn.functional as F
from pytorchchess.utils.constants_new import (
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
    )
from pytorchchess.state.legal_moves_new import LegalMovesNew
import pytorchchess.utils.constants_new as c
from pytorchchess.utils.utils import move_dtype
import importlib

importlib.reload(c)

class GetMoves:

    # ------------------------------------------------------------------
    # Occupancy helpers
    # ------------------------------------------------------------------
    def occ_white(self) -> torch.Tensor:
        boards = self.board_flat
        return (boards >=1) & (boards <= 6) # (B, 64)

    def occ_black(self) -> torch.Tensor:
        boards = self.board_flat
        return (boards >=7) & (boards <= 12) # (B, 64)

    def occ_all(self) -> torch.Tensor:
        return self.occ_white() | self.occ_black()
    
    def friendly_occ(self) -> torch.Tensor:
        """Return (B,) bitboards of own pieces."""
        return torch.where(
            self.side_to_move == 1,
            self.occ_white(),
            self.occ_black()
        )

    def enemy_occ(self) -> torch.Tensor:
        """Return (B,) bitboards of opponent pieces."""
        return torch.where(
            self.side_to_move == 1,
            self.occ_black(),
            self.occ_white()
        )
    
    def get_moves(self):
        """
        1.  Take self.board_flat of shape (B,64)           – encoded 0-12
        2.  Spread it to (B,32,64) so each slider/piece
            lives in its own channel              (empty → all zeros)
        3.  Fetch geometric-move masks for every channel
            in parallel, producing geo:(B,32,64)
        """
        
        board   = self.board_flat                          # (B, 64)
        device  = board.device
        B       = board.size(0)

        if self.board_tensor.device.type != self.device.type:
            raise RuntimeError("Board tensor on different device than TorchBoard.device")

        # ------------------------------------------------------------------
        # 0) Geometric moves for all pieces
        # ------------------------------------------------------------------
        sq = torch.arange(64, device=device, dtype=torch.int32).view(1,-1,1) # (1,64,1)
        t  = torch.arange(64, device=device, dtype=torch.int32).view(1,1,-1) # (1,1,64)

        geo = c.MOVES2[board[..., None].int(), sq, t] # (B,64,64)

        # ------------------------------------------------------------------
        # 1) Short range pieces
        # ------------------------------------------------------------------
        board   = board.unsqueeze(dim=-1)           # (B,64,1)
        is_wp   = (board == WP)
        is_bp   = (board == BP)
        is_wn   = (board == WN)
        is_bn   = (board == BN)
        is_wk   = (board == WK)
        is_bk   = (board == BK)
        pawns   = is_wp | is_bp                     # (B,64,1)
        knights = is_wn | is_bn                     # (B,64,1)
        kings   = is_wk | is_bk                     # (B,64,1)

        board = board.permute(0,2,1)        # (B,1,64)
        side = self.side_to_move.unsqueeze(dim=-1)      # (B,1,1)
        occ_W = (board > 0) & (board < 7)   # (B,1,64)
        occ_B = (board > 6)                 # (B,1,64)

        threats = ((geo == 4) & pawns) | (geo.bool() & (knights | kings))  # (B,64,64)
        
        # -------- Short range moves ---------------------------------------
        
        # en‑passant
        ep_mask = (torch.arange(64, device=device) == self.ep[..., None, None])    # (B,1,64)    
        ep_B = ep_mask & (side == 1)
        ep_W = ep_mask & (side == 0)
        
        # pawn moves encoding: 
        # 1) single pawn push + standard captures -> 1
        # 2) two pawn pushes + en passant captures -> 5
        # 3) promotions (by capture and push) -> 10         
        
        moves = (
            # Pawn pushes and captures
            (
                pawns * (
                    ((geo == 1) & (board == 0)).to(torch.uint8) + 
                    5 * (
                        (geo == 2) & (board == 0) &
                        (
                            torch.roll((geo == 1) & (board == 0), 8, -1) |
                            torch.roll((geo == 1) & (board == 0), -8, -1)
                        )
                    ).to(torch.uint8)
                ) + 
                (
                    (occ_B + 5 * ep_B) * is_wp +
                    (occ_W + 5 * ep_W) * is_bp
                ) * (geo == 4)
            ) * (1 + 9 * c.PROMOTION_MASK)
            +
            # Knight and king moves
            (
                ((~occ_W) * (is_wn + is_wk)) +
                ((~occ_B) * (is_bn + is_bk))
            ) * geo
        )

        # ------------------------------------------------------------------
        # 2) Long-range sliders (bishop / rook / queen)
        # ------------------------------------------------------------------
        board = board.permute(0,2,1)                                # (B,64,1)
        is_bishop = (board == WB) | (board == BB)                   # (B,64,1)
        is_rook   = (board == WR) | (board == BR)                   # (B,64,1)
        is_queen  = (board == WQ) | (board == BQ)                   # (B,64,1)
        is_slider = is_bishop | is_rook | is_queen                  # (B,64,1)

        # Ray-ID = value % 8
        ray_id = torch.arange(8, device=device).view(1, 8, 1, 1)   # (1,8,1,1)
        antipodals = (ray_id + 4) % 8                              # (1,8,1,1)

        is_bishop_ray = is_bishop.unsqueeze(1) & (ray_id % 2 == 1)
        is_rook_ray   = is_rook  .unsqueeze(1) & (ray_id % 2 == 0)
        is_queen_ray  = is_queen .unsqueeze(1)
        ray_mask      = is_bishop_ray | is_rook_ray | is_queen_ray  # (B,8,64,1)

        # Friendly / enemy occupancy per square
        same_color     = torch.where(occ_W.permute(0,2,1), occ_W, occ_B)    #. (B,64,64)
        opposite_color = torch.where(occ_W.permute(0,2,1), occ_B, occ_W)    #. (B,64,64)
        
        occ = (occ_W | occ_B).unsqueeze(1)                                  # (B,1,1,64)
        
        # Work only on slider channels
        geo = geo.unsqueeze(1)                                      # (B,1,64,64)
        geo = geo * ray_mask                                        # (B,8,64,64)

        # 128 = ray center. 
        rays      = geo * ((geo % 8 == ray_id) | (geo == 128))      # (B,8,64,64)
        
        ray_blockers = geo * occ * (rays != 0)                       # (B,8,64,64)
        ray_blockers.masked_fill_(ray_blockers == 0, 64)

        # Note nearest <= 64, so slider own squares are not included
        nearest = ray_blockers.min(dim=-1, keepdim=True)[0]     # (B,8,64,1)
        slider_threats = ((rays <= nearest) & (rays != 0)).any(dim=1) # (B,64,64)
        slider_threats &= is_slider

        # Mask out friendly collisions
        moves += (slider_threats & (~same_color))                # (B,64,64) 

        threats |= slider_threats                                #. (B,64,64)
        threatened  = (threats & opposite_color).any(dim=1)      #. (B,64)
        threatening = (threats & opposite_color).any(dim=2)      #. (B,64)

        # From king (of side to move) generate rays from which king can be checked or pinned
        king_mask = (board == torch.where(side == 1, WK, BK))           # (B,64,1)
        king_sq   = torch.argmax(king_mask.long().squeeze(-1), dim=1)   # (B,)
        king_rays = c.MOVES2[5, king_sq].view(-1, 1, 1, 64)             # (B,1,1,64) rays from king square
                                                                        # 5 = queen's channel for all possible rays
        # king_slider_rays includes the slider square
        king_rays = king_rays * ((king_rays % 8 == antipodals) & (king_rays != 128))     # (B,8,1,64)
        king_slider_rays = (rays != 0) & (king_rays != 0)            #. (B,8,64,64)
        
        # ------------------------------------------------------------------
        # 3) Filter king moves
        # ------------------------------------------------------------------
        
        enemy_channels = torch.where(side == 1, occ_B, occ_W)   # (B,1,64)
        enemy_channels = enemy_channels.permute(0,2,1)          # (B,64,1)
    
        threats = threats & enemy_channels
        threats_board = (threats).any(dim=1)                    # (B,64)  any piece threatens square

        b_idx = torch.arange(B, device=device)
        moves[b_idx, king_sq] *= ~threats_board[b_idx]

        # ------------------------------------------------------------------
        # 4) Filter moves when in check (rays are treated separately)
        # ------------------------------------------------------------------
        in_check_from_sq = threats[b_idx, :, king_sq]  # (B, 64)  in-check mask (by sq)
        in_check = (in_check_from_sq & ~is_slider.squeeze(-1)).any(dim=-1, keepdim=True) # (B,1)

        board_allowed = torch.ones((B,64), dtype=torch.bool, device=device)
        non_slider_checks = in_check_from_sq & ~is_slider.squeeze(-1)
        board_allowed = non_slider_checks | ~in_check # (B,64)

        #board_allowed = (in_check_from_sq | ~in_check) # (B,64)

        # ------------------------------------------------------------------
        # 5) Ray (slider) threats for pin and check
        # ------------------------------------------------------------------
        # Only one ray to king per slider is possible -> use `any` to accumulate
        ray = king_slider_rays.any(dim=-1).permute(0,2,1) # (B,64,8)  rays to king
        king_slider_rays = king_slider_rays.any(dim=1)  # (B,64,64) pin data

        # mask out channels of side to play, attack by opponent -> check and pin
        king_slider_rays &= enemy_channels  # (B,64,64) pin data

        # king_slider_rays: (B,64,64)  True on squares strictly between king & slider (including slider's sq)
        # same/opposite color are from the channel’s (slider's) perspective
        same_on_ray  = king_slider_rays & same_color        # (B,64,64)
        opp_on_ray   = king_slider_rays & opposite_color    # (B,64,64)

        num_same = same_on_ray.int().sum(-1, keepdim=True)   # (B,64,1) 
        num_opp   = opp_on_ray.int().sum(-1, keepdim=True)   # (B,64,1)

        # ray includes slider. ally/opp are relative to slider.
        is_check_ray = (num_same == 1) & (num_opp == 0)        # (B,64,1)
        is_pin_ray   = (num_same == 1) & (num_opp == 1)        # (B,64,1)

        in_check |= is_check_ray.any(dim=-2)

        # -------- filter by pin ---------------------------------------------
        # pinned_piece[b,t,s] = True if slider on s pines a piece on t (for batch item b)
        pinned_piece = (opp_on_ray & is_pin_ray).permute(0,2,1)             # (B,64,64) 
        pin = torch.argmax(pinned_piece.int(), dim=-1,keepdim=True)         # (B,64,1)
        pinned_piece = pinned_piece.any(dim=-1, keepdim=True)               # (B,64,1)
        
        moves *= (king_slider_rays[b_idx.view(-1,1,1), pin, t] | (~pinned_piece))

        # -------- filter by check -------------------------------------------
        # slider_check_ray = (king_slider_rays * is_check_ray) | (~is_check_ray)  # (B,64,64)
        # board_allowed &= slider_check_ray.all(dim=1)                            # (B,64)

        board_allowed &= (
            (king_slider_rays * is_check_ray) | (~is_check_ray)
        ).all(dim=1)                                                            # (B,64)

        # when in check by a slider the sq behind the king is disallowed
        ray = ray.long().argmax(dim=-1, keepdim=True)      # (B,64,1)
        ray[ray == 0] = 8                           # set ray=0 -> ray=8

        moves[b_idx, king_sq] *= (
            (~(moves[b_idx, king_sq].unsqueeze(1) == ray) & is_check_ray) | (~is_check_ray)
        ).all(dim=1)

        
        # Only filter friendly channels (side to play) excluding king
        friendly_sq = torch.where(side==1, occ_W, occ_B).permute(0,2,1)     # (B,64,1)
        friendly_sq &= (~king_mask)                                         # (B,64,1)
        
        board_allowed = board_allowed.unsqueeze(1)                          # (B,1,64)
        #board_allowed = (friendly_sq * board_allowed) | (~friendly_sq)      # (B,64,64)

        in_check = in_check.unsqueeze(-1)                               # (B,1,1)
        two_pawn_push_check = in_check & ep_mask                        # (B,1,64)
        ep_check_removal = two_pawn_push_check & (moves == 5)           # (B,64,64)

        moves *= (friendly_sq * board_allowed) | (~friendly_sq)

        # Reintroduce en passant move to remove check
        moves[ep_check_removal] = 5  # (B,P,64) zero out ep squares

        # set non-castling king moves to 1
        moves[b_idx, king_sq] = moves[b_idx, king_sq].clamp(max=1)  

        # ------------------------------------------------------------------
        # 6) Check castling moves
        # ------------------------------------------------------------------
        # CASTLING_ZONES (4, 64) - (K,Q,k,q):
        # squares that must be empty for castling
        # CASTLING_ATTACK_ZONES (4, 64): 
        # squares that cannot be attacked for castling to be legal

        occ_all = self.occ_all().view(-1, 1, 64)                                # (B,1,64)
        threats_board = threats_board.view(-1, 1, 64)                           # (B,1,64)

        castling = ~(c.CASTLING_ZONES & occ_all).any(-1)                        # (B,4)
        castling &= ~(c.CASTLING_ATTACK_ZONES & threats_board).any(-1)          # (B,4)
        castling &= self.state.castling.to(castling.dtype)                      # (B,4)  castling rights

        castling = torch.where(self.side_to_move == 1, castling[:, :2], castling[:, 2:])     # (B,2) 
        king_to = torch.where(self.side_to_move == 1, c.KING_TO[:2], c.KING_TO[2:])          # (B,2)

        b_casting, castling_side = castling.nonzero(as_tuple=True)  # (N_castling,)
        king_sq = king_sq[b_casting]  # (N_castling,)
        king_to = king_to[b_casting, castling_side]  # (N_castling,)

        # Set castling moves: 6 = king side, 7 = queen side (both colors)
        moves[b_casting, king_sq, king_to] = 6 + castling_side  # (B,64,64)

        friendly_sq = torch.where(side==1, occ_W, occ_B).view(-1,64)    # (B,64)
        any_moves = (moves != 0).any(dim=-1)                            # (B,64) any move in sq

        # ------------------------------------------------------------------
        # 7) Legal moves encoding
        # ------------------------------------------------------------------
        legal_moves = moves * friendly_sq.unsqueeze(-1)                 # (B,64,64)
        move_type = legal_moves.masked_fill(legal_moves == 1, 0).to(torch.int16)    
        move_type = move_type.masked_fill_(move_type == 10, 1)
        lm16 = (legal_moves != 0) * c.MOVE_ENCODING + 64**2 * move_type
        lm16 = lm16.view(B,-1) # (B,64*64) 

        promotion_mask = (move_type.view(B,-1) == 1) # (B,64*64)
        qpm = (promotion_mask * lm16).topk(22, dim=1, largest=True)[0]  # queen promotion moves (move type = 1)
                                                                        # 22 is an upper bound on simultaneous 
                                                                        # promotion moves in a single board
        m = (qpm != 0)
        lm16 = torch.cat([lm16, qpm + 64**2 * m, qpm + 2*64**2 * m, qpm + 3*64**2 * m], dim=-1) # (B,64*64+66)

        lm16 = lm16.topk(128, dim=1, largest=True)[0]   # (B,128) 
                                                        # 1) More than 128 legal moves is theoretically possible
                                                        #    but in realistic games this is exceedingly rare.
                                                        # 2) Using topk is an efficient way to filter the moves.
        lm16 = lm16.masked_fill_(lm16 == 0, -1)


        # ------------------------------------------------------------------
        # 8) Legal move helper tensors
        # ------------------------------------------------------------------
        lm16 = lm16.unsqueeze(-1) # (B,L_max,1)
        from_sq = lm16 % 64
        to_sq = (lm16 // 64) % 64
        move_type = lm16 // 4096
        promotion_type = ((move_type > 0) & (move_type < 5)) * (6 - move_type)
        piece = board.gather(1, from_sq.long()).to(torch.int)
        piece += - 6 * (piece > 6)                 # (B,L_max) color independent piece type
        piece += promotion_type

        # Track up to four square changes per move (from/to plus special cases).
        L_max = from_sq.size(1)
        valid_moves = (lm16 != -1).squeeze(-1)                             # (B,L_max)
        from_sq_int16 = from_sq.squeeze(-1).to(torch.int16)                # (B,L_max)
        to_sq_int16 = to_sq.squeeze(-1).to(torch.int16)                    # (B,L_max)
        piece_labels = piece.squeeze(-1).to(torch.int8)                    # (B,L_max)
        zero_labels = torch.zeros_like(piece_labels)                       # (B,L_max)

        sq_changes = torch.full(
            (B, L_max, 4), -1, dtype=torch.int16, device=device
        )                                                                  # (B,L_max,4)
        label_changes = torch.zeros(
            (B, L_max, 4), dtype=torch.int8, device=device
        )                                                                  # (B,L_max,4)

        sq_changes[:, :, 0] = torch.where(valid_moves, from_sq_int16, sq_changes[:, :, 0])
        sq_changes[:, :, 1] = torch.where(valid_moves, to_sq_int16, sq_changes[:, :, 1])
        label_changes[:, :, 1] = torch.where(valid_moves, piece_labels, label_changes[:, :, 1])

        # -------- en passant ----------------------------------------------
        ep_move = (move_type == 5) & (torch.abs(from_sq - to_sq) < 16)              # (B,L_max,1)
        ep_capture_sq = torch.where(
            side == 1, 
            torch.roll(ep_mask, -8, -1), 
            torch.roll(ep_mask, 8, -1)
        )                                                                           # (B,1,64)
        ep_capture_sq = ep_capture_sq & ep_move                                     # (B,L_max,64)

        ep_capture_mask = ep_move.squeeze(-1).bool() & valid_moves
        if ep_capture_mask.any():
            white_to_move = (side == 1).reshape(B, 1).expand(-1, L_max)             # (B,L_max)
            ep_capture_idx = torch.where(
                white_to_move, to_sq_int16 - 8, to_sq_int16 + 8
            ).clamp(0, 63).to(torch.int16)
            sq_changes[:, :, 2] = torch.where(ep_capture_mask, ep_capture_idx, sq_changes[:, :, 2])
            label_changes[:, :, 2] = torch.where(ep_capture_mask, zero_labels, label_changes[:, :, 2])

        # -------- castling ------------------------------------------------
        ks = (move_type == 6)
        qs = (move_type == 7)
        king_mask = king_mask.permute(0,2,1)                                                # (B,1,64)
        rook_from = (ks & torch.roll(king_mask,3,-1)) | (qs & torch.roll(king_mask,-4,-1))  # (B,L_max,64)
        rook_to   = (ks & torch.roll(king_mask,1,-1)) | (qs & torch.roll(king_mask,-1,-1))  # (B,L_max,64)

        castling_mask = (ks | qs).squeeze(-1).bool() & valid_moves
        if castling_mask.any():
            rook_from_idx = rook_from.float().argmax(dim=-1).to(torch.int16)
            rook_to_idx = rook_to.float().argmax(dim=-1).to(torch.int16)
            rook_label = torch.full_like(piece_labels, 4)

            sq_changes[:, :, 2] = torch.where(castling_mask, rook_from_idx, sq_changes[:, :, 2])
            label_changes[:, :, 2] = torch.where(castling_mask, zero_labels, label_changes[:, :, 2])
            sq_changes[:, :, 3] = torch.where(castling_mask, rook_to_idx, sq_changes[:, :, 3])
            label_changes[:, :, 3] = torch.where(castling_mask, rook_label, label_changes[:, :, 3])

        # ------------------------------------------------------------------
        # 9) feature tensor
        # ------------------------------------------------------------------
        x = torch.zeros(B, 64, 21, device=device, dtype=torch.uint8)    # (B,64,21)
        
        x0 = F.one_hot(board.squeeze(-1).long(), num_classes=13).to(torch.uint8)    # (B,64,13)
        x[:,:,:13] = x0

        x[:,:,13] = side.squeeze(-2)
        x[:,:,14] = in_check.squeeze(-1).to(torch.uint8)
        x[:,:,15] = threatened.to(torch.uint8)
        x[:,:,16] = threatening.to(torch.uint8)
        x[:,:,17] = (friendly_sq & any_moves).to(torch.uint8)
        x[:,:,18] = self.state.castling[:, :2].any(dim=-1, keepdim=True)
        x[:,:,19] = self.state.castling[:, 2:].any(dim=-1, keepdim=True)
        x[:,:,20] = ep_mask.squeeze(-2).to(torch.uint8)

        # ------------------------------------------------------------------
        # 10) Package results
        # ------------------------------------------------------------------
        encoded = lm16.squeeze(-1).to(move_dtype(self.device))                # (B, L_max)
        mask = encoded != -1                                                 # (B, L_max)
        sq_changes = sq_changes.to(self.device)
        label_changes = label_changes.to(self.device)
        legal_moves = LegalMovesNew(
            encoded=encoded,
            mask=mask,
            sq_changes=sq_changes,
            label_changes=label_changes,
        )

        feature_tensor = x.view(B, 8, 8, 21).contiguous().to(self.device)

        no_moves = ~mask.any(dim=-1)                                     # (B,)
        if hasattr(self, "cache") and self.cache is not None:
            self.cache.in_check_mask = in_check.view(-1).clone()
            self.cache.no_move_mask = no_moves.clone()
            self.cache.legal_moves = legal_moves.clone()
            self.cache.features = feature_tensor.clone()
            print("Cached features on", self.cache.features.device)

        return legal_moves, feature_tensor
