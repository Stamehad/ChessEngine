import torch
#from move import MoveEncoder

from pytorchchess.utils.constants import (
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    SHORT_RANGE_MOVES, LONG_RANGE_MOVES, CASTLING_ZONES, PROMOTION_MASK,
    PAWN_CAP_W, PAWN_CAP_B, CASTLING_ATTACK_ZONES
    )
from pytorchchess.state.premoves import PreMoves, Pieces
import pytorchchess.utils.constants as c

# # filter out non capture pawn moves
# PAWNS = c.SHORT_RANGE_MOVES[2:].clone() # (2, 64, 64)
# PAWNS[PAWNS != 4] = 0 # (2, 64, 64)
# SHORT_RANGE = torch.cat([c.SHORT_RANGE_MOVES[:2], PAWNS], dim=0) # (4, 64, 64)

class PseudoMoveGenerator:

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
    
    # ------------------------------------------------------------------
    # Get Pieces
    # ------------------------------------------------------------------

    def get_short_range_pieces(self, enemy=False):
        board = self.board_flat  # (B, 64)
        side = self.side_to_move  # (B, 1)
        if enemy:
            side = 1 - side

        # Identify friendly knight, king, and pawn squares
        knight_mask = torch.where(side == 1, board == WN, board == BN)  # (B, 64)
        king_mask   = torch.where(side == 1, board == WK, board == BK)  # (B, 64)
        pawn_mask   = torch.where(side == 1, board == WP, board == BP)  # (B, 64)
        
        # Combine masks to identify pieces of interest
        piece_mask = knight_mask | king_mask | pawn_mask 
        b_idx, sq_idx = piece_mask.nonzero(as_tuple=True)  # (N,)

        # Determine piece ids: 
        piece_ids = board[b_idx, sq_idx] # (N,)

        return Pieces(
            sq=sq_idx,    # (N,)
            id=piece_ids, # (N,)
            board=b_idx   # (N,)
        )
    
    def get_long_range_pieces(self, enemy=False):
        board = self.board_flat # (B, 64)
        side = self.side_to_move # (B, 1)
        if enemy:
            side = 1 - side

        bishop_mask = torch.where(side == 1, board == WB, board == BB)
        rook_mask   = torch.where(side == 1, board == WR, board == BR)
        queen_mask  = torch.where(side == 1, board == WQ, board == BQ)
        
        # Combine masks to identify pieces of interest
        piece_mask = bishop_mask | rook_mask | queen_mask
        b_idx, sq_idx = piece_mask.nonzero(as_tuple=True)  # (N,)

        # Determine piece ids: 
        piece_ids = board[b_idx, sq_idx] # (N,)

        return Pieces(
            sq=sq_idx,    # (N,)
            id=piece_ids, # (N,)
            board=b_idx   # (N,)
        )
    
    # ------------------------------------------------------------------------------------------
    # Get geometric moves - fetch possible moves of pieces from lookup tables:
    # 1) based on piece type (geometric moves), piece square, and board edge constraints.
    # 2) ignores constraints from other pieces.
    # ------------------------------------------------------------------------------------------

    def get_short_range_geometric_moves(self, pieces: Pieces, side: torch.Tensor, capture_only: bool = False):
        # depending on side choose the correct pawn short range moves
        w_slice = [0, 1, 2]
        b_slice = [0, 1, 3]
        short_ragne_moves = torch.where(side.view(-1, 1, 1, 1) == 1,
                                        c.SHORT_RANGE_MOVES[w_slice],
                                        c.SHORT_RANGE_MOVES[b_slice]
                                        ) # (B, 3, 64, 64)

        # Move type index: 0 = knight, 1 = king, 2 = pawn
        # Determines how to index into SHORT_RANGE_MOVES_LOCAL
        move_type = (
            (pieces.id == WK) + (pieces.id == WP) * 2 + 
            (pieces.id == BK) + (pieces.id == BP) * 2 
        ).to(torch.long) # (N,)

        is_pawn = move_type == 2

        # Lookup move masks in parallel
        moves = short_ragne_moves[pieces.board, move_type, pieces.sq]  # (N, 64), values are bit-encoded: 1=push1, 2=push2, 4=capture

        if capture_only:
            capture_mask = (c.PAWN_CAP_W + c.PAWN_CAP_B).clamp(max=1).view(64, 64)  # (64, 64)
            sq = pieces.sq[is_pawn]  # (N_p,)
            moves[is_pawn] = moves[is_pawn] * capture_mask[sq]  # (N_p, 64)

        return moves, is_pawn # (N, 64), (N,)
    
    def get_long_range_geometric_moves(self, pieces: Pieces):
        # Move type index: 0 = bishop, 1 = rook, 2 = queen
        # Determines how to index into SHORT_RANGE_MOVES_LOCAL
        move_type = (
            (pieces.id == WR) + (pieces.id == WQ) * 2 + 
            (pieces.id == BR) + (pieces.id == BQ) * 2
        ).to(torch.long) # (N,)

        # Lookup move masks in parallel
        moves = c.LONG_RANGE_MOVES[move_type, pieces.sq]  # (N, 64)
        return moves # (N, 64)
    
    # ------------------------------------------------------------------
    # Attack maps (pseudolegal, no blocker cutoff for sliders yet)
    # ------------------------------------------------------------------
    # Important note: enemy attack map ignores friendly king. 
    # Only changes the attack map when the king is attacked by slider pieces.
    def attack_map(self, enemy=True, exclude_defended_pieces=False) -> torch.Tensor:
        """Return (B, 64) mask of attacked squares.
        Args:
            exclude_defended_pieces (bool): if True exclude defended pieces. 
            A piece is defended if it is attacked by a piece of the same color.
            A defended piece cannot be captured by the king. 
        
        """
        side = self.side_to_move  # (B, 1)
        if enemy:
            side = 1 - side           # switch sides
    
        pieces = self.get_short_range_pieces(enemy=enemy) # Pieces (N,)        
        attacks_by_piece, _ = self.get_short_range_geometric_moves(pieces, side, capture_only=True) # (N, 64)

        short_range_attacks = torch.zeros(side.shape[0], 64, device=attacks_by_piece.device, dtype=attacks_by_piece.dtype) # (B, 64)
        short_range_attacks.scatter_add_(0, pieces.board[:, None].expand(-1, 64), attacks_by_piece)
    
        # pieces_to_boards = torch.nn.functional.one_hot(pieces.board, num_classes=side.shape[0]) # (N, B)
        # short_range_attacks = pieces_to_boards.T @ attacks_by_piece.long() # (B, 64)
        
        short_range_attacks = short_range_attacks.bool() # (B, 64)

        # long range attacks
        pieces = self.get_long_range_pieces(enemy=enemy) # Pieces (N,)
        attacks_by_piece = self.get_long_range_geometric_moves(pieces) # (N, 64)

        # exclude friendly king
        # friendly_no_king = self.friendly_occ() & ~torch.where(side == 1, self.board_flat == WK, self.board_flat == BK) # (B, 64)
        # friendly_no_king = friendly_no_king[pieces.board] # (N, 64)
        # enemy = self.enemy_occ()[pieces.board] # (N, 64)
        # occ = enemy + friendly_no_king
        occ = self.enemy_occ()[pieces.board] + self.friendly_occ()[pieces.board] # (N, 64)

        blockers = attacks_by_piece * occ # (N, 64)

        rays = torch.arange(8, device=self.device).view(-1, 1, 1) # (8, 1, 1)
        ray_moves = attacks_by_piece * (attacks_by_piece % 8 == rays) # (8, N, 64)
        ray_blockers = blockers * (ray_moves != 0) # (8, N, 64)
        ray_blockers = ray_blockers.masked_fill(ray_blockers == 0, 64) # (8, N, 64) 

        # Nearest blocker value along this ray (per board)
        nearest_blocker = ray_blockers.min(dim=-1, keepdim=True)[0] # (8, N, 1)
        attacks_by_piece = (ray_moves <= nearest_blocker) & (ray_moves != 0) # (8, N, 64)
        attacks_by_piece = attacks_by_piece.sum(dim=0) # (N, 64)

        long_range_attacks = torch.zeros(side.shape[0], 64, device=side.device, dtype=attacks_by_piece.dtype) # (B, 64)
        long_range_attacks.scatter_add_(0, pieces.board[:, None].expand(-1, 64), attacks_by_piece) # (B, 64)
        
        # pieces_to_boards = torch.nn.functional.one_hot(pieces.board, num_classes=side.shape[0])  # (N, B)
        # long_range_attacks = pieces_to_boards.T @ long_range_attacks.long() # (B, 64)
        
        long_range_attacks = long_range_attacks.bool()

        attacks = short_range_attacks | long_range_attacks # (B, 64)

        if exclude_defended_pieces:
            attacks &= ~self.enemy_occ() # (B, 64)

        return attacks
    
    def combined_attack_map(self) -> torch.Tensor:
        """Return (B, 64) mask of attacked squares.
        Args:
            exclude_defended_pieces (bool): if True exclude defended pieces. 
            A piece is defended if it is attacked by a piece of the same color.
            A defended piece cannot be captured by the king. 
        
        """
        enemy_attack_map = self.attack_map().to(torch.uint8) # (B, 64)
        friendly_attack_map = self.attack_map(enemy=False).to(torch.uint8) # (B, 64)
        
        # 0 = empty, 1 = friendly, 2 = enemy, 3 = both
        combined_attack_map = 2 * enemy_attack_map + friendly_attack_map # (B, 64)
        return combined_attack_map # (B, 64)
    
    def kings_disallowed_squares(self) -> torch.Tensor:
        if self.cache.attack_map is None:
            self.cache.attack_map = self.combined_attack_map() # (B, 64)

        attack_map = self.cache.attack_map.clone() # (B, 64)
        attack_map = (attack_map > 1).to(torch.uint8) # (B, 64)

        if self.in_check.any():
            # If in check by a slider, the square behind the king is also disallowed
            check_data = self.cache.check_info.check_data
            kg_sq = check_data.king_sq # (N_check,)
            rays = check_data.attack_ray # (N_check,)
            b_idx = check_data.board # (N_check,)

            # filter out non-sliders checks
            mask = (rays != -1)  
            rays = rays[mask] # (N_sl,)
            kg_sq = kg_sq[mask] # (N_sl,)
            b_idx = b_idx[mask] # (N_sl,)
            
            antipodals = (rays + 4) % 8                                 # (N_check,)
            antipodals = antipodals.view(-1, 1)                         # (N_check, 1)
            sliders = c.LONG_RANGE_MOVES[2]                               # (64, 64)
            mask = (sliders[kg_sq] % 8  == antipodals) * sliders[kg_sq] # (N_check, 64)
            mask = mask.masked_fill(mask == 0, 64)
            values, sqs = mask.min(dim=-1)                              # (N_check,)
            mask = (values < 64)
            attack_map[b_idx[mask], sqs[mask]] = 1                      # (B, 64)

        return attack_map # (B, 64)
    
    def short_range_moves(self):
        side = self.side_to_move  # (B, 1)
        pieces = self.get_short_range_pieces() # Pieces (N,)
        if pieces.is_empty():
            return PreMoves.empty(side.device)
        
        geo_moves, is_pawn = self.get_short_range_geometric_moves(pieces, side) # (N, 64)

        # Mask out friendly collisions
        friendly = self.friendly_occ()[pieces.board]  # (N, 64)
        enemy = self.enemy_occ()[pieces.board]  # (N, 64)

        if is_pawn.any():
            pawn_indices = torch.nonzero(is_pawn, as_tuple=True)[0] # (Np,)

            # Extract pawn moves
            pawn_moves = geo_moves[pawn_indices]  # (Np, 64)
            
            p_friendly = friendly[pawn_indices]  # (Np, 64)
            p_enemy = enemy[pawn_indices]        # (Np, 64)

            push1_mask = (pawn_moves == 1).to(torch.uint8) # (Np, 64)
            push2_mask = (pawn_moves == 2).to(torch.uint8) # (Np, 64)

            pawn_cap = (pawn_moves * p_enemy == 4).to(torch.uint8) # (Np, 64)
            cap_mask = (pawn_moves == 4) # (Np, 64)

            # En passant logic
            if (self.ep != 64).any():
                ep_target = self.ep[pieces.board][pawn_indices].long()  # (Np,)
                ep = torch.nn.functional.one_hot(ep_target, num_classes=65)  # (Np, 65)
                ep = ep[:, 0:64]  # (Np, 64)
                ep_cap = (pawn_moves * ep == 4) # (Np, 64)
                # Single out en passant with mask = 5 
                pawn_cap += 5 * ep_cap # (Np, 64)

            pawn_push = pawn_moves * ~ cap_mask # (Np, 64)
            occ = (p_enemy + p_friendly).to(torch.uint8) # (Np, 64)
            
            # Sum of weights tells us which pushes are legal:
            # 1 → push1 only, 3 → push1 + push2, 2 → push2 only (illegal)
            # Single out push2 with mask = 5 to later indicate en passant squares
            pawn_push = (pawn_push * (1 - occ)).sum(dim=-1, keepdim=True) # (Np, 1)
            pawn_push = (
                (pawn_push == 1) * push1_mask + 
                (pawn_push == 3) * (push1_mask + 5 * push2_mask) + 
                pawn_cap
            ) # (Np, 64)

            # Single out promotions with mask = 10
            pawn_push = pawn_push * (1 + 9 * c.PROMOTION_MASK)  # (Np, 64)

            # Update move masks
            geo_moves[pawn_indices] = pawn_push
        
        geo_moves *= (~friendly).to(torch.uint8) # (N, 64)

        pre_moves = PreMoves(
            moves=geo_moves,         # (N, 64)
            sq=pieces.sq,            # (N,)
            id=pieces.id,            # (N,)
            board=pieces.board       # (N,)
        )

        # Optionally filter empty moves
        pre_moves.filter_empty()
        return pre_moves
    
    def long_range_moves(self):
        pieces = self.get_long_range_pieces() # Pieces (N,)
        if pieces.is_empty():
            return PreMoves.empty(pieces.sq.device)
        
        geo_moves = self.get_long_range_geometric_moves(pieces) # (N, 64)
        
        # Mask out friendly collisions
        friendly = self.friendly_occ()[pieces.board]  # (N, 64)
        enemy = self.enemy_occ()[pieces.board]  # (N, 64)
        occ = enemy + friendly

        blockers = geo_moves * occ # (N, 64)
        
        # Each move square is encoded with a unique number (1–64+),
        # where ray ID = value % 8 and concentric rings increase by 8
        # move_masks == 0 means not on any ray — filters out non-sliders
        rays = torch.arange(8, device=self.device).view(-1, 1, 1) # (8, 1, 1)
        ray_moves = geo_moves * (geo_moves % 8 == rays) # (8, N, 64)
        ray_blockers = blockers * (ray_moves != 0) # (8, N, 64)
        ray_blockers = ray_blockers.masked_fill(ray_blockers == 0, 64) # (8, N, 64) 

        # Nearest blocker value along this ray (per board)
        nearest_blocker = ray_blockers.min(dim=-1, keepdim=True)[0] # (8, N, 1)
        moves = (ray_moves <= nearest_blocker) & (ray_moves != 0) # (8, N, 64)
        moves = moves.sum(dim=0) # (N, 64)
        
        # Mask out friendly collisions
        moves &= ~friendly

        pre_moves = PreMoves(
            moves=moves,               # (N, 64)
            sq=pieces.sq,              # (N,)
            id=pieces.id,              # (N,)
            board=pieces.board         # (N,)
        )

        # Optionally filter empty moves
        pre_moves.filter_empty()
        return pre_moves
    
    def get_pre_moves(self):
        
        premoves = self.short_range_moves()
        long_range_moves = self.long_range_moves()
        premoves.concat(long_range_moves)

        castling_moves = self.check_castling()
        premoves.concat(castling_moves)

        self.cache.pre_moves = premoves

    def premoves_both_sides(self):
        # Note issue: en passant is incorrectly included in enemy premoves
        friendly = self.cache.pre_moves        # already for side_to_move
        # build enemy premoves on the fly
        #original_side = self.side_to_move.clone()
        self.state.side_to_move ^= 1                 # flip colour
        enemy_pre = self.short_range_moves()
        enemy_pre.concat(self.long_range_moves())
        self.state.side_to_move ^= 1
        #self.side_to_move = original_side      # restore
        return friendly, enemy_pre

    def check_castling(self) -> PreMoves:
        side = self.side_to_move # (B, 1)
        
        castling = torch.where(side == 1, self.state.castling[:, 0:2], self.state.castling[:, 2:4]) # (B, 2)
        
        if not castling.any():
            return PreMoves.empty(side.device)
        castling_zones = torch.where(side.unsqueeze(-1) == 1, c.CASTLING_ZONES[0:2], c.CASTLING_ZONES[2:4]) # (B, 2, 64)
        castling_zones = castling_zones.permute(0,2,1) # (B, 64, 2)
        castling_path = castling_zones.sum(dim=1) # (B, 2) should be 3,4
        
        #filter out kings
        occ = self.occ_all()
        king_mask = torch.where(side == 1, self.board_flat == WK, self.board_flat == BK) # (B, 64)
        occ = occ & ~king_mask
        occ = occ.view(-1, 64, 1)

        self.enemy_attack_mask = self.attack_map() # (B, 64)

        # Castling filters:
        # 1. No pieces in the way (3 squares for ks, 4 for qs)
        # 2. No attacks on the squares the king moves through
        # Note: with qs castling, the knight's square (b1 or b8) must be empty be can be attacked!
        castling_zones = castling_zones & ~occ # (B, 64, 2)
        no_obstructing_piece = castling_zones.sum(dim=1) == castling_path # (B, 2)
        castling_zones = castling_zones & c.CASTLING_ATTACK_ZONES.view(-1, 64, 1) # (B, 64, 2)
        castling_zones = castling_zones & ~self.enemy_attack_mask[..., None] # (B, 64, 2)
        no_zone_attacks = castling_zones.sum(dim=1) == 3 # (B, 2)

        castling_zones = no_zone_attacks & no_obstructing_piece # (B, 2)

        if castling.any():
            kingside = castling[:, 0] & castling_zones[:, 0] # (B,)
            queenside = castling[:, 1] & castling_zones[:, 1] # (B,)
            # Define castling move targets (from square is always king square)
            
            KING_SQ = torch.tensor([4, 60], dtype=torch.long, device=self.device) # (2,)
            KING_SIDE_TO = torch.tensor([6, 62], dtype=torch.long, device=self.device) # (2,)
            QUEEN_SIDE_TO = torch.tensor([2, 58], dtype=torch.long, device=self.device) # (2,)
            
            KING_FROM = torch.where(side == 1, KING_SQ[0], KING_SQ[1]) # (B, 1)
            
            kingside_to = torch.where(side == 1, KING_SIDE_TO[0], KING_SIDE_TO[1])
            queenside_to = torch.where(side == 1, QUEEN_SIDE_TO[0], QUEEN_SIDE_TO[1])

            # Collect valid castling moves
            # ks_idx = torch.nonzero(kingside).squeeze(-1) 
            # qs_idx = torch.nonzero(queenside).squeeze(-1)
            
            ks_idx = torch.nonzero(kingside, as_tuple=True)[0]
            qs_idx = torch.nonzero(queenside, as_tuple=True)[0]

            ks_moves = torch.zeros((len(ks_idx), 64), dtype=torch.long, device=self.device)
            qs_moves = torch.zeros((len(qs_idx), 64), dtype=torch.long, device=self.device)

            # indicate castling move type (6 for kingside, 7 for queenside)
            if len(ks_idx) > 0:
                ks_moves[torch.arange(len(ks_idx)), kingside_to[ks_idx]] = 6
                
            if len(qs_idx) > 0:
                qs_moves[torch.arange(len(qs_idx)), queenside_to[qs_idx]] = 7
                
            kg_side = side[ks_idx].view(-1)
            kg_ids = torch.where(kg_side == 1, WK, BK)
            qg_side = side[qs_idx].view(-1)
            qg_ids = torch.where(qg_side == 1, WK, BK)

            moves = torch.cat([ks_moves, qs_moves], dim=0) # (N, 64)
            sq = torch.cat([KING_FROM[ks_idx, 0], KING_FROM[qs_idx, 0]], dim=0) # (N,)
            board = torch.cat([ks_idx, qs_idx], dim=0) # (N,)
            ids = torch.cat([kg_ids, qg_ids], dim=0) # (N,)

            return PreMoves(moves=moves, sq=sq, id=ids, board=board)

