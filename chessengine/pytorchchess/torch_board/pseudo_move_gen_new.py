import torch
from pytorchchess.utils.constants_new import (
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, KING_TO, MOVES
    )
from pytorchchess.state.premoves import PreMoves, Pieces
import pytorchchess.utils.constants_new as c

class PseudoMoveGeneratorNew:

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
    def get_geometric_moves(self):
        board = self.board_flat  # (B, 64)
        b_idx, sq_idx = board.nonzero(as_tuple=True)  # (N,), (N,)
        geo_moves = MOVES[board[b_idx, sq_idx], sq_idx]  # (N, 64)
        return geo_moves, Pieces(
            sq=sq_idx,    # (N,)
            id=board[b_idx, sq_idx][:, None], # (N, 1)
            board=b_idx   # (N,)
        )

    def get_moves2(self):
        geo, pieces = self.get_geometric_moves() # (N, 64), Pieces (N,)

        side = self.side[pieces.board][:, None]  # (N, 1)
        W = self.occ_white()[pieces.board]  # (N, 64)
        B = self.occ_black()[pieces.board]  # (N, 64)
        empty_sq = ~(W | B)  # (N, 64)

        #-----------------------------------------------------------------
        # Piece masks
        #-----------------------------------------------------------------
        wp = pieces.id == WP  # (N, 1)
        bp = pieces.id == BP  # (N, 1)
        wbp = wp | bp         # (N, 1)
        wn = pieces.id == WN  # (N, 1)
        bn = pieces.id == BN  # (N, 1)
        wk = pieces.id == WK  # (N, 1)
        bk = pieces.id == BK  # (N, 1)

        #-----------------------------------------------------------------
        # Pawn pushes
        #-----------------------------------------------------------------        
        # Valid single pushes
        push1 = (geo == 1) & empty_sq  # (N, 64)
        push2 = (geo == 2) & empty_sq  # (N, 64)

        # For double pushes, check intermediate square
        shifted_push1 = torch.roll(push1, shifts=8, dims=-1) + torch.roll(push1, shifts=-8, dims=-1)
        push2 = push2 & shifted_push1 # (N, 64)

        pawn_pushes = wbp * (push1 + 2 * push2)  # (N, 64)

        #-----------------------------------------------------------------
        # Pawn captures
        #-----------------------------------------------------------------
        ep = self.ep[pieces.board] # (N,)
        ep_valid = (ep < 64).unsqueeze(1) # (N,)
        ep_hits = torch.arange(64, device=ep.device) == ep.unsqueeze(1) # (N, 64)
        ep = ep_valid & ep_hits 
        ep_B = ep * (side == 1) # (N, 64)
        ep_W = ep * (side == 0) # (N, 64)

        captures = (
            ((4 * B + 5 * ep_B) * wp) +  # white pawns capture black pieces or en passant
            ((4 * W + 5 * ep_W) * bp)    # black pawns capture white pieces or en passant
        ) * (geo == 4)  # (N, 64)

        pawn = pawn_pushes + captures  # (N, 64)
        knight = (~B) * wn + (~W) * bn # (N, 64)
        king = (~B) * wk + (~W) * bk   # (N, 64)

        filtered_geo = pawn + geo * (knight + king) # (N, 64)

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
        P       = 32                                       # fixed 32 channels

        # ------------------------------------------------------------------
        # 1)  Compute for every non-zero square:
        #     • board index  b_idx
        #     • square       sq_idx
        #     • channel id   ch_idx   (0..31 per board)
        # ------------------------------------------------------------------
        piece_mask          = board != 0                   # (B, 64) boolean
        # cumulative count per row → gives channel index
        cum_counts          = piece_mask.cumsum(dim=1) - 1 # (B, 64)
        channel_idx         = torch.where(piece_mask, cum_counts, torch.full_like(cum_counts, -1))
        # keep only first 32 pieces – extremely rare to exceed this
        channel_idx         = channel_idx.clamp(min=-1, max=P-1)

        # gather indices of existing pieces
        b_idx, sq_idx       = piece_mask.nonzero(as_tuple=True)      # (N,)

        ch_idx              = channel_idx[b_idx, sq_idx]             # (N,)
        valid_mask          = ch_idx >= 0                            # safety

        b_idx, sq_idx, ch_idx = b_idx[valid_mask], sq_idx[valid_mask], ch_idx[valid_mask]
        piece_id            = board[b_idx, sq_idx]                   # (N,)  1..12

        piece_id = piece_id.to(torch.long)  # ensure long dtype

        # ------------------------------------------------------------------
        # 2)  Build (B,32) tensors of piece-ids and their squares
        # ------------------------------------------------------------------
        spread_ids = torch.zeros((B, P),  dtype=torch.long,  device=device)   # 0 = empty
        spread_sq  = torch.zeros((B, P),  dtype=torch.long,  device=device)   # 0 is dummy

        spread_ids.index_put_((b_idx, ch_idx), piece_id, accumulate=False)
        spread_sq .index_put_((b_idx, ch_idx), sq_idx,   accumulate=False)

        # ------------------------------------------------------------------
        # 3)  Fetch geometric move masks  → geo:(B,32,64)
        # ------------------------------------------------------------------
        geo = torch.zeros((B, P, 64), dtype=torch.uint8, device=device)

        nonzero_mask       = spread_ids != 0
        nb, nc            = nonzero_mask.nonzero(as_tuple=True)
        pcs               = spread_ids[nb, nc] - 1           # range 0..11
        sqs               = spread_sq [nb, nc]

        geo_moves         = c.MOVES[pcs, sqs]                 # (N, 64)
        geo[nb, nc]       = geo_moves

        # ------------------------------------------------------------------
        # 4)  Build PreMoves exactly like original get_moves, but vectorised
        #     over the extra channel dimension.  Workflow:
        #         • derive piece–type masks (pawn/knight/king)
        #         • compute pawn‑push / capture logic per channel
        #         • compute knight/king legality
        #         • zero‑out collisions with friendly pieces
        #         • finally compress the (B,P,64) tensor back to the
        #           flattened (N,64) representation expected by PreMoves.
        # ------------------------------------------------------------------

        # Expand helper masks to (B,P,64)
        occ_W = self.occ_white().unsqueeze(1)                         # (B,1,64)
        occ_B = self.occ_black().unsqueeze(1)
        # occ_W = (board > 0) & (board < 7)
        #empty_sq = ~(occ_W | occ_B)                                   # (B,1,64)

        white = self.side_to_move == 1                                # (B,1)
        side_B = self.side_to_move.unsqueeze(2)                       # (B,1,1)

        ids      = spread_ids.unsqueeze(-1)                           # (B,P,1)
        is_wp    = ids == WP
        is_bp    = ids == BP
        is_wn    = ids == WN
        is_bn    = ids == BN
        is_wk    = ids == WK
        is_bk    = ids == BK
        pawns    = is_wp | is_bp
        knights  = is_wn | is_bn
        kings    = is_wk | is_bk

        short_range_threats = ((geo == 4) & pawns) | (geo.bool() & (knights | kings))  # (B,P,64)
        # ------------------------------------------------------------------
        # Pawn moves
        # ------------------------------------------------------------------
        push1 = (geo == 1) & ((board == 0).unsqueeze(-2)) #empty_sq
        push2 = (geo == 2) & ((board == 0).unsqueeze(-2)) #empty_sq
        shifted_push1 = torch.roll(push1, 8, -1) + torch.roll(push1, -8, -1)
        push2 &= shifted_push1

        pawn_pushes = pawns * (push1 + 5 * push2)                     # (B,P,64)

        # Captures & en‑passant
        ep_file  = self.ep.unsqueeze(1).expand(-1, P)             # (B,P)
        ep_valid = (ep_file < 64).unsqueeze(-1)                       # (B,P,1)
        ep_hits  = torch.arange(64, device=device) == ep_file.unsqueeze(-1)  # (B,P,64)
        ep_mask  = ep_valid & ep_hits

        ep_B = ep_mask & (side_B == 1)
        ep_W = ep_mask & (side_B == 0)

        captures = (
            ((occ_B + 5 * ep_B) * is_wp) +
            ((occ_W + 5 * ep_W) * is_bp)
        ) * (geo == 4)

        # encoding: 
        # 1) single pawn push + standard captures -> 1
        # 2) two pawn pushes + en passant captures -> 5
        # 3) promotions (by capture and push) -> 10
        pawn_moves = pawn_pushes + captures                           # (B,P,64)
        pawn_moves = pawn_moves * (1 + 9 * c.PROMOTION_MASK)  

        # ------------------------------------------------------------------
        # Knight / king moves
        # ------------------------------------------------------------------
        knight_moves = (~occ_W) * is_wn + (~occ_B) * is_bn
        king_moves   = (~occ_W) * is_wk + (~occ_B) * is_bk

        moves = pawn_moves + geo * (knight_moves + king_moves)        # (B,P,64)

        # Zero out channels with no piece
        moves = moves * (spread_ids != 0).unsqueeze(-1)

        # ------------------------------------------------------------------
        # Compress back to (N,64) for PreMoves
        # ------------------------------------------------------------------
        # bp_idx, ch_idx = (spread_ids != 0).nonzero(as_tuple=True)     # (N,)
        # sq_origin      = spread_sq[bp_idx, ch_idx]                    # (N,)
        # ids_origin     = spread_ids[bp_idx, ch_idx]                   # (N,)
        # boards_origin  = bp_idx                                       # (N,)
        # move_masks     = moves[bp_idx, ch_idx]                        # (N,64)

        # ------------------------------------------------------------------
        # Long-range sliders (bishop / rook / queen) – vectorised over (B,P)
        # ------------------------------------------------------------------
        is_bishop = (ids == WB) | (ids == BB)
        is_rook   = (ids == WR) | (ids == BR)
        is_queen  = (ids == WQ) | (ids == BQ)
        is_slider = is_bishop | is_rook | is_queen                     # (B,P,1)

        # Friendly / enemy occupancy per channel
        is_white_piece = (ids >= 1) & (ids <= 6)                   # (B,P,1)
        same_color     = torch.where(is_white_piece, occ_W, occ_B) # (B,P,64)
        opposite_color = torch.where(is_white_piece, occ_B, occ_W) # (B,P,64)
        occ            = same_color | opposite_color               # (B,P,64)

        # For computing pin data
        king_sq = (board == torch.where(white, WK, BK))            # (B,64)
        king_b, king_sq = king_sq.nonzero(as_tuple=True)           # (B,)
        king_rays = c.MOVES[4, king_sq].view(1, -1, 1, 64)           # (1,B,1,64) rays from king square

        # Work only on slider channels
        geo_slider = (geo * is_slider).unsqueeze(0)                # (1,B,P,64)
        blockers   = (geo_slider * occ.to(device))                 # (1,B,P,64)

        # Ray-ID = value % 8
        ray_id = torch.arange(8, device=device).view(8, 1, 1, 1)   # (8,1,1,1)
        antipodals = (ray_id + 4) % 8                              # (8,1,1,1)

        is_bishop_ray = is_bishop.unsqueeze(0) & (ray_id % 2 == 1)
        is_rook_ray   = is_rook.unsqueeze(0) & (ray_id % 2 == 0)
        is_queen_ray  = is_queen.unsqueeze(0)
        ray_mask  = is_bishop_ray | is_rook_ray | is_queen_ray     # (8, B, P, 1)
        geo_slider = geo_slider * ray_mask                         # (8, B, P, 64)

        # 128 = ray center. king_slider_rays includes the slider square
        rays      = geo_slider * ((geo_slider % 8 == ray_id) | (geo_slider == 128))      # (8,B,P,64)
        king_rays = king_rays * ((king_rays % 8 == antipodals) & (king_rays != 128))     # (8,B,1,64)

        king_slider_rays = (rays != 0) & (king_rays != 0)           # (8,B,P,64) pin data
        ray_blockers = blockers * (rays != 0)                       # (8,B,P,64)
        ray_blockers = ray_blockers.masked_fill(ray_blockers == 0, 64)

        # Note nearest <= 64, so slider own squares are not included
        nearest = ray_blockers.min(dim=-1, keepdim=True)[0]     # (8,B,P,1)
        slider_threats = (rays <= nearest) & (rays != 0)        # (8,B,P,64)
        slider_threats = slider_threats.any(dim=0)              # (B,P,64) bool

        # Mask out friendly collisions
        moves += (slider_threats & (~same_color))               # (B, P, 64) 

        # ------------------------------------------------------------------
        # Filter king moves
        # ------------------------------------------------------------------
        threats = short_range_threats | slider_threats          # (B,P,64)
        enemy_channels = torch.where(white.view(-1, 1, 1), ids >= 7, (ids < 7) & (ids > 0))  # (B,P,1)
        threats = threats & enemy_channels
        threats_board = (threats).any(dim=1)                    # (B,64)  any piece threatens square

        # find king's channel
        batch_sqs = spread_sq[king_b]                  # (B,P) 
        sq_mask = (batch_sqs == king_sq.unsqueeze(1))  # (B,P)
        king_chan = sq_mask.float().argmax(dim=1)      # (B,)

        moves[king_b, king_chan] *= ~threats_board[king_b]

        # ------------------------------------------------------------------
        # Filter moves when in check (rays are treated separately)
        # ------------------------------------------------------------------
        in_check = threats[king_b, :, king_sq]  # (B, P)  in-check mask
        b_check, piece_chan = (in_check & ~is_slider.squeeze(-1)).nonzero(as_tuple=True)  # (N_check,)
        #b_check, piece_chan = (in_check.squeeze(-1)).nonzero(as_tuple=True)  # (N_check,)

        piece_sq = spread_sq[b_check, piece_chan] # (N_check)
        board_allowed = torch.ones((B,64), dtype=torch.bool, device=device)
        board_allowed[b_check] = False
        board_allowed[b_check, piece_sq] = True

        piece_sq = spread_sq[b_check, piece_chan]  # (N_check,)
        candidate_ep = piece_sq + torch.where(white[b_check, 0], 8, -8)  # (N_check,)

        # A check created by an two pawn push can be removed by en passant
        piece_id = spread_ids[b_check, piece_chan]  # (N_check)
        two_pawn_push_check = (piece_id == torch.where(white[b_check].view(-1), BP, WP))  # (N_check)
        two_pawn_push_check &= (self.ep[b_check] == candidate_ep)

        b_two_push = b_check[two_pawn_push_check]  # (N_two_push,)
        #two_push_chan = piece_chan[two_pawn_push_check]  # (N_two_push,)

        ep_check_removal = (moves[b_two_push, :, self.ep[b_two_push].long()] == 5)  # (N_two_push,P)
        b_ep_check, chan_ep_check = ep_check_removal.nonzero(as_tuple=True)         # (N_ep_check,)
        ep_check_sq = self.ep[b_two_push[b_ep_check]].long()  # (N_ep_check,)
        b_ep_check = b_two_push[b_ep_check]
        board_allowed[b_ep_check, ep_check_sq] = True

        # ------------------------------------------------------------------
        # 4)  Pin data
        # ------------------------------------------------------------------
        # Only one ray to king per slider is possible -> use `any` to accumulate
        ray = king_slider_rays.any(dim=-1).permute(1,2,0) # (B,P,8)  rays to king
        king_slider_rays = king_slider_rays.any(dim=0)  # (B,P,64) pin data

        # mask out channels of side to play, attack by opponent -> check and pin
        king_slider_rays &= enemy_channels  # (B,P,64) pin data

        # king_slider_rays: (8,B,P,64)  True on squares strictly between king & slider
        # same/opposite color are from the channel’s (slider's) perspective
        same_on_ray  = king_slider_rays & same_color   # (B,P,64)
        opp_on_ray   = king_slider_rays & opposite_color      # (B,P,64)

        num_same = same_on_ray.int().sum(-1, keepdim=True)   # (B,P,1) 
        num_opp   = opp_on_ray.int().sum(-1, keepdim=True)   # (B,P,1)

        # ray includes slider. ally/opp are relative to slider.
        is_check_ray = (num_same == 1) & (num_opp == 0)        # (B,P,1)
        is_pin_ray   = (num_same == 1) & (num_opp == 1)        # (B,P,1)

        # -------- filter by pin ---------------------------------------------
        b_pin, slider_chan, sq_pin = (opp_on_ray * is_pin_ray).nonzero(as_tuple=True)  # (N_pin,)
        batch_sqs = spread_sq[b_pin]  # (N_pin,P) 
        sq_mask = (batch_sqs == sq_pin.unsqueeze(1))  # (N_pin,P)
        pinned_chan = sq_mask.float().argmax(dim=1)  # (N_pin,)

        moves[b_pin, pinned_chan] *= king_slider_rays[b_pin, slider_chan]               # (B,P,64)

        # -------- filter by check -------------------------------------------
        is_check_ray = is_check_ray.squeeze(-1)                                 # (B,P) bool
        b_ray_check, slider_chan = is_check_ray.nonzero(as_tuple=True)              # (N_check,)
        b_ray, ray_check = ray[b_ray_check, slider_chan].nonzero(as_tuple=True)     # (N_rays,)
        ray_check[ray_check == 0] = 8                                           # (N_rays,)  set ray=0 -> ray=8

        bs = b_ray_check[b_ray]                        # scalar board index
        ps = king_chan[bs]                          # scalar channel index

        # double ray check can never have same parity -> unique index for dir
        parity = ray_check % 2 
        dir = torch.zeros(B, 2, 1, device=device, dtype=torch.long)
        dir[bs, parity, 0] = ray_check
    
        # moves[b_sel, p_sel] *= (moves[b_sel, p_sel] != ray_check).to(moves.dtype)
        moves[bs, ps] *= ((moves[bs, ps] != dir[bs, 0]) & (moves[bs, ps] != dir[bs, 1]))

        # Board-level allowed squares initialised all-True
        combined = torch.ones_like(board_allowed, dtype=torch.uint8) 
        combined.scatter_reduce_(
                dim        = 0,
                index      = b_ray_check.unsqueeze(1).expand(-1, 64),   # (N_check,64)
                src        = king_slider_rays[b_ray_check, slider_chan].to(torch.uint8),# (N_check,64)
                reduce     = "amin",    # amin on bool == logical AND
                include_self = True
        )
        board_allowed &= combined.bool()
        
        # Only filter friendly channels (side to play) excluding king
        b_check = in_check.nonzero(as_tuple=True)[0]
        friend_channels = torch.where(white.view(-1, 1, 1), (ids < 6) & (ids > 0), (ids >= 7) & (ids < 12))  # (B,P,1)
        friend_channels = friend_channels.squeeze(-1)  # (B,P) bool
        b_chan, chan = friend_channels[b_check].nonzero(as_tuple=True)  # (N_friend,)
        moves[b_check[b_chan], chan] *= board_allowed[b_check[b_chan]]  # (B,P,64)    

        # Reintroduce en passant move to remove check
        moves[b_ep_check, chan_ep_check, ep_check_sq] = 5  # (B,P,64) zero out ep squares

        # set non-castling king moves to 1
        moves[king_b, king_chan] = moves[king_b, king_chan].clamp(max=1)  

        # ------------------------------------------------------------------
        # Check castling moves
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

        castling = torch.where(self.side_to_move == 1, castling[:, :2], castling[:, 2:]) # (B,2) 
        king_to = torch.where(self.side_to_move == 1, c.KING_TO[:2], c.KING_TO[2:])          # (B,2)

        b_casting, castling_side = castling.nonzero(as_tuple=True)  # (N_castling,)
        king_chan = king_chan[b_casting]  # (N_castling,)
        king_to = king_to[b_casting, castling_side]  # (N_castling,)

        # Set castling moves: 6 = king side, 7 = queen side (both colors)
        moves[b_casting, king_chan, king_to] = 6 + castling_side  # (B,P,64)

        ids = ids.squeeze(-1)  # (B,P)  remove last dimension
        channels_to_move = torch.where(white.view(-1, 1) == 1, (ids < 7) & (ids > 0), ids >= 7) # (B,P)
        any_moves = (moves != 0).any(dim=-1)  # (B,P)  any move in channel
        b, p = (channels_to_move & any_moves).nonzero(as_tuple=True)  # (N_moves,)
        premoves = PreMoves(
            moves=moves[b, p],      # (N_moves, 64)
            sq=spread_sq[b, p],     # (N_moves,)
            id=ids[b, p],           # (N_moves,)
            board=b                 # (N_moves,)
        )

        return premoves, in_check  # PreMoves, (B,P)
    
    def get_moves_fused(self):
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
        P       = 32                                       # fixed 32 channels

        # ------------------------------------------------------------------
        # 1)  Compute for every non-zero square:
        #     • board index  b_idx
        #     • square       sq_idx
        #     • channel id   ch_idx   (0..31 per board)
        # ------------------------------------------------------------------
        piece_mask = board != 0
        ch = piece_mask.cumsum(-1) * piece_mask - 1         # (B, 64)
        ch = ch.flatten()                                   # (B * 64)

        b64 = torch.arange(B)
        b64 = b64.unsqueeze(-1).expand((B, 64)).flatten()

        sq = torch.arange(64)
        sq = sq.unsqueeze(0).expand((B, 64)).flatten()

        spread_ids = torch.zeros(B, P+1, dtype=torch.uint8)
        spread_ids[b64, ch] = board[b64, sq]

        spread_sq = torch.zeros(B, P+1, dtype=torch.uint8)         
        spread_sq[b64, ch] = sq.to(torch.uint8)

        spread_ids = spread_ids[:, :P]                      # (B, P)
        spread_sq  = spread_sq [:, :P]                      # (B, P)

        b32  = torch.arange(B).unsqueeze(-1).expand(B, P).flatten()
        ch32 = torch.arange(P).unsqueeze(0).expand(B, P).flatten()

        geo = torch.zeros(B, P, 64, dtype=torch.uint8)
        geo[b32, ch32] = c.MOVES[spread_ids.flatten().long()-1, spread_sq.flatten().long()]



        
        # ------------------------------------------------------------------
        # 4)  Build PreMoves exactly like original get_moves, but vectorised
        #     over the extra channel dimension.  Workflow:
        #         • derive piece–type masks (pawn/knight/king)
        #         • compute pawn‑push / capture logic per channel
        #         • compute knight/king legality
        #         • zero‑out collisions with friendly pieces
        #         • finally compress the (B,P,64) tensor back to the
        #           flattened (N,64) representation expected by PreMoves.
        # ------------------------------------------------------------------

        # Expand helper masks to (B,P,64)
        occ_W = self.occ_white().unsqueeze(1)                         # (B,1,64)
        occ_B = self.occ_black().unsqueeze(1)
        # occ_W = (board > 0) & (board < 7)
        #empty_sq = ~(occ_W | occ_B)                                   # (B,1,64)

        white = self.side_to_move == 1                                # (B,1)
        side_B = self.side_to_move.unsqueeze(2)                       # (B,1,1)

        ids      = spread_ids.unsqueeze(-1)                           # (B,P,1)
        is_wp    = ids == WP
        is_bp    = ids == BP
        is_wn    = ids == WN
        is_bn    = ids == BN
        is_wk    = ids == WK
        is_bk    = ids == BK
        pawns    = is_wp | is_bp
        knights  = is_wn | is_bn
        kings    = is_wk | is_bk

        short_range_threats = ((geo == 4) & pawns) | (geo.bool() & (knights | kings))  # (B,P,64)
        # ------------------------------------------------------------------
        # Pawn moves
        # ------------------------------------------------------------------
        push1 = (geo == 1) & ((board == 0).unsqueeze(-2)) #empty_sq
        push2 = (geo == 2) & ((board == 0).unsqueeze(-2)) #empty_sq
        shifted_push1 = torch.roll(push1, 8, -1) + torch.roll(push1, -8, -1)
        push2 &= shifted_push1

        pawn_pushes = pawns * (push1 + 5 * push2)                     # (B,P,64)

        # Captures & en‑passant
        ep_file  = self.ep.unsqueeze(1).expand(-1, P)             # (B,P)
        ep_valid = (ep_file < 64).unsqueeze(-1)                       # (B,P,1)
        ep_hits  = torch.arange(64, device=device) == ep_file.unsqueeze(-1)  # (B,P,64)
        ep_mask  = ep_valid & ep_hits

        ep_B = ep_mask & (side_B == 1)
        ep_W = ep_mask & (side_B == 0)

        captures = (
            ((occ_B + 5 * ep_B) * is_wp) +
            ((occ_W + 5 * ep_W) * is_bp)
        ) * (geo == 4)

        # encoding: 
        # 1) single pawn push + standard captures -> 1
        # 2) two pawn pushes + en passant captures -> 5
        # 3) promotions (by capture and push) -> 10
        pawn_moves = pawn_pushes + captures                           # (B,P,64)
        pawn_moves = pawn_moves * (1 + 9 * c.PROMOTION_MASK)  

        # ------------------------------------------------------------------
        # Knight / king moves
        # ------------------------------------------------------------------
        knight_moves = (~occ_W) * is_wn + (~occ_B) * is_bn
        king_moves   = (~occ_W) * is_wk + (~occ_B) * is_bk

        moves = pawn_moves + geo * (knight_moves + king_moves)        # (B,P,64)

        # Zero out channels with no piece
        moves = moves * (spread_ids != 0).unsqueeze(-1)

        # ------------------------------------------------------------------
        # Compress back to (N,64) for PreMoves
        # ------------------------------------------------------------------
        # bp_idx, ch_idx = (spread_ids != 0).nonzero(as_tuple=True)     # (N,)
        # sq_origin      = spread_sq[bp_idx, ch_idx]                    # (N,)
        # ids_origin     = spread_ids[bp_idx, ch_idx]                   # (N,)
        # boards_origin  = bp_idx                                       # (N,)
        # move_masks     = moves[bp_idx, ch_idx]                        # (N,64)

        # ------------------------------------------------------------------
        # Long-range sliders (bishop / rook / queen) – vectorised over (B,P)
        # ------------------------------------------------------------------
        is_bishop = (ids == WB) | (ids == BB)
        is_rook   = (ids == WR) | (ids == BR)
        is_queen  = (ids == WQ) | (ids == BQ)
        is_slider = is_bishop | is_rook | is_queen                     # (B,P,1)

        # Friendly / enemy occupancy per channel
        is_white_piece = (ids >= 1) & (ids <= 6)                   # (B,P,1)
        same_color     = torch.where(is_white_piece, occ_W, occ_B) # (B,P,64)
        opposite_color = torch.where(is_white_piece, occ_B, occ_W) # (B,P,64)
        occ            = same_color | opposite_color               # (B,P,64)

        # For computing pin data
        king_sq = (board == torch.where(white, WK, BK))            # (B,64)
        king_b, king_sq = king_sq.nonzero(as_tuple=True)           # (B,)
        king_rays = c.MOVES[4, king_sq].view(1, -1, 1, 64)           # (1,B,1,64) rays from king square

        # Work only on slider channels
        geo_slider = (geo * is_slider).unsqueeze(0)                # (1,B,P,64)
        blockers   = (geo_slider * occ.to(device))                 # (1,B,P,64)

        # Ray-ID = value % 8
        ray_id = torch.arange(8, device=device).view(8, 1, 1, 1)   # (8,1,1,1)
        antipodals = (ray_id + 4) % 8                              # (8,1,1,1)

        is_bishop_ray = is_bishop.unsqueeze(0) & (ray_id % 2 == 1)
        is_rook_ray   = is_rook.unsqueeze(0) & (ray_id % 2 == 0)
        is_queen_ray  = is_queen.unsqueeze(0)
        ray_mask  = is_bishop_ray | is_rook_ray | is_queen_ray     # (8, B, P, 1)
        geo_slider = geo_slider * ray_mask                         # (8, B, P, 64)

        # 128 = ray center. king_slider_rays includes the slider square
        rays      = geo_slider * ((geo_slider % 8 == ray_id) | (geo_slider == 128))      # (8,B,P,64)
        king_rays = king_rays * ((king_rays % 8 == antipodals) & (king_rays != 128))     # (8,B,1,64)

        king_slider_rays = (rays != 0) & (king_rays != 0)           # (8,B,P,64) pin data
        ray_blockers = blockers * (rays != 0)                       # (8,B,P,64)
        ray_blockers = ray_blockers.masked_fill(ray_blockers == 0, 64)

        # Note nearest <= 64, so slider own squares are not included
        nearest = ray_blockers.min(dim=-1, keepdim=True)[0]     # (8,B,P,1)
        slider_threats = (rays <= nearest) & (rays != 0)        # (8,B,P,64)
        slider_threats = slider_threats.any(dim=0)              # (B,P,64) bool

        # Mask out friendly collisions
        moves += (slider_threats & (~same_color))               # (B, P, 64) 

        # ------------------------------------------------------------------
        # Filter king moves
        # ------------------------------------------------------------------
        threats = short_range_threats | slider_threats          # (B,P,64)
        enemy_channels = torch.where(white.view(-1, 1, 1), ids >= 7, (ids < 7) & (ids > 0))  # (B,P,1)
        threats = threats & enemy_channels
        threats_board = (threats).any(dim=1)                    # (B,64)  any piece threatens square

        # find king's channel
        batch_sqs = spread_sq[king_b]                  # (B,P) 
        sq_mask = (batch_sqs == king_sq.unsqueeze(1))  # (B,P)
        king_chan = sq_mask.float().argmax(dim=1)      # (B,)

        moves[king_b, king_chan] *= ~threats_board[king_b]

        # ------------------------------------------------------------------
        # Filter moves when in check (rays are treated separately)
        # ------------------------------------------------------------------
        in_check = threats[king_b, :, king_sq]  # (B, P)  in-check mask
        b_check, piece_chan = (in_check & ~is_slider.squeeze(-1)).nonzero(as_tuple=True)  # (N_check,)
        #b_check, piece_chan = (in_check.squeeze(-1)).nonzero(as_tuple=True)  # (N_check,)

        piece_sq = spread_sq[b_check, piece_chan] # (N_check)
        board_allowed = torch.ones((B,64), dtype=torch.bool, device=device)
        board_allowed[b_check] = False
        board_allowed[b_check, piece_sq] = True

        piece_sq = spread_sq[b_check, piece_chan]  # (N_check,)
        candidate_ep = piece_sq + torch.where(white[b_check, 0], 8, -8)  # (N_check,)

        # A check created by an two pawn push can be removed by en passant
        piece_id = spread_ids[b_check, piece_chan]  # (N_check)
        two_pawn_push_check = (piece_id == torch.where(white[b_check].view(-1), BP, WP))  # (N_check)
        two_pawn_push_check &= (self.ep[b_check] == candidate_ep)

        b_two_push = b_check[two_pawn_push_check]  # (N_two_push,)
        #two_push_chan = piece_chan[two_pawn_push_check]  # (N_two_push,)

        ep_check_removal = (moves[b_two_push, :, self.ep[b_two_push].long()] == 5)  # (N_two_push,P)
        b_ep_check, chan_ep_check = ep_check_removal.nonzero(as_tuple=True)         # (N_ep_check,)
        ep_check_sq = self.ep[b_two_push[b_ep_check]].long()  # (N_ep_check,)
        b_ep_check = b_two_push[b_ep_check]
        board_allowed[b_ep_check, ep_check_sq] = True

        # ------------------------------------------------------------------
        # 4)  Pin data
        # ------------------------------------------------------------------
        # Only one ray to king per slider is possible -> use `any` to accumulate
        ray = king_slider_rays.any(dim=-1).permute(1,2,0) # (B,P,8)  rays to king
        king_slider_rays = king_slider_rays.any(dim=0)  # (B,P,64) pin data

        # mask out channels of side to play, attack by opponent -> check and pin
        king_slider_rays &= enemy_channels  # (B,P,64) pin data

        # king_slider_rays: (8,B,P,64)  True on squares strictly between king & slider
        # same/opposite color are from the channel’s (slider's) perspective
        same_on_ray  = king_slider_rays & same_color   # (B,P,64)
        opp_on_ray   = king_slider_rays & opposite_color      # (B,P,64)

        num_same = same_on_ray.int().sum(-1, keepdim=True)   # (B,P,1) 
        num_opp   = opp_on_ray.int().sum(-1, keepdim=True)   # (B,P,1)

        # ray includes slider. ally/opp are relative to slider.
        is_check_ray = (num_same == 1) & (num_opp == 0)        # (B,P,1)
        is_pin_ray   = (num_same == 1) & (num_opp == 1)        # (B,P,1)

        # -------- filter by pin ---------------------------------------------
        b_pin, slider_chan, sq_pin = (opp_on_ray * is_pin_ray).nonzero(as_tuple=True)  # (N_pin,)
        batch_sqs = spread_sq[b_pin]  # (N_pin,P) 
        sq_mask = (batch_sqs == sq_pin.unsqueeze(1))  # (N_pin,P)
        pinned_chan = sq_mask.float().argmax(dim=1)  # (N_pin,)

        moves[b_pin, pinned_chan] *= king_slider_rays[b_pin, slider_chan]               # (B,P,64)

        # -------- filter by check -------------------------------------------
        is_check_ray = is_check_ray.squeeze(-1)                                 # (B,P) bool
        b_ray_check, slider_chan = is_check_ray.nonzero(as_tuple=True)              # (N_check,)
        b_ray, ray_check = ray[b_ray_check, slider_chan].nonzero(as_tuple=True)     # (N_rays,)
        ray_check[ray_check == 0] = 8                                           # (N_rays,)  set ray=0 -> ray=8

        bs = b_ray_check[b_ray]                        # scalar board index
        ps = king_chan[bs]                          # scalar channel index

        # double ray check can never have same parity -> unique index for dir
        parity = ray_check % 2 
        dir = torch.zeros(B, 2, 1, device=device, dtype=torch.long)
        dir[bs, parity, 0] = ray_check
    
        # moves[b_sel, p_sel] *= (moves[b_sel, p_sel] != ray_check).to(moves.dtype)
        moves[bs, ps] *= ((moves[bs, ps] != dir[bs, 0]) & (moves[bs, ps] != dir[bs, 1]))

        # Board-level allowed squares initialised all-True
        combined = torch.ones_like(board_allowed, dtype=torch.uint8) 
        combined.scatter_reduce_(
                dim        = 0,
                index      = b_ray_check.unsqueeze(1).expand(-1, 64),   # (N_check,64)
                src        = king_slider_rays[b_ray_check, slider_chan].to(torch.uint8),# (N_check,64)
                reduce     = "amin",    # amin on bool == logical AND
                include_self = True
        )
        board_allowed &= combined.bool()
        
        # Only filter friendly channels (side to play) excluding king
        b_check = in_check.nonzero(as_tuple=True)[0]
        friend_channels = torch.where(white.view(-1, 1, 1), (ids < 6) & (ids > 0), (ids >= 7) & (ids < 12))  # (B,P,1)
        friend_channels = friend_channels.squeeze(-1)  # (B,P) bool
        b_chan, chan = friend_channels[b_check].nonzero(as_tuple=True)  # (N_friend,)
        moves[b_check[b_chan], chan] *= board_allowed[b_check[b_chan]]  # (B,P,64)    

        # Reintroduce en passant move to remove check
        moves[b_ep_check, chan_ep_check, ep_check_sq] = 5  # (B,P,64) zero out ep squares

        # set non-castling king moves to 1
        moves[king_b, king_chan] = moves[king_b, king_chan].clamp(max=1)  

        # ------------------------------------------------------------------
        # Check castling moves
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

        castling = torch.where(self.side_to_move == 1, castling[:, :2], castling[:, 2:]) # (B,2) 
        king_to = torch.where(self.side_to_move == 1, c.KING_TO[:2], c.KING_TO[2:])          # (B,2)

        b_casting, castling_side = castling.nonzero(as_tuple=True)  # (N_castling,)
        king_chan = king_chan[b_casting]  # (N_castling,)
        king_to = king_to[b_casting, castling_side]  # (N_castling,)

        # Set castling moves: 6 = king side, 7 = queen side (both colors)
        moves[b_casting, king_chan, king_to] = 6 + castling_side  # (B,P,64)

        ids = ids.squeeze(-1)  # (B,P)  remove last dimension
        channels_to_move = torch.where(white.view(-1, 1) == 1, (ids < 7) & (ids > 0), ids >= 7) # (B,P)
        any_moves = (moves != 0).any(dim=-1)  # (B,P)  any move in channel
        b, p = (channels_to_move & any_moves).nonzero(as_tuple=True)  # (N_moves,)
        premoves = PreMoves(
            moves=moves[b, p],      # (N_moves, 64)
            sq=spread_sq[b, p],     # (N_moves,)
            id=ids[b, p],           # (N_moves,)
            board=b                 # (N_moves,)
        )

        return premoves, in_check  # PreMoves, (B,P)