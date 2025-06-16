import torch
from pytorchchess.state.check_info import CheckInfo, PinData, CheckData

from pytorchchess.utils.constants import (
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    SHORT_RANGE_MOVES, QUEEN_MOVES,
    )
import pytorchchess.utils.constants as c

def get_short_range():
    no_king = [0, 2, 3]
    SHORT_RANGE = c.SHORT_RANGE_MOVES[no_king].clone()
    PAWNS = SHORT_RANGE[1:]
    PAWNS[PAWNS != 4] = 0
    return torch.cat([SHORT_RANGE[:1], PAWNS], dim=0)  # shape (3, 64, 64)

class InCheck:

    def _get_short_range_check(self):
        board = self.board_flat # (B, 64)
        side = self.side_to_move # (B, 1)
        king_mask = torch.where(side == 1, board == WK, board == BK)  # (B, 64)

        #------------------------------------------------------------
        # Attacking squares
        #------------------------------------------------------------
        # From king's perspective: 
        # Knight can attack a sq from its possible moves from that sq
        # Pawn can attack a sq from possible same-color(!) capture moves from that sq
        w_slice = [0, 1]
        b_slice = [0, 2]
        SHORT_RANGE = get_short_range() # (3, 64, 64)
        ENEMY_SHORT_RANGE = torch.where(side.view(-1, 1, 1, 1) == 1,
                                        SHORT_RANGE[w_slice],
                                        SHORT_RANGE[b_slice]
                                        ) # (B, 2, 64, 64)
        #ATTACK_MASK = king_mask[:, None, None, :] @ ENEMY_SHORT_RANGE.bool() # (B, 2, 1, 64)
        attacking_sq = (king_mask.view(-1, 1, 64, 1) * ENEMY_SHORT_RANGE).sum(dim=-2) # (B, 2, 64)
        #ATTACK_MASK = ATTACK_MASK.squeeze(2) # (B, 2, 64)

        #------------------------------------------------------------
        # Attacking pieces
        #------------------------------------------------------------
        enemy_knight = torch.where(side == 1, board == BN, board == WN) # (B, 64)
        enemy_pawn = torch.where(side == 1, board == BP, board == WP) # (B, 64)
        enemy = torch.stack([enemy_knight, enemy_pawn], dim=1) # (B, 2, 64)

        #------------------------------------------------------------
        # King in check - enemy on attacking square
        #------------------------------------------------------------
        attacker = attacking_sq * enemy # (B, 2, 64)
        check = attacker.sum(dim=(1,2)) # (B,)
        in_check = check > 0 # (B,)

        if not in_check.any():
            return in_check, CheckData.empty()
        
        #------------------------------------------------------------
        # Get check data - attacker and king square
        #------------------------------------------------------------
        b_idx = in_check.nonzero(as_tuple=True)[0] # (N_check,)
        king_sq = king_mask[b_idx].nonzero(as_tuple=True)[1] # (N_check,)
        attacker = attacker.sum(dim=1) # (B, 64)
        attacker_sq = attacker[b_idx].nonzero(as_tuple=True)[1] # (N_check, 64)

        #------------------------------------------------------------
        # Pawn check by 2-square push (can be removed by ep capture)
        #------------------------------------------------------------
        ep_sq = self.state.ep[b_idx]                # (N_check,)
        two_pawn_push_check = (ep_sq < 64)          # (N_check,)

        check_data = CheckData(
            king_sq=king_sq,                        # (N_check,)
            attacker_sq=attacker_sq,                # (N_check,)
            attack_ray=-torch.ones_like(b_idx),     # (N_check,)
            board=b_idx,                            # (N_check,)
            two_pawn_push_check=two_pawn_push_check

        )

        
        return in_check, check_data # (B,), CheckData
    
    def _get_long_range_check(self):
        board = self.board_flat
        side = self.side_to_move
        king_mask = torch.where(side == 1, board == WK, board == BK).to(torch.uint8) # (B, 64)

        #------------------------------------------------------------
        # Attacking directions (rays)
        #------------------------------------------------------------
        # Computing attack directions from the king's perspective!
        rays = torch.arange(8, device=board.device)[:, None, None] # (8, 1, 1)
        ATTACK_MASK = king_mask @ c.QUEEN_MOVES.view(-1, 64) # (B, 64)
        attacking_rays = ATTACK_MASK * (ATTACK_MASK % 8 == rays) # (8, B, 64)
    
        #------------------------------------------------------------
        # Attacking pieces
        #------------------------------------------------------------
        # diagonal rays -> bishop and queen, straight rays -> rook and queen
        enemy_queen  = torch.where(side == 1, board == BQ, board == WQ) # (B, 64)
        enemy_rook   = torch.where(side == 1, board == BR, board == WR)
        enemy_bishop = torch.where(side == 1, board == BB, board == WB)
             
        enemy_rb = torch.where(rays % 2 == 1, enemy_rook, enemy_bishop) # (8, B, 64)
        attacking_pieces = enemy_queen + enemy_rb # (8, B, 64)
        
        #------------------------------------------------------------
        # Collisions - attack direction meets enemy piece
        #------------------------------------------------------------
        # 64 = ‘no blocker’ ( > max possible distance ).
        collisions = attacking_rays * attacking_pieces # (8, B, 64)
        collisions = collisions.masked_fill(collisions == 0, 64)

        # find the closest collision
        attacker = collisions.min(dim=-1)[0] # (8, B)
        attacker_on_board = (collisions < 64) & (attacker[..., None] == collisions) # (8, B, 64)

        #------------------------------------------------------------
        # Pieces obstructing the attack (friendly and enemy)
        #------------------------------------------------------------
        friendly = self.friendly_occ() # (B, 64)
        friendly_on_rays = attacking_rays * friendly # (8, B, 64)
        friendly_on_rays = friendly_on_rays.masked_fill(friendly_on_rays == 0, 64)
        friendly_obstacles = friendly_on_rays.min(dim=-1)[0] # (8, B)
        
        enemy = self.enemy_occ() # (B, 64)
        enemy_on_rays = attacking_rays * enemy # (8, B, 64)
        enemy_on_rays = enemy_on_rays.masked_fill(enemy_on_rays == 0, 64)
        enemy_obstacles = enemy_on_rays.min(dim=-1)[0] # (8, B)
        
        # keep only the obstacle closest to the king
        obstacle = torch.min(friendly_obstacles, enemy_obstacles) # (8, B)

        #------------------------------------------------------------
        # King in check - attacker is nearest on the ray 
        #------------------------------------------------------------
        # obstacles include enemies so obstacle == attacker -> in check
        in_check_ray = (attacker < 64) & (attacker == obstacle) # (8, B)
        in_check = in_check_ray.sum(dim=0) > 0 # (B,)

        # pieces checking the king
        if in_check.any():
            ray_idx, b_idx = in_check_ray.nonzero(as_tuple=True) # (N_check,), (N_check,)
            king_sq = king_mask[b_idx].nonzero(as_tuple=True)[1] # (N_check,)
            attacker_sq = attacker_on_board[ray_idx, b_idx].nonzero(as_tuple=True)[1] # (N_check,)

            check_data = CheckData(
                king_sq=king_sq,
                attacker_sq=attacker_sq,
                attack_ray=ray_idx,
                board=b_idx,
                two_pawn_push_check=torch.zeros_like(b_idx, dtype=torch.bool)
            )
        
        else:
            check_data = CheckData.empty()

        #------------------------------------------------------------
        # Pinned pieces
        #------------------------------------------------------------
        # Attacker on ray (slider) is not hidden by other enemy pieces
        # A single defender is blocking the attack on the ray
        slider_is_closest_enemy = (attacker == enemy_obstacles) & (attacker < 64) # (8, B)
        defenders = friendly_on_rays < attacker.unsqueeze(-1) # (8, B, 64)
        only_one_defender = (defenders.sum(dim=-1) == 1) # (8, B)
        pins = slider_is_closest_enemy & only_one_defender # (8, B)

        if pins.any():
            ray_idx, pin_b = torch.nonzero(pins, as_tuple=True) # (N_pin,), (N_pin,)
            pin_sq = king_mask[pin_b].nonzero(as_tuple=True)[1] # (N_pin,) 

            defenders = defenders * only_one_defender.unsqueeze(-1) # (8, B, 64)
            pinned_piece_sq = defenders[ray_idx, pin_b] # (N_pin, 64)
            pinned_piece_sq = pinned_piece_sq.nonzero(as_tuple=True)[1] # (N_pin,)

            pin_data = PinData(
                king_sq=pin_sq,
                pinned_piece_sq=pinned_piece_sq,
                ray=ray_idx,
                board=pin_b
            )
        else:
            pin_data = PinData.empty()
        
        return in_check, check_data, pin_data # (B,), CheckData, PinData
    
    def compute_check_info(self):
        """Check if the king is in check"""
        in_check, check_data = self._get_short_range_check()
        in_check_long, check_data_long, pin_data = self._get_long_range_check()
        in_check = in_check | in_check_long
        check_data = check_data.concat(check_data_long)

        check_info = CheckInfo(
            in_check=in_check,
            check_data=check_data,
            pin_data=pin_data
        )
        return check_info






