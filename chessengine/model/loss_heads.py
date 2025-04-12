import torch
import torch.nn as nn
import torch.nn.functional as F
from chessengine.model.evalhead import EvalHead, ProbEvalHead
from chessengine.model.incheckhead import InCheckHead
from chessengine.model.threathead import ThreatHead
from chessengine.model.utils import masked_one_hot

class EvalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = EvalHead(config['model'])
        self.loss_fn = nn.MSELoss()
        # Optional: self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, eval):
        """
        x: transformer output (B, 65, H)
        preds: Tensor of shape (B, 1) with values in [0, 1]
        eval: Tensor of shape (B, 1) with values in {0, 0.5, 1}
        """
        # Prepare evals
        eval = eval.float()/2  # (B, 1)
        preds = self.head(x)  # (B, 1)
        loss = self.loss_fn(preds, eval)
        return loss, preds
    
class ProbEvalLoss(nn.Module):
    """
    Computes the cross-entropy loss for the probabilistic evaluation head.
    I.e. given a position the head predicts the probability of each outcome:
    - White winning
    - Draw
    - Black winning
    When no ground truth move is available, the loss is ignored.
    The predition is obtained from the [EVAL] token.
    """
    def __init__(self, config):
        super().__init__()
        self.head = ProbEvalHead(config['model'])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, eval):
        """
        x: transformer output (B, 65, H)
        eval_logits: Tensor of shape (B, 3) with logits for [white_win, draw, black_win]
        eval: Tensor of shape (B,) with values in {0, 1, 2}
        """
        eval_logits = self.head(x)  # (B, 3)
        loss = self.loss_fn(eval_logits.view(-1, 3), eval.view(-1))
        return loss, eval_logits
    
class InCheckLoss(nn.Module):
    """
    Computes the binary cross-entropy loss for the in-check prediction.
    I.e. given a position the head predicts whether the king will be in check after a move.
    When no ground truth move is available, the loss is ignored.
    The predition is obtained from the king's square.
    """
    def __init__(self, config):
        super().__init__()
        self.head = InCheckHead(config['model'])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, king_square, check):
        """
        x: transformer output (B, 65, H)
        king_square: LongTensor (B, 1), values in [0, 63]  — flattened board square
        check: FloatTensor (B, 1), with values in {0.0, 1.0}
        """
        board_x = x[:, 1:, :]  # Remove the [EVAL] token — shape becomes (B, 64, H)

        index = king_square.unsqueeze(-1).expand(-1, -1, board_x.size(-1))  # (B, 1, H)
        king_repr = board_x.gather(dim=1, index=index)  # (B, 1, H)

        logits = self.head(king_repr).view(-1)  # (B,)
        check = check.view(-1)  # (B,)

        # This removes the -100 entries from the loss
        valid = (check >= 0.0) & (check <= 1.0)  # (B,)
    
        if valid.sum() == 0:
            return torch.tensor(0.0, device=x.device), logits.view(-1, 1)  # safe fallback

        loss = self.loss_fn(logits[valid], check[valid])
        return loss, logits.view(-1, 1)
    
class ThreatLoss(nn.Module):
    """
    Computes the binary cross-entropy loss for newly threatened squares after a move.
    Only applies loss to squares indicated by a mask.
    """
    def __init__(self, config):
        super().__init__()
        self.head = ThreatHead(config['model'])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, target):
        """
        x: transformer output (B, 65, H)
        target: Tensor (B, 64) with 0 or 1 for each square and -100 to ignore
        """

        logits = self.head(x)  # (B, 64, 2)
        loss = self.loss_fn(logits.view(-1, 2), target.view(-1))  # (B*64)
        return loss, logits
        
class LegalMoveLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, move_pred, legal_moves, true_index, move_weight=None):
        """
        move_pred: (B, 64, 7)
        legal_moves: (B, 64, L) — entries in [0–6] or -100
        true_index: (B,) — index of correct move among the L legal moves
        move_weight: (B, 1) or (B,) — optional scalar weight for each sample
        """
        # One-hot encode with zero vector for -100 entries
        legal_moves_one_hot = masked_one_hot(legal_moves, num_classes=7, mask_value=-100)  # (B, 64, L, 7)

        # If for l in [0, L-1] all squares are -100, then l does not represent a legal move and should be ignored
        legal_moves_mask = (legal_moves != -100).any(dim=1)  # (B, L)

        # Expand prediction to align
        move_pred = move_pred.unsqueeze(2)  # (B, 64, 1, 7)

        # Multiply and sum over piece type dimension to select correct logit
        move_logits = (move_pred * legal_moves_one_hot).sum(dim=-1)  # (B, 64, L)

        # Average over changed squares
        mask = (legal_moves != -100)  # (B, 64, L)
        move_logits = move_logits.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, L)

        # (B, L) — set logits of illegal moves to -inf
        masked_logits = move_logits.masked_fill(~legal_moves_mask, float('-inf'))

        # Cross-entropy loss: ignore masked-out (illegal) moves
        ce_loss = F.cross_entropy(masked_logits, true_index, reduction='none', ignore_index=-1)  # (B,)
        
        # Apply weights if provided
        if move_weight is not None:
            move_weight = move_weight.view(-1)  # ensure shape (B,)
            ce_loss = ce_loss * move_weight

        loss = ce_loss.mean()
        return loss, masked_logits