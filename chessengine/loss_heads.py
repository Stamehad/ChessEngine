import torch.nn as nn
import torch.nn.functional as F
from chessengine.evalhead import EvalHead
from chessengine.incheckhead import InCheckHead
from chessengine.threathead import ThreatHead

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
        preds = self.head(x)  # (B, 1)
        loss = self.loss_fn(preds, eval)
        return loss, preds
    
class InCheckLoss(nn.Module):
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

        logits = self.head(king_repr)  # (B, 1)
        loss = self.loss_fn(logits, check)

        return loss, logits
    
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
    
class MoveLoss(nn.Module):
    """
    Computes cross-entropy loss for changed squares after a move.
    Expects predictions from ChessEngine and sparse labels with -100 to ignore.
    """
    def __init__(self, config):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, move_pred, target, move_weight=None):
        """
        move_pred: Tensor (B, 64, 7) — logits over move classes per square
        target: LongTensor (B, 64) — each square has label in [0, 6] or -100 to ignore
        move_weight: FloatTensor (B, 1) — optional, multiplies move loss per sample
        """
        B, S, C = move_pred.shape  # B=batch, S=64 squares, C=7 classes
        loss = F.cross_entropy(move_pred.view(B * S, C), target.view(B * S), reduction='none') # (B*S)
        if move_weight is not None:
            loss = loss.view(B, S) * move_weight
            
        return loss.mean() 