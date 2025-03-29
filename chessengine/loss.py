import torch
import torch.nn as nn
from chessengine.loss_heads import EvalLoss, InCheckLoss, ThreatLoss, MoveLoss

class Loss(nn.Module):
    """
    Aggregates all individual losses into a single weighted total loss.
    Weights for each loss component are read from the config.
    Optionally weights move loss per-sample based on alignment with game result.
    """
    def __init__(self, config):
        super().__init__()
        self.eval_loss = EvalLoss(config)
        self.incheck_loss = InCheckLoss(config)
        self.threat_loss = ThreatLoss(config)
        self.move_loss = MoveLoss(config)

        # Default weights if not specified in config
        loss_config = config['loss']
        self.weights = {
            'eval': loss_config.get('eval_loss_weight', 1.0),
            'incheck': loss_config.get('incheck_loss_weight', 1.0),
            'threat': loss_config.get('threat_loss_weight', 1.0),
            'move': loss_config.get('move_loss_weight', 3.0),
        }

        self.USE_MOVE_WEIGHT = loss_config.get('use_move_weight', False)

    def forward(self, x, move_pred, labels):
        """
        x: transformer output (B, 65, H)
        move_pred: (B, 64, 7)
        labels: dict containing:
            - 'eval': (B, 1)
            - 'move_target': (B, 64) (-100 for masked squares)
            - 'check': (B, 1)
            - 'king_square': (B, 1)
            - 'threat_target': (B, 64) (-100 for masked squares)
            - 'move_weight': (B,) — optional, multiplies move loss per sample
        """
        total = 0.0
        loss_dict = {}

        # Optionally weight moves based on game result
        if self.USE_MOVE_WEIGHT:
            move_weight = 0.5 + 0.5 * labels.get('eval') # (B, 1)
        else:
            move_weight = None

        loss_eval, _ = self.eval_loss(x, labels["eval"])
        loss_check, _ = self.incheck_loss(x, labels["king_square"], labels["check"])
        loss_threat, _ = self.threat_loss(x, labels["threat_target"])
        loss_move = self.move_loss(move_pred, labels["move_target"], move_weight) 

        total += (
            self.weights['eval'] * loss_eval + 
            self.weights['incheck'] * loss_check +
            self.weights['threat'] * loss_threat +
            self.weights['move'] * loss_move
        )

        loss_dict["eval"] = loss_eval.item()
        loss_dict["incheck"] = loss_check.item()
        loss_dict["threat"] = loss_threat.item()
        loss_dict["move"] = loss_move.item()

        return total, loss_dict