import torch
import torch.nn as nn
from chessengine.model.loss_heads import EvalLoss, ProbEvalLoss, InCheckLoss, ThreatLoss, LegalMoveLoss

class Loss(nn.Module):
    """
    Aggregates all individual losses into a single weighted total loss.
    Weights for each loss component are read from the config.
    Optionally weights move loss per-sample based on alignment with game result.
    """
    def __init__(self, config):
        super().__init__()

        self.USE_PROB_EVAL = config['model'].get('use_prob_eval', True)
        if self.USE_PROB_EVAL:
            self.prob_eval_loss = ProbEvalLoss(config)
            eval_weight = config['loss'].get('prob_eval_loss_weight', 1.0)
        else:
            self.eval_loss = EvalLoss(config)
            eval_weight = config['loss'].get('eval_loss_weight', 1.0)
        self.incheck_loss = InCheckLoss(config)
        self.threat_loss = ThreatLoss(config)
        self.move_loss = LegalMoveLoss(config)

        # Default weights if not specified in config
        loss_config = config['loss']
        self.weights = {
            'eval': eval_weight,
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
            - 'terminal_flag': (B, 1)
            - 'legal_moves': (B,64,L) 
            - 'true_index': (B, ) 
        """
        total = 0.0

        # Optionally weight moves based on game result
        if self.USE_MOVE_WEIGHT:
            move_weight = 0.5 + 0.5 * labels.get('eval') # (B, 1)
        else:
            move_weight = None
        if self.USE_PROB_EVAL:
            loss_eval, eval_logits = self.prob_eval_loss(x, labels["eval"])
        else:
            loss_eval, _ = self.eval_loss(x, labels["eval"])
            eval_logits = None
           
        loss_check, _ = self.incheck_loss(x, labels["king_square"], labels["check"])
        loss_threat, _ = self.threat_loss(x, labels["threat_target"])
        loss_move, move_logits = self.move_loss(move_pred, labels["legal_moves"], labels["true_index"], move_weight) 

        total += (
            self.weights['eval'] * loss_eval + 
            self.weights['incheck'] * loss_check +
            self.weights['threat'] * loss_threat +
            self.weights['move'] * loss_move
        )

        loss_dict = {
            "eval": loss_eval.item(),
            "incheck": loss_check.item(),
            "threat": loss_threat.item(),
            "move": loss_move.item(),
        }

        logit_dict = {
            "eval": eval_logits,    # (B, 3)
            "move": move_logits     # (B, L)
        }

        return total, loss_dict, logit_dict # (1,)  