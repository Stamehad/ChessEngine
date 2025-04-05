import torch.nn as nn

class InCheckHead(nn.Module):
    """
    Predicts whether the king is in check after a move.
    Returns a binary logit indicating if the king is in check.
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config["embed_dim"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, king_square_repr):  # king_square_repr: (B, 1, H)
        return self.classifier(king_square_repr).squeeze(-1)  # (B, 1)