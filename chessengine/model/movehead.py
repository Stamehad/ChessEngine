import torch.nn as nn

class MoveHead(nn.Module):
    """
    Head to predict the next move. Each square can be moved to one of 7 classes.
    The 7 classes are: 0 (empty square), 1-6 (piece to move: pawn, knight, bishop, rook, queen, king),
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config["embed_dim"]
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 7)  # predict one of 13 piece classes
        )

    def forward(self, x):  # x: (B, 65, H)
        x = x[:, 1:, :] # Remove the [EVAL] token — shape becomes (B, 64, H)
        return self.head(x)  # (B, 64, 7)