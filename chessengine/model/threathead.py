import torch.nn as nn

class ThreatHead(nn.Module):
    """
    Predicts which squares become newly threatened after a move.
    Returns a per-square binary logit indicating new threats.
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config["embed_dim"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, x):  # output tensor (B, 65, H)
        x = x[:, 1:, :] # Remove the [EVAL] token — shape becomes (B, 64, H)
        return self.classifier(x)  # (B, 64, 2)