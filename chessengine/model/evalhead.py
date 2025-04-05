import torch
import torch.nn as nn

class EvalHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config["embed_dim"]),
            nn.Linear(config["embed_dim"], config.get("eval_hidden_dim", 128)),
            nn.GELU(),
            nn.Linear(config.get("eval_hidden_dim", 128), 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        # x: (B, 65, H) → we want only the first token
        val_token = x[:, 0]  # (B, H)
        return self.mlp(val_token)  # (B, 1)