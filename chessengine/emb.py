import torch
import torch.nn as nn

class BoardEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.F = config["feature_dim"]
        self.H = config["embed_dim"]
        self.board_size = 64  # 8x8 board

        self.proj = nn.Linear(self.F, self.H)

        # Learnable [VAL] token
        self.val_token = nn.Parameter(torch.zeros(1, 1, self.H))
        nn.init.trunc_normal_(self.val_token, std=0.02)

        # Positional embeddings: one extra for the [VAL] token
        self.pos_embed = nn.Embedding(self.board_size + 1, self.H)

    def forward(self, x):
        # x: (B, 8, 8, F)
        B = x.shape[0]
        x = x.view(B, self.board_size, self.F)  # (B, 64, F)
        x = self.proj(x)  # (B, 64, H)

        # Prepend the val token
        val_token = self.val_token.expand(B, 1, -1)  # (B, 1, H)
        x = torch.cat([val_token, x], dim=1)  # (B, 65, H)

        # Positional indices: 0 for [VAL], 1–64 for board squares
        pos_ids = torch.arange(self.board_size + 1, device=x.device).unsqueeze(0)  # (1, 65)
        pos_embed = self.pos_embed(pos_ids)  # (1, 65, H)
        x = x + pos_embed  # broadcasted over batch
        
        return x