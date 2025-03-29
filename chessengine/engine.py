import torch
import torch.nn as nn
from chessengine.emb import BoardEmbedding
from chessengine.attn import TransformerBlock
from chessengine.movehead import MoveHead

class RecycleEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.recycle_emb = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.recycle_emb(x)

class ChessEngine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = BoardEmbedding(config)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["num_layers"])])
        self.recycle_embedding = RecycleEmbedding(config["embed_dim"])
        self.move_head = MoveHead(config)
        self.max_recycles = config.get("n_recycles", 4)

    def forward(self, x):
        # Initial embedding from one-hot input
        x0 = self.embedding(x)
        x = self.blocks(x0)

        # Determine number of recycling steps
        n_recycles = torch.randint(1, self.max_recycles + 1, (1,)).item() if self.training else self.max_recycles

        # Residual recycling loop
        for _ in range(n_recycles - 1):
            delta = self.recycle_embedding(x.detach())
            x = self.blocks(x0 + delta)

        move_pred = self.move_head(x) # (B, 64, 7)
        return x, move_pred