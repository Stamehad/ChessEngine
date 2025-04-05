import torch
import torch.nn as nn
import torch.nn.functional as F
    
class SwiGLU(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = mlp_ratio * input_dim
        self.linear1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x_proj = self.linear1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return self.linear2(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["embed_dim"]
        self.n_heads = config["num_heads"]
        self.dropout_prob = config.get("dropout", 0.1)
        self.mlp_ratio = config.get("mlp_ratio", 4)

        self.attn_norm = nn.LayerNorm(self.dim)
        self.mlp_norm = nn.LayerNorm(self.dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.n_heads,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(self.dropout_prob)

        self.mlp = nn.Sequential(
            SwiGLU(self.dim, self.mlp_ratio),
            nn.Dropout(self.dropout_prob),
        )

    def forward(self, x):
        # Attention block
        x_norm = self.attn_norm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.attn_dropout(attn_output)

        # MLP block
        x_norm = self.mlp_norm(x)
        x = x + self.mlp(x_norm)
        return x