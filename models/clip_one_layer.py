import torch
from torch import nn


class ClipOneLayer(nn.Module):
    """Minimal 1-layer Transformer used for CLIP feature matching."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, extra=None) -> torch.Tensor:
        """Forward pass taking patch embeddings and time."""
        time_emb = self.time_mlp(t[:, None])
        x = x + time_emb[:, None, :]
        return self.encoder(x)

