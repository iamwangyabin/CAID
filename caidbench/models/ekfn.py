from __future__ import annotations

import torch
from torch import nn


class ExpertKnowledgeFusionNetwork(nn.Module):
    """Transformer-based EKFN used by E3.

    Input: [B, num_experts, embed_dim]. The transformer contextualizes expert
    evidence. Its output gates the original embeddings by elementwise product;
    gated embeddings are flattened and fed to a two-layer MLP.
    """

    def __init__(self, embed_dim: int, max_experts: int = 64, num_classes: int = 2, transformer_layers: int = 2, nhead: int = 4, mlp_hidden: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_experts = int(max_experts)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=max(embed_dim * 4, 128), dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_experts, embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(max_experts * embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, expert_embeddings: torch.Tensor) -> torch.Tensor:
        b, e, d = expert_embeddings.shape
        if e > self.max_experts:
            raise ValueError(f"EKFN max_experts={self.max_experts}, got {e}")
        x = expert_embeddings + self.pos[:, :e]
        ctx = self.transformer(x)
        gated = ctx * expert_embeddings
        if e < self.max_experts:
            pad = gated.new_zeros(b, self.max_experts - e, d)
            gated = torch.cat([gated, pad], dim=1)
        return self.mlp(gated.reshape(b, -1))
