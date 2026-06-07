from __future__ import annotations

from torch import nn

from .e3_official import OfficialE3FusionNetwork


class ExpertKnowledgeFusionNetwork(OfficialE3FusionNetwork):
    """Official E3 EKFN wrapper kept under the legacy import path."""

    def __init__(
        self,
        embed_dim: int,
        max_experts: int | None = None,
        num_classes: int = 2,
        transformer_layers: int = 5,
        nhead: int = 8,
        mlp_hidden: int | None = None,
        dropout: float | None = None,
        activation: str | None = None,
        num_experts: int | None = None,
    ) -> None:
        del max_experts, mlp_hidden, dropout, activation
        super().__init__(
            expert_n_features=embed_dim,
            num_experts=int(num_experts or 1),
            embed_dim=200,
            depth=int(transformer_layers),
            num_heads=int(nhead),
            num_classes=int(num_classes),
        )

    @property
    def trainable_head(self) -> nn.Module:
        return self.classifier_head
