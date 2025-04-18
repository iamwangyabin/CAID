import torch
import torch.nn as nn


class LoRAPathAvgPoolClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, lora_cls_tensor: torch.Tensor) -> torch.Tensor:
        if lora_cls_tensor.dim() != 3 or lora_cls_tensor.shape[-1] != self.embed_dim:
            raise ValueError(f"输入张量形状应为 (B, depth, {self.embed_dim}), "
                             f"但得到 {lora_cls_tensor.shape}")

        pooled_features = torch.mean(lora_cls_tensor, dim=1)

        logits = self.classifier(pooled_features)
        return logits