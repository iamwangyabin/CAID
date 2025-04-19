import torch
import torch.nn as nn


class LoRAPathClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, lora_cls_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(lora_cls_tensor)
        return logits