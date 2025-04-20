import torch
import torch.nn as nn


class LoRAPathClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, lora_cls_tensor: torch.Tensor) -> torch.Tensor:
        pooled_features = torch.mean(lora_cls_tensor, dim=1) 
        logits = self.classifier(pooled_features) 
        return logits