import torch
import torch.nn as nn


class LoRAPathClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, lora_cls_tensor: torch.Tensor) -> torch.Tensor:
        # lora_cls_tensor 形状应为 (batch_size, num_blocks, embed_dim)
        # 对 num_blocks 维度进行平均池化
        pooled_features = torch.mean(lora_cls_tensor, dim=1) # 形状变为 (batch_size, embed_dim)
        logits = self.classifier(pooled_features) # 形状变为 (batch_size, num_classes)
        return logits