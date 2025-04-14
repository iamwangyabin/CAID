import torch
import torch.nn as nn
import math
from timm.models import create_model

class _LoRA_qkv_timm(nn.Module):
    """
    LoRA module for timm ViT qkv projection.
    """
    def __init__(self, qkv: nn.Module, dim: int, r: int):
        super().__init__()
        self.qkv = qkv
        self.dim = dim
        # LoRA low-rank adapters for q and v
        self.w_a_q = nn.Linear(dim, r, bias=False)
        self.w_b_q = nn.Linear(r, dim, bias=False)
        self.w_a_v = nn.Linear(dim, r, bias=False)
        self.w_b_v = nn.Linear(r, dim, bias=False)
        # New LoRA adapters for staged adaptation
        self.wnew_a_q = nn.Linear(dim, r, bias=False)
        self.wnew_b_q = nn.Linear(r, dim, bias=False)
        self.wnew_a_v = nn.Linear(dim, r, bias=False)
        self.wnew_b_v = nn.Linear(r, dim, bias=False)
        self.reset_lora_new_parameters()

    def forward(self, x, use_new=True):
        qkv = self.qkv(x)
        # Add LoRA adaptation
        qkv[:, :, :self.dim] += self.w_b_q(self.w_a_q(x))
        qkv[:, :, -self.dim:] += self.w_b_v(self.w_a_v(x))
        if use_new:
            qkv[:, :, :self.dim] += self.wnew_b_q(self.wnew_a_q(x))
            qkv[:, :, -self.dim:] += self.wnew_b_v(self.wnew_a_v(x))
        return qkv

    def reset_lora_new_parameters(self):
        nn.init.kaiming_uniform_(self.wnew_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.wnew_b_q.weight)
        nn.init.kaiming_uniform_(self.wnew_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.wnew_b_v.weight)

    def accumulate_lora(self):
        # Accumulate new LoRA weights into main LoRA
        self.w_a_q.weight.data += self.wnew_a_q.weight.data
        self.w_b_q.weight.data += self.wnew_b_q.weight.data
        self.w_a_v.weight.data += self.wnew_a_v.weight.data
        self.w_b_v.weight.data += self.wnew_b_v.weight.data
        self.reset_lora_new_parameters()

class LoRA_ViT_timm(nn.Module):
    """
    Vision Transformer with LoRA for domain incremental learning.
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=50, r=4):
        super().__init__()
        # Load timm ViT backbone
        self.vit = create_model(model_name, pretrained=True, num_classes=num_classes)
        self.r = r
        # Insert LoRA modules into all transformer blocks
        for i, blk in enumerate(self.vit.blocks):
            dim = blk.attn.qkv.in_features
            blk.attn.qkv = _LoRA_qkv_timm(blk.attn.qkv, dim, r)

    def forward(self, x):
        return self.vit(x)

    def update_and_reset_lora_parameters(self):
        # Accumulate and reset LoRA in all blocks
        for blk in self.vit.blocks:
            if isinstance(blk.attn.qkv, _LoRA_qkv_timm):
                blk.attn.qkv.accumulate_lora()

# Example usage:
if __name__ == "__main__":
    model = LoRA_ViT_timm(model_name='vit_base_patch16_224', num_classes=50, r=4)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    # Simulate loss plateau and LoRA reset
    model.update_and_reset_lora_parameters()