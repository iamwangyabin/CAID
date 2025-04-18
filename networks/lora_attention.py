import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from typing import List, Tuple, Optional

from .lora_components import DynamicRankLoRALayer

class ParallelDynamicLoRAAttention(nn.Module): # Inherit from nn.Module
    def __init__(self,
                 original_attn: Attention, # Accept the original Attention module
                 dim: int, # Keep dim for LoRA layer init
                 num_heads: int, # Keep num_heads for reshaping
                 num_loras=1,
                 max_rank_potential=8,
                 num_stages=1,
                 rank_dropout_p=0.0):
        super().__init__() # Call nn.Module's init
        self.original_attn = original_attn # Store the original module
        self.num_loras = num_loras
        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages
        self.num_heads = num_heads # Store num_heads

        # Use attributes from original_attn where possible
        self.scale = self.original_attn.scale
        self.attn_drop = self.original_attn.attn_drop
        self.proj = self.original_attn.proj # Use the original projection layer
        self.proj_drop = self.original_attn.proj_drop # Use the original projection dropout

        self.lora_q = nn.ModuleList()
        self.lora_v = nn.ModuleList()
        if max_rank_potential > 0 and num_loras > 0:
            for _ in range(num_loras):
                # Ensure DynamicRankLoRALayer receives correct input/output dims
                self.lora_q.append(DynamicRankLoRALayer(dim, dim, max_rank_potential, num_stages, rank_dropout_p))
                self.lora_v.append(DynamicRankLoRALayer(dim, dim, max_rank_potential, num_stages, rank_dropout_p))
        else:
            self.num_loras = 0

    def set_current_stage_and_rank_for_all(self, stage_index, active_rank_count):
        for i in range(self.num_loras):
            if i < len(self.lora_q):
                 self.lora_q[i].set_current_stage_and_rank(stage_index, active_rank_count)
            if i < len(self.lora_v):
                 self.lora_v[i].set_current_stage_and_rank(stage_index, active_rank_count)

    def get_params_for_stage(self, stage_index):
        params = []
        for i in range(self.num_loras):
             if i < len(self.lora_q):
                  params.extend(self.lora_q[i].get_params_for_stage(stage_index))
             if i < len(self.lora_v):
                  params.extend(self.lora_v[i].get_params_for_stage(stage_index))
        return params

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, N, C = x.shape

        # Use original_attn's qkv layer
        qkv_base = self.original_attn.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_base, k_base, v_base = qkv_base.unbind(0)

        # --- Base Path Calculation (using original_attn components) ---
        attn_base = (q_base @ k_base.transpose(-2, -1)) * self.scale
        attn_base = attn_base.softmax(dim=-1)
        attn_base = self.attn_drop(attn_base)
        x_base = (attn_base @ v_base).transpose(1, 2).reshape(B, N, C)
        x_base_attn_proj = self.proj_drop(self.proj(x_base)) # Apply original projection and dropout

        # --- LoRA Path Calculations ---
        list_x_lora_attn_proj = []
        if self.num_loras > 0:
            for i in range(self.num_loras):
                lora_delta_q = torch.zeros_like(q_base)
                lora_delta_v = torch.zeros_like(v_base)

                if i < len(self.lora_q):
                    lora_delta_q_raw = self.lora_q[i](x)
                    lora_delta_q = lora_delta_q_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                if i < len(self.lora_v):
                    lora_delta_v_raw = self.lora_v[i](x)
                    lora_delta_v = lora_delta_v_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # Combine base and LoRA delta
                q_lora = q_base + lora_delta_q
                v_lora = v_base + lora_delta_v
                k_lora = k_base # Key remains unchanged in this setup

                # Recalculate attention for this LoRA path
                attn_lora = (q_lora @ k_lora.transpose(-2, -1)) * self.scale
                attn_lora = attn_lora.softmax(dim=-1)
                attn_lora = self.attn_drop(attn_lora) # Use original dropout
                x_lora = (attn_lora @ v_lora).transpose(1, 2).reshape(B, N, C)

                # Use the original projection layer for LoRA path as well
                x_lora_attn_proj = self.proj_drop(self.proj(x_lora))
                list_x_lora_attn_proj.append(x_lora_attn_proj)

        return x_base_attn_proj, list_x_lora_attn_proj