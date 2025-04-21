import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from typing import List, Tuple, Optional

from .lora_components import DynamicRankLoRALayer


class ParallelDynamicLoRAAttention(nn.Module):
    def __init__(self,
                 original_attn: Attention, # 原始注意力层
                 dim: int, # 输入/输出维度
                 num_heads: int, # 注意力头数量
                 max_rank_potential=8, # LoRA 层的最大潜在秩
                 num_stages=1, # LoRA 阶段数量
                 rank_dropout_p=0.0): # 秩 dropout 概率

        super().__init__()
        self.original_attn = original_attn
        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages
        self.num_heads = num_heads

        self.scale = self.original_attn.scale
        self.attn_drop = self.original_attn.attn_drop
        self.proj = self.original_attn.proj
        self.proj_drop = self.original_attn.proj_drop

        self.lora_q = nn.ModuleDict()
        self.lora_v = nn.ModuleDict()
        for i in range(num_stages):
            stage_key = f'stage_{i}' 
            self.lora_q[stage_key] = DynamicRankLoRALayer(dim, dim, max_rank_potential, rank_dropout_p)
            self.lora_v[stage_key] = DynamicRankLoRALayer(dim, dim, max_rank_potential, rank_dropout_p)

        self.current_stage_index = -1

    def set_current_stage_and_rank(self, stage_index: int, active_rank_count: int):
        self.current_stage_index = stage_index
        stage_key = f'stage_{stage_index}' 
        self.lora_q[stage_key].set_active_rank(active_rank_count)
        self.lora_v[stage_key].set_active_rank(active_rank_count)

    def set_rank_for_stage(self, stage_index: int, active_rank_count: int):
         stage_key = f'stage_{stage_index}'
         if stage_key in self.lora_q: self.lora_q[stage_key].set_active_rank(active_rank_count)
         else: print(f"Warning: LoRA Q layer for Stage key {stage_key} not found.")
         if stage_key in self.lora_v: self.lora_v[stage_key].set_active_rank(active_rank_count)
         else: print(f"Warning: LoRA V layer for Stage key {stage_key} not found.")

    def set_current_stage(self, stage_index: int):
        self.current_stage_index = stage_index

    def get_params_for_stage(self, stage_index: int) -> List[nn.Parameter]:
        params = []
        stage_key = f'stage_{stage_index}'
        if stage_key in self.lora_q:
            # params.extend(self.lora_q[stage_key].get_active_params())
            params.extend(self.lora_q[stage_key].get_params())
        if stage_key in self.lora_v:
            # params.extend(self.lora_v[stage_key].get_active_params())
            params.extend(self.lora_v[stage_key].get_params())
        return list(dict.fromkeys(params))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, N, C = x.shape 
        qkv_base = self.original_attn.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_base, k_base, v_base = qkv_base.unbind(0)

        attn_base_weights = (q_base @ k_base.transpose(-2, -1)) * self.scale
        attn_base_weights = attn_base_weights.softmax(dim=-1)
        attn_base_weights = self.attn_drop(attn_base_weights)

        x_base = (attn_base_weights @ v_base).transpose(1, 2).reshape(B, N, C)
        x_base_attn_proj = self.proj_drop(self.proj(x_base)) 

        stage_outputs = {} 


        # TODO: 有个特别需要注意的点，就是比如训练到了stage1，那么这时候stage0的activate rank=0.这个以后可能需要修正。
        # active_rank_count 没有被包含在 checkpoint 中是因为它是 DynamicRankLoRALayer 类中的一个标准的 Python 整数属性，而不是 PyTorch 的 nn.Parameter 或通过 register_buffer 注册的持久性 buffer。PyTorch 的 checkpoint 机制默认只保存模型的可学习参数和持久性 buffer，而不会自动保存普通的 Python 属性。这是为了提高效率和灵活性，只保存模型可计算状态所必需的部分。要保存和恢复 active_rank_count，需要在代码中显式地将其注册为 buffer 或在自定义的状态字典方法中处
        if 0 <= self.current_stage_index < self.num_stages:
            for i in range(self.current_stage_index + 1):
                stage_key = f'stage_{i}' 
                if stage_key in self.lora_q and stage_key in self.lora_v:
                    lora_q_layer = self.lora_q[stage_key]
                    lora_v_layer = self.lora_v[stage_key]

                    lora_delta_q_raw = lora_q_layer(x) + torch.zeros_like(x)
                    lora_delta_q_stage = lora_delta_q_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                    lora_delta_v_raw = lora_v_layer(x) + torch.zeros_like(x)
                    lora_delta_v_stage = lora_delta_v_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                    q_stage = q_base + lora_delta_q_stage
                    v_stage = v_base + lora_delta_v_stage
                    k_stage = k_base 

                    attn_stage = (q_stage @ k_stage.transpose(-2, -1)) * self.scale
                    attn_stage = attn_stage.softmax(dim=-1)
                    attn_stage = self.attn_drop(attn_stage)

                    x_stage_combined = (attn_stage @ v_stage).transpose(1, 2).reshape(B, N, C)
                    stage_output = self.proj_drop(self.proj(x_stage_combined))
                    stage_outputs[stage_key] = stage_output

                else:
                    print(f"Warning: LoRA layer for Stage key {stage_key} not found during parallel computation.")

        return x_base_attn_proj, stage_outputs