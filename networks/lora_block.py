import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TimmBlock # Rename to avoid conflict
from typing import List, Tuple

from .lora_attention import ParallelDynamicLoRAAttention

class ParallelDynamicLoRABlock(nn.Module):
    def __init__(
            self,
            original_block: TimmBlock,
            embed_dim: int, # 添加 embed_dim 参数
            max_rank_potential=8,
            num_stages=1,
            rank_dropout_p=0.0):
        super().__init__()
        self.original_block = original_block

        self.attn = ParallelDynamicLoRAAttention(
            original_attn=self.original_block.attn,
            dim=embed_dim, # 使用传递进来的 embed_dim
            num_heads=self.original_block.attn.num_heads,
            max_rank_potential=max_rank_potential,
            num_stages=num_stages,
            rank_dropout_p=rank_dropout_p
        )

    def set_current_stage_and_rank(self, stage_index: int, active_rank_count: int):
        self.attn.set_current_stage_and_rank(stage_index, active_rank_count)

    def set_current_stage(self, stage_index: int):
        self.attn.set_current_stage(stage_index)

    def set_rank_for_stage(self, stage_index: int, active_rank_count: int):
        self.attn.set_rank_for_stage(stage_index, active_rank_count)

    def get_params_for_stage(self, stage_index: int):
        return self.attn.get_params_for_stage(stage_index)

    def forward(self, x) -> Tuple[torch.Tensor, dict]:
        x_norm1 = self.original_block.norm1(x)
        x_base_attn_proj, stage_attn_outputs = self.attn(x_norm1)

        drop_path = self.original_block.drop_path if hasattr(self.original_block, 'drop_path') else nn.Identity()


        x_base_residual = x + drop_path(x_base_attn_proj)
        x_base_mlp = self.original_block.mlp(self.original_block.norm2(x_base_residual))
        x_base_output = x_base_residual + drop_path(x_base_mlp)

        stage_block_outputs = {}
        for stage_key, stage_attn_proj in stage_attn_outputs.items():
            stage_residual = x + drop_path(stage_attn_proj)
            stage_mlp = self.original_block.mlp(self.original_block.norm2(stage_residual))
            stage_block_output = stage_residual + drop_path(stage_mlp)
            stage_block_outputs[stage_key] = stage_block_output

        return x_base_output, stage_block_outputs