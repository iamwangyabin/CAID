import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TimmBlock # Rename to avoid conflict
from typing import List, Tuple

from .lora_attention import ParallelDynamicLoRAAttention

class ParallelDynamicLoRABlock(nn.Module):
    """
    并行动态 LoRA Block 模块。
    在原始 Block 基础上，集成并行多阶段动态秩 LoRA 注意力机制，支持增量训练和多阶段参数管理。
    """

    def __init__(
            self,
            original_block: TimmBlock,
            max_rank_potential=8,
            num_stages=1,
            rank_dropout_p=0.0):
        super().__init__()
        self.original_block = original_block

        self.attn = ParallelDynamicLoRAAttention(
            original_attn=self.original_block.attn,
            dim=self.original_block.attn.dim,
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
        """
        前向传播。返回主分支输出和所有激活阶段的完整 Block 输出（并行）。
        """
        x_norm1 = self.original_block.norm1(x)
        x_base_attn_proj, stage_attn_outputs = self.attn(x_norm1)

        # 主分支 Block 输出
        x_base_residual = x + self.original_block.drop_path(x_base_attn_proj)
        x_base_mlp = self.original_block.mlp(self.original_block.norm2(x_base_residual))
        x_base_output = x_base_residual + self.original_block.drop_path(x_base_mlp)

        # 并行分支：每个阶段的完整 Block 输出
        stage_block_outputs = {}
        for stage_key, stage_attn_proj in stage_attn_outputs.items():
            stage_residual = x + self.original_block.drop_path(stage_attn_proj)
            stage_mlp = self.original_block.mlp(self.original_block.norm2(stage_residual))
            stage_block_output = stage_residual + self.original_block.drop_path(stage_mlp)
            stage_block_outputs[stage_key] = stage_block_output

        return x_base_output, stage_block_outputs