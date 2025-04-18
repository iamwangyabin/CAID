import torch
import torch.nn as nn
import math
from typing import List, Dict, Tuple, Optional

class Rank1LoRAComponent(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(in_features, 1) / math.sqrt(in_features))
        self.lora_b = nn.Parameter(torch.zeros(1, out_features))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        input_shape = x.shape
        if x.dim() > 2:
            x_reshaped = x.reshape(-1, self.in_features)
            delta_h_reshaped = (x_reshaped @ self.lora_a) @ self.lora_b
            delta_h = delta_h_reshaped.view(*input_shape[:-1], self.out_features)
        else:
             delta_h = (x @ self.lora_a) @ self.lora_b
        return delta_h


class DynamicRankLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, max_rank_potential, num_stages, rank_dropout_p=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages
        self.rank_dropout_p = rank_dropout_p
        if not (0.0 <= self.rank_dropout_p <= 1.0):
            raise ValueError(f"rank_dropout_p 必须在 0 和 1 之间, 但得到 {rank_dropout_p}")

        self.lora_components_per_stage = nn.ModuleDict()
        for i in range(num_stages):
            stage_key = str(i)
            stage_components = nn.ModuleList()
            for _ in range(self.max_rank_potential):
                stage_components.append(Rank1LoRAComponent(self.in_features, self.out_features))
            self.lora_components_per_stage[stage_key] = stage_components

        self.stage_active_ranks = {i: 0 for i in range(num_stages)}
        self.current_stage_index = -1

    def set_current_stage_and_rank(self, stage_index, active_rank_count):
        """设置当前激活的阶段索引以及该阶段要使用的秩数量。"""
        if stage_index < 0:
             self.current_stage_index = -1
             return

        self.current_stage_index = stage_index
        stage_key = str(stage_index)
        if stage_key in self.lora_components_per_stage:
            self.stage_active_ranks[stage_index] = min(active_rank_count, self.max_rank_potential)
        else:
             print(f"警告: 阶段 {stage_key} 在 set_current_stage_and_rank 中未找到。")
             self.stage_active_ranks[stage_index] = 0

    def forward(self, x):
        """计算当前激活阶段和激活秩对应的LoRA增量。"""
        lora_delta = 0.0
        if self.current_stage_index < 0:
            return lora_delta

        stage_key = str(self.current_stage_index)
        if stage_key not in self.lora_components_per_stage:
            return lora_delta

        target_rank = self.stage_active_ranks.get(self.current_stage_index, 0)
        effective_rank = target_rank

        if self.training and self.rank_dropout_p > 0 and target_rank > 0:
            if torch.rand(1).item() < self.rank_dropout_p:
                effective_rank = torch.randint(0, target_rank, (1,)).item()

        if effective_rank > 0:
            stage_components = self.lora_components_per_stage[stage_key]
            for i in range(min(effective_rank, len(stage_components))):
                lora_delta = lora_delta + stage_components[i](x)

        return lora_delta

    def get_params_for_stage(self, stage_index):
        """获取指定阶段中，当前激活秩对应的所有LoRA参数（用于优化器）。"""
        params = []
        stage_key = str(stage_index)
        if stage_key in self.lora_components_per_stage:
            stage_components = self.lora_components_per_stage[stage_key]
            active_rank = self.stage_active_ranks.get(stage_index, 0)
            for i in range(min(active_rank, len(stage_components))):
                 params.extend(stage_components[i].parameters())
        return params