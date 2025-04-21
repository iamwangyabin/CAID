import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from typing import List, Tuple, Optional

from .lora_components import DynamicRankLoRALayer


class ParallelDynamicLoRAAttention(nn.Module):
    """
    并行动态 LoRA 注意力模块。
    在原始注意力层的基础上，并行添加多个具有动态秩的 LoRA 层，以支持增量训练。
    """

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

        # 继承原始注意力层的属性
        self.scale = self.original_attn.scale
        self.attn_drop = self.original_attn.attn_drop
        self.proj = self.original_attn.proj
        self.proj_drop = self.original_attn.proj_drop

        # 为每个阶段创建动态秩 LoRA 层
        self.lora_q = nn.ModuleDict()
        self.lora_v = nn.ModuleDict()
        for i in range(num_stages):
            stage_key = f'stage_{i}' # 统一键格式为 'stage_i'
            self.lora_q[stage_key] = DynamicRankLoRALayer(dim, dim, max_rank_potential, rank_dropout_p)
            self.lora_v[stage_key] = DynamicRankLoRALayer(dim, dim, max_rank_potential, rank_dropout_p)

        # 当前激活的 LoRA 阶段索引
        self.current_stage_index = -1

    def set_current_stage_and_rank(self, stage_index: int, active_rank_count: int):
        """
        设置当前激活的 LoRA 阶段及其秩。
        """
        if not (0 <= stage_index < self.num_stages):
            print(f"警告: 提供了无效的阶段索引 {stage_index}。将重置 current_stage_index 为 -1。")
            self.current_stage_index = -1
            return

        self.current_stage_index = stage_index

        stage_key = f'stage_{stage_index}' # 统一键格式为 'stage_i'
        if stage_key in self.lora_q:
            self.lora_q[stage_key].set_active_rank(active_rank_count)
        else:
             print(f"警告: 在 set_current_stage_and_rank 中未找到 Stage key {stage_key} 对应的 LoRA Q 层。")

        if stage_key in self.lora_v:
            self.lora_v[stage_key].set_active_rank(active_rank_count)
        else:
            print(f"警告: 在 set_current_stage_and_rank 中未找到 Stage key {stage_key} 对应的 LoRA V 层。")


    def set_rank_for_stage(self, stage_index: int, active_rank_count: int):
         """
         为指定阶段设置 LoRA 秩。
         """
         if not (0 <= stage_index < self.num_stages):
             print(f"警告: 尝试为无效的阶段索引 {stage_index} 设置秩。操作已忽略。")
             return
         stage_key = f'stage_{stage_index}' # 统一键格式为 'stage_i'
         if stage_key in self.lora_q: self.lora_q[stage_key].set_active_rank(active_rank_count)
         else: print(f"警告: 未找到 Stage key {stage_key} 对应的 LoRA Q 层。")
         if stage_key in self.lora_v: self.lora_v[stage_key].set_active_rank(active_rank_count)
         else: print(f"警告: 未找到 Stage key {stage_key} 对应的 LoRA V 层。")

    def set_current_stage(self, stage_index: int):
        """
        设置当前激活的 LoRA 阶段。
        """
        if not (0 <= stage_index < self.num_stages):
            print(f"警告: 提供了无效的阶段索引 {stage_index}。将重置 current_stage_index 为 -1，禁用所有 LoRA 阶段。")
            self.current_stage_index = -1
        else:
            self.current_stage_index = stage_index


    def get_params_for_stage(self, stage_index: int) -> List[nn.Parameter]:
        """
        获取指定阶段 LoRA 层的可训练参数。
        """
        params = []
        if not (0 <= stage_index < self.num_stages):
            print(f"警告: 在 get_params_for_stage 中请求了无效的阶段索引 {stage_index}。")
            return params
        stage_key = f'stage_{stage_index}' # 统一键格式为 'stage_i'
        if stage_key in self.lora_q:
             params.extend(self.lora_q[stage_key].get_active_params())
        if stage_key in self.lora_v:
             params.extend(self.lora_v[stage_key].get_active_params())
        return list(dict.fromkeys(params)) # 返回去重后的参数列表



    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        前向传播。计算原始注意力输出和每个激活 LoRA 阶段的输出。
        """

        B, N, C = x.shape # 批次大小，序列长度，特征维度


        # 计算原始注意力层的 QKV
        qkv_base = self.original_attn.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_base, k_base, v_base = qkv_base.unbind(0) # 分离 Q, K, V

        # 计算原始注意力权重
        attn_base_weights = (q_base @ k_base.transpose(-2, -1)) * self.scale
        attn_base_weights = attn_base_weights.softmax(dim=-1)
        attn_base_weights = self.attn_drop(attn_base_weights)

        # 计算原始注意力输出
        x_base = (attn_base_weights @ v_base).transpose(1, 2).reshape(B, N, C)
        x_base_attn_proj = self.proj_drop(self.proj(x_base)) # 原始注意力层的投影输出

        stage_outputs = {} # 存储每个阶段的输出

        if 0 <= self.current_stage_index < self.num_stages:
            for i in range(self.current_stage_index + 1):
                stage_key = f'stage_{i}' 
                if stage_key in self.lora_q and stage_key in self.lora_v:
                    lora_q_layer = self.lora_q[stage_key]
                    lora_v_layer = self.lora_v[stage_key]

                    # 如果当前阶段的 LoRA 层有激活的秩
                    if lora_q_layer.active_rank_count > 0 or lora_v_layer.active_rank_count > 0:

                        lora_delta_q_raw = lora_q_layer(x) + torch.zeros_like(x)
                        lora_delta_q_stage = lora_delta_q_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                        lora_delta_v_raw = lora_v_layer(x) + torch.zeros_like(x)
                        lora_delta_v_stage = lora_delta_v_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                        # 将 LoRA 增量添加到原始 Q 和 V
                        q_stage = q_base + lora_delta_q_stage
                        v_stage = v_base + lora_delta_v_stage
                        k_stage = k_base # K 保持不变

                        # 计算当前阶段的注意力权重
                        attn_stage = (q_stage @ k_stage.transpose(-2, -1)) * self.scale
                        attn_stage = attn_stage.softmax(dim=-1)
                        attn_stage = self.attn_drop(attn_stage)

                        # 计算当前阶段的注意力输出
                        x_stage_combined = (attn_stage @ v_stage).transpose(1, 2).reshape(B, N, C)

                        # 计算当前阶段的投影输出
                        stage_output = self.proj_drop(self.proj(x_stage_combined))

                        # 存储当前阶段的输出
                        stage_outputs[stage_key] = stage_output

                else:
                    print(f"警告: 在并行计算中未找到 Stage key {stage_key} 对应的 LoRA 层。")

        return x_base_attn_proj, stage_outputs