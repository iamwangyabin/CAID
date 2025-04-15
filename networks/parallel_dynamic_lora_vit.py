import torch
import torch.nn as nn
import math
import timm
from timm.models.vision_transformer import Block, Mlp, Attention, PatchEmbed, VisionTransformer
from timm.models import create_model
from functools import partial
from typing import List, Dict, Tuple, Optional



# 单个秩为1的LoRA更新组件。
class Rank1LoRAComponent(nn.Module):
    """ Represents a single rank-1 LoRA update. """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize parameters, maybe scale initialization
        self.lora_a = nn.Parameter(torch.randn(in_features, 1) / math.sqrt(in_features))
        self.lora_b = nn.Parameter(torch.zeros(1, out_features)) # Note: B is (1, out) for matmul
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x shape: (B, N, C_in) or (B*N, C_in)
        # lora_a: (C_in, 1)
        # lora_b: (1, C_out)
        # Need to handle potential 3D input (B, N, C)
        input_shape = x.shape
        if x.dim() > 2:
            x_reshaped = x.reshape(-1, self.in_features) # (B*N, C_in)
            delta_h_reshaped = (x_reshaped @ self.lora_a) @ self.lora_b # (B*N, C_out)
            delta_h = delta_h_reshaped.view(*input_shape[:-1], self.out_features) # (B, N, C_out)
        else: # Handle 2D input (e.g., if applied directly in MLP)
             delta_h = (x @ self.lora_a) @ self.lora_b # (B, C_out)
        return delta_h




# 管理单个线性变换（如Attention中的Q或V投影）的多个阶段的LoRA更新。
# 每个阶段包含一组秩为1的LoRA组件（Rank1LoRAComponent）。
# "动态秩"指的是在特定阶段，可以控制激活多少个该阶段的秩1组件来构成最终的LoRA更新。
# 这允许在不同训练阶段调整LoRA的有效秩，并可能增量地添加新的LoRA参数。
class DynamicRankLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, max_rank_potential, num_stages, rank_dropout_p=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages
        self.rank_dropout_p = rank_dropout_p
        if not (0.0 <= self.rank_dropout_p <= 1.0):
            raise ValueError(f"rank_dropout_p must be between 0 and 1, but got {rank_dropout_p}")

        # Pre-initialize components for all stages
        self.lora_components_per_stage = nn.ModuleDict()
        for i in range(num_stages):
            stage_key = str(i)
            stage_components = nn.ModuleList()
            for _ in range(self.max_rank_potential):
                stage_components.append(Rank1LoRAComponent(self.in_features, self.out_features))
            self.lora_components_per_stage[stage_key] = stage_components

        self.stage_active_ranks = {i: 0 for i in range(num_stages)} # Initialize all stage ranks to 0
        self.current_stage_index = -1 # No stage active initially

    # Removed add_stage method as components are pre-initialized
    def set_current_stage_and_rank(self, stage_index, active_rank_count):
        if stage_index < 0:
             self.current_stage_index = -1
             return

        self.current_stage_index = stage_index
        stage_key = str(stage_index)
        # Stage key should always exist now due to pre-initialization
        if stage_key in self.lora_components_per_stage:
            max_possible_rank = len(self.lora_components_per_stage[stage_key]) # Should equal max_rank_potential
            # Ensure active_rank_count does not exceed the maximum potential rank
            self.stage_active_ranks[stage_index] = min(active_rank_count, self.max_rank_potential)
        else:
             # This case should ideally not happen with pre-initialization
             print(f"Warning: Stage key {stage_key} not found during set_current_stage_and_rank, though it should have been pre-initialized.")
             self.stage_active_ranks[stage_index] = 0
    def forward(self, x):
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
        params = []
        stage_key = str(stage_index)
        if stage_key in self.lora_components_per_stage:
            stage_components = self.lora_components_per_stage[stage_key]
            active_rank = self.stage_active_ranks.get(stage_index, 0)
            for i in range(min(active_rank, len(stage_components))):
                 params.extend(stage_components[i].parameters())
        return params
















# 继承自timm的Attention模块，并为其增加了多个并行的动态LoRA路径。
# 它包含一个基础的QKV计算路径（与原始Attention类似）以及`num_loras`个独立的LoRA路径。
# 每个LoRA路径使用独立的`DynamicRankLoRALayer`来修改Q和V（K保持不变）。
# `forward`方法返回基础路径的输出，以及一个包含所有并行LoRA路径输出的列表。
# 提供方法来统一控制其内部所有`DynamicRankLoRALayer`实例的阶段(stage)和激活秩(active rank)。
class ParallelDynamicLoRAAttention(Attention):
    """ Attention module supporting multiple parallel DynamicRankLoRALayer paths for Q and V. """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 num_loras=1, max_rank_potential=8, num_stages=1, rank_dropout_p=0.0):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_loras = num_loras
        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages # Store num_stages

        self.lora_q = nn.ModuleList()
        self.lora_v = nn.ModuleList()
        if max_rank_potential > 0 and num_loras > 0:
            for _ in range(num_loras):
                self.lora_q.append(DynamicRankLoRALayer(dim, dim, max_rank_potential, num_stages, rank_dropout_p))
                self.lora_v.append(DynamicRankLoRALayer(dim, dim, max_rank_potential, num_stages, rank_dropout_p))
        else:
            self.num_loras = 0 # Ensure num_loras is 0 if rank is 0

    # Removed add_stage_for_all method
    def set_current_stage_and_rank_for_all(self, stage_index, active_rank_count):
        """ Sets the current stage and active rank for all contained DynamicRankLoRALayer instances. """
        for i in range(self.num_loras):
            self.lora_q[i].set_current_stage_and_rank(stage_index, active_rank_count)
            self.lora_v[i].set_current_stage_and_rank(stage_index, active_rank_count)

    def get_params_for_stage(self, stage_index):
        """ Gets parameters for a specific stage from all contained LoRA layers. """
        params = []
        for i in range(self.num_loras):
            params.extend(self.lora_q[i].get_params_for_stage(stage_index))
            params.extend(self.lora_v[i].get_params_for_stage(stage_index))
        return params

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, N, C = x.shape
        qkv_base = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_base, k_base, v_base = qkv_base.unbind(0)

        # --- Base Path ---
        attn_base = (q_base @ k_base.transpose(-2, -1)) * self.scale
        attn_base = attn_base.softmax(dim=-1)
        attn_base = self.attn_drop(attn_base)
        x_base = (attn_base @ v_base).transpose(1, 2).reshape(B, N, C)
        x_base_attn = self.proj(x_base) # Apply output projection

        # --- LoRA Paths ---
        list_x_lora_attn = []
        if self.num_loras > 0:
            for i in range(self.num_loras):
                # Calculate LoRA deltas using DynamicRankLoRALayer
                lora_delta_q = self.lora_q[i](x) # Shape: (B, N, C)
                lora_delta_v = self.lora_v[i](x) # Shape: (B, N, C)

                # Reshape LoRA deltas
                lora_delta_q = lora_delta_q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                lora_delta_v = lora_delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # Apply LoRA to Q and V
                q_lora = q_base + lora_delta_q
                v_lora = v_base + lora_delta_v
                k_lora = k_base

                # Calculate LoRA Attention
                attn_lora = (q_lora @ k_lora.transpose(-2, -1)) * self.scale
                attn_lora = attn_lora.softmax(dim=-1)
                attn_lora = self.attn_drop(attn_lora)
                x_lora = (attn_lora @ v_lora).transpose(1, 2).reshape(B, N, C)
                x_lora_attn = self.proj(x_lora) # Apply output projection
                list_x_lora_attn.append(x_lora_attn)

        return x_base_attn, list_x_lora_attn











# 继承自timm的Block模块，是构成ViT的基本单元。
# 主要修改是将内部的Attention模块替换为`ParallelDynamicLoRAAttention`。
# 它保留了标准的MLP层和残差连接结构。
# `forward`方法不仅计算基于基础注意力路径的主输出(`x_output`)，
# 还为每个并行LoRA注意力路径计算并返回一个经过完整Block（包括MLP和残差）处理后的特征列表(`lora_combined_features`)。
# 它将阶段(stage)和秩(rank)的控制信号传递给内部的`ParallelDynamicLoRAAttention`模块。
class ParallelDynamicLoRABlock(Block):
    """ ViT Block using ParallelDynamicLoRAAttention. """
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            num_loras=1, max_rank_potential=8, num_stages=1, rank_dropout_p=0.0):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.num_loras = num_loras
        # Replace Attention module
        self.attn = ParallelDynamicLoRAAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            num_loras=num_loras, max_rank_potential=max_rank_potential, num_stages=num_stages,
            rank_dropout_p=rank_dropout_p
        )

    # --- Methods to pass control signals down ---
    # Removed add_stage_for_all method

    def set_current_stage_and_rank_for_all(self, stage_index, active_rank_count):
        self.attn.set_current_stage_and_rank_for_all(stage_index, active_rank_count)

    def get_params_for_stage(self, stage_index):
        return self.attn.get_params_for_stage(stage_index)
    # --- End control signal methods ---

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_norm1 = self.norm1(x)
        x_base_attn_proj, list_x_lora_attn_proj = self.attn(x_norm1)

        # --- Base Path ---
        x_base_residual = x + self.drop_path(x_base_attn_proj)
        x_base_mlp = self.mlp(self.norm2(x_base_residual))
        x_output = x_base_residual + self.drop_path(x_base_mlp)

        # --- LoRA Combined Feature Calculation ---
        lora_combined_features = []
        if self.num_loras > 0:
            for x_lora_attn_proj_i in list_x_lora_attn_proj:
                x_lora_residual_i = x + self.drop_path(x_lora_attn_proj_i)
                x_lora_mlp_i = self.mlp(self.norm2(x_lora_residual_i))
                lora_combined_feature_i = x_lora_residual_i + self.drop_path(x_lora_mlp_i)
                lora_combined_features.append(lora_combined_feature_i)

        return x_output, lora_combined_features














# --- LoRA Path Classifier ---
class LoRAPathAvgPoolClassifier(nn.Module):
    """
    对单个LoRA路径输出的CLS Token序列进行平均池化，然后进行二分类。
    """
    def __init__(self, embed_dim: int, num_classes: int = 1):
        """
        Args:
            embed_dim: ViT模型的嵌入维度 (C)。
            num_classes: 输出类别数 (1 for binary with BCEWithLogitsLoss, 2 for CrossEntropyLoss).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, lora_cls_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lora_cls_tensor: 单个LoRA路径的CLS token特征序列，
                             形状为 (B, depth, C)。

        Returns:
            分类器的logits，形状为 (B, num_classes)。
        """
        if lora_cls_tensor.dim() != 3 or lora_cls_tensor.shape[-1] != self.embed_dim:
            raise ValueError(f"Input tensor shape must be (B, depth, {self.embed_dim}), "
                             f"but got {lora_cls_tensor.shape}")
        # 平均池化: (B, depth, C) -> (B, C)
        pooled_features = torch.mean(lora_cls_tensor, dim=1)
        # 分类: (B, C) -> (B, num_classes)
        logits = self.classifier(pooled_features)
        return logits

# --- Main ViT Model ---
# 继承自timm的VisionTransformer，构建一个完整的ViT模型。
# 核心修改是用`ParallelDynamicLoRABlock`序列替换了原始的`Block`序列。
# 集成了并行、动态秩的LoRA机制，并提供了顶层控制方法来管理所有LoRA层的阶段和秩。
# 提供了`freeze_base`选项，用于冻结除LoRA组件外的所有基础模型参数。
# `forward_features`方法被重写，以收集并返回基础路径输出以及每个并行LoRA路径在所有Block中产生的特征序列。
# `forward`方法返回一个包含基础路径分类结果(`logits`)和所有LoRA路径特征(`lora_features`)的字典。
class ParallelDynamicLoRA_ViT_timm(VisionTransformer):
    """ ViT with parallel dynamic LoRA blocks. """
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
            act_layer=None, weight_init='',
            num_loras=1, max_rank_potential=8, num_stages=1, rank_dropout_p=0.0, freeze_base=True):

        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            representation_size=representation_size, distilled=distilled, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer,
            norm_layer=norm_layer, act_layer=act_layer, weight_init=weight_init
        )
        self.num_loras = num_loras
        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages # Store num_stages
        self.current_stage = -1
        self.current_stage_active_rank_count = 0

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Replace blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            ParallelDynamicLoRABlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, num_loras=num_loras, max_rank_potential=max_rank_potential,
                num_stages=num_stages, rank_dropout_p=rank_dropout_p)
            for i in range(depth)])

        # --- Initialize LoRA Path Classifiers ---
        # Create num_loras independent classifiers, one for each parallel path
        # We assume the output dimension for these classifiers might be different
        # from the main head's num_classes. Let's default to 1 for binary classification.
        # You might want to pass a specific num_classes_lora argument if needed.
        lora_classifier_output_dim = 1 # Default for binary classification logits
        self.lora_path_classifiers = nn.ModuleList(
            [LoRAPathAvgPoolClassifier(embed_dim, num_classes=lora_classifier_output_dim) for _ in range(self.num_loras)]
        )

        # Init weights
        self.init_weights(weight_init)

        # Freeze base model
        if freeze_base:
            self.freeze_base_parameters()

    def freeze_base_parameters(self):
        """ Freezes all parameters except LoRA layers. Head is also frozen. """
        for name, param in self.named_parameters():
             # Check if the parameter name indicates it's part of a LoRA component
             # This relies on the structure: blocks -> attn -> lora_q/v -> components -> lora_a/b
             is_lora_param = False
             if '.lora_q.' in name or '.lora_v.' in name:
                 # Further check if it's within lora_components_per_stage
                 if '.lora_components_per_stage.' in name and ('lora_a' in name or 'lora_b' in name):
                      is_lora_param = True

             # Freeze if it's not a LoRA param AND not part of the new LoRA path classifiers
             if not is_lora_param and not name.startswith('lora_path_classifiers.'):
                 param.requires_grad = False

        # Ensure ALL pre-initialized LoRA parameters are trainable
        for block in self.blocks:
            if isinstance(block, ParallelDynamicLoRABlock):
                # Iterate through each parallel LoRA path (for Q and V)
                for lora_layer in block.attn.lora_q:
                    # Iterate through each pre-initialized stage in the layer
                    for stage_key in lora_layer.lora_components_per_stage:
                        # Iterate through each rank-1 component in the stage's ModuleList
                        for component in lora_layer.lora_components_per_stage[stage_key]:
                            for param in component.parameters():
                                param.requires_grad = True # Make sure it's trainable
                for lora_layer in block.attn.lora_v:
                    for stage_key in lora_layer.lora_components_per_stage:
                        for component in lora_layer.lora_components_per_stage[stage_key]:
                            for param in component.parameters():
                                param.requires_grad = True
        # Ensure LoRA path classifiers are trainable (should be by default, but good practice)
        for classifier in self.lora_path_classifiers:
            for param in classifier.parameters():
                param.requires_grad = True

        # Explicitly ensure main head is frozen (if freeze_base is True)
        if self.head is not None:
            for param in self.head.parameters(): param.requires_grad = False
        if self.head_dist is not None:
             for param in self.head_dist.parameters(): param.requires_grad = False

    # --- Stage and Rank Control Methods ---
    def start_new_stage(self, stage_index, initial_active_rank=1):
        """ Initializes a new learning stage across all blocks and LoRA adapters. """
        print(f"Starting new stage {stage_index} with initial active rank {initial_active_rank}")
        self.current_stage = stage_index
        self.current_stage_active_rank_count = min(initial_active_rank, self.max_rank_potential)

        # Set the current stage and rank for all blocks, no need to add stages anymore
        for block in self.blocks:
            if isinstance(block, ParallelDynamicLoRABlock):
                # block.add_stage_for_all(stage_index) # Removed
                block.set_current_stage_and_rank_for_all(self.current_stage, self.current_stage_active_rank_count)
    def increment_active_rank(self, increment=1):
        """ Increments the active rank for the current stage across all LoRA adapters. """
        if self.current_stage < 0:
            print("Warning: Trying to increment rank before starting a stage.")
            return
        new_active_count = min(self.current_stage_active_rank_count + increment, self.max_rank_potential)
        if new_active_count > self.current_stage_active_rank_count:
            print(f"Incrementing active rank for stage {self.current_stage} from {self.current_stage_active_rank_count} to {new_active_count}")
            self.current_stage_active_rank_count = new_active_count
            for block in self.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                    block.set_current_stage_and_rank_for_all(self.current_stage, self.current_stage_active_rank_count)
        else:
             print(f"Active rank for stage {self.current_stage} already at max potential ({self.max_rank_potential}). Cannot increment.")


    def get_params_for_current_stage(self):
        """ Collects trainable parameters ONLY from the current active stage. """
        params = []
        if self.current_stage >= 0:
            for block in self.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                    params.extend(block.get_params_for_stage(self.current_stage))
        # Remove duplicates just in case
        return list(set(params))
    
    def set_inference_configuration(self, stage_index: int, rank_config: List[List[int]]):
        """
        Sets the model to use a specific LoRA configuration for inference with granular rank control per layer,
        assuming Q and V ranks are the same for a given path within a block.

        Args:
            stage_index: The index of the pre-initialized stage whose parameters to use.
            rank_config: A list where each element corresponds to a block.
                         Each element is a list of integers, where the i-th integer
                         is the desired rank for the i-th parallel LoRA path (Q and V) in that block.
                         The length of the inner list must equal `num_loras`.
                         Example for depth=12, num_loras=2:
                         [ [4, 2],  # Block 0: Path 0 Rank 4, Path 1 Rank 2
                           ...
                           [8, 4] ] # Block 11
        """
        if not (0 <= stage_index < self.num_stages):
            raise ValueError(f"Invalid stage_index {stage_index}. Must be between 0 and {self.num_stages - 1}.")

        if not isinstance(rank_config, list) or len(rank_config) != len(self.blocks):
            raise ValueError(f"rank_config must be a list of length {len(self.blocks)} (model depth).")

        print(f"Setting granular inference configuration (Q=V rank, list format) for Stage {stage_index}...")
        self.current_stage = stage_index
        # self.current_stage_active_rank_count is no longer meaningful with granular ranks

        # Propagate the granular settings down to each specific LoRA layer
        for block_idx, block in enumerate(self.blocks):
            if isinstance(block, ParallelDynamicLoRABlock):
                if not isinstance(rank_config[block_idx], list) or len(rank_config[block_idx]) != block.attn.num_loras:
                     raise ValueError(f"Element {block_idx} in rank_config must be a list of length {block.attn.num_loras} (num_loras).")

                # Iterate through parallel LoRA paths using index
                for lora_idx, rank_count in enumerate(rank_config[block_idx]):
                    if not isinstance(rank_count, int) or not (0 <= rank_count <= self.max_rank_potential):
                        raise ValueError(f"Invalid rank_count {rank_count} for block {block_idx}, path {lora_idx}. Must be an integer between 0 and {self.max_rank_potential}.")

                    # Set stage and the same rank for both Q and V LoRA
                    if lora_idx < len(block.attn.lora_q):
                         block.attn.lora_q[lora_idx].set_current_stage_and_rank(self.current_stage, rank_count)
                    if lora_idx < len(block.attn.lora_v):
                         block.attn.lora_v[lora_idx].set_current_stage_and_rank(self.current_stage, rank_count)
            # else: Block is not a ParallelDynamicLoRABlock (shouldn't happen with current init)
    # --- End Control Methods ---

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)

        collected_lora_features_per_lora = [[] for _ in range(self.num_loras)]

        for blk in self.blocks:
            x, lora_combined_features_block = blk(x)
            if self.num_loras > 0 and lora_combined_features_block: # Check if list is not empty
                for i in range(self.num_loras):
                    collected_lora_features_per_lora[i].append(lora_combined_features_block[i])

        x_base_norm = self.norm(x)

        final_lora_feature_tensors = []
        if self.num_loras > 0:
            # Check if features were collected before stacking
            if all(collected_lora_features_per_lora):
                 for features_one_lora in collected_lora_features_per_lora:
                     final_lora_feature_tensors.append(torch.stack(features_one_lora, dim=1))
            else:
                 # Handle case where no features were collected (e.g., num_loras=0 or error)
                 pass # Or maybe raise an error or return empty list based on desired behavior

        return x_base_norm, final_lora_feature_tensors

    def extract_lora_cls_tokens(self, list_of_lora_feature_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Extracts the CLS token features from the list of full LoRA feature tensors.

        Args:
            list_of_lora_feature_tensors: List[Tensor(B, depth, N, C)] from forward_features.

        Returns:
            List[Tensor(B, depth, C)] containing only CLS token features for each LoRA path.
        """
        list_of_lora_cls_token_tensors = []
        if not list_of_lora_feature_tensors:
            return []

        for feature_tensor in list_of_lora_feature_tensors:
            # feature_tensor shape: (B, depth, N, C)
            # Extract CLS token (index 0 along the N dimension)
            cls_token_tensor = feature_tensor[:, :, 0, :] # Shape: (B, depth, C)
            list_of_lora_cls_token_tensors.append(cls_token_tensor)

        return list_of_lora_cls_token_tensors

    def forward(self, x) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        # x_base_norm shape: (B, N, C)
        # list_of_lora_feature_tensors: List[Tensor(B, depth, N, C)], len=num_loras
        x_base_norm, list_of_lora_feature_tensors = self.forward_features(x)

        # Extract CLS tokens from the full LoRA features using the helper method
        # list_of_lora_cls_token_tensors: List[Tensor(B, depth, C)], len=num_loras
        list_of_lora_cls_token_tensors = self.extract_lora_cls_tokens(list_of_lora_feature_tensors)

        # --- Calculate Logits ---
        # 1. Base path logits (using the main head)
        base_logits = self.head(x_base_norm[:, 0]) # Use index 0 for CLS token

        # 2. LoRA path logits (using the dedicated classifiers)
        lora_path_logits = []
        # Only calculate logits for LoRA paths up to the current active stage
        # to simulate incremental learning prediction.
        if self.current_stage >= 0 and list_of_lora_cls_token_tensors:
            # Iterate only up to the current stage (inclusive)
            num_active_paths = self.current_stage + 1
            if len(self.lora_path_classifiers) >= num_active_paths and len(list_of_lora_cls_token_tensors) >= num_active_paths:
                for i in range(num_active_paths):
                    # Pass the i-th LoRA path's CLS token sequence to the i-th classifier
                    path_logits = self.lora_path_classifiers[i](list_of_lora_cls_token_tensors[i])
                    lora_path_logits.append(path_logits)
            else:
                 # This might happen if current_stage is somehow larger than num_loras - 1
                 # or if cls tokens weren't collected properly.
                 print(f"Warning: Cannot compute LoRA logits. current_stage={self.current_stage}, "
                       f"num_classifiers={len(self.lora_path_classifiers)}, "
                       f"num_cls_token_tensors={len(list_of_lora_cls_token_tensors)}")


        return {
            'base_logits': base_logits,
            'lora_path_logits': lora_path_logits # List of logits, one per LoRA path
        }































# --- Creator Function ---
# 创建带有并行、动态秩LoRA块的ViT模型的工厂函数。
def create_parallel_dynamic_lora_vit(
        model_name='vit_base_patch16_224', pretrained=True, num_classes=1000,
        num_loras=1, max_rank_potential=8, num_stages=1, rank_dropout_p=0.0, freeze_base=True, **kwargs):
    """ Creates a ViT model with parallel, dynamic rank LoRA blocks. """
    vit_model = create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

    model = ParallelDynamicLoRA_ViT_timm(
        img_size=vit_model.img_size,
        patch_size=vit_model.patch_size,
        in_chans=vit_model.in_chans,
        num_classes=vit_model.num_classes,
        embed_dim=vit_model.embed_dim,
        depth=len(vit_model.blocks),
        num_heads=vit_model.blocks[0].attn.num_heads,
        mlp_ratio=vit_model.blocks[0].mlp.mlp_ratio,
        qkv_bias=isinstance(vit_model.blocks[0].attn.qkv, nn.Linear) and vit_model.blocks[0].attn.qkv.bias is not None,
        drop_rate=vit_model.drop_rate,
        attn_drop_rate=vit_model.attn_drop_rate,
        drop_path_rate=vit_model.drop_path_rate,
        norm_layer=vit_model.norm.__class__,
        act_layer=vit_model.blocks[0].mlp.act.__class__,
        num_loras=num_loras,
        max_rank_potential=max_rank_potential,
        num_stages=num_stages, # Pass num_stages here
        rank_dropout_p=rank_dropout_p,
        freeze_base=freeze_base
    )

    print("Loading pretrained weights (strict=False)...")
    load_info = model.load_state_dict(vit_model.state_dict(), strict=False)
    print("Weight loading info:", load_info)

    # Explicitly freeze again after loading state_dict if needed
    if freeze_base:
        model.freeze_base_parameters()
        # Verify freezing
        print("\nVerifying parameter freezing after loading:")
        trainable_params = 0
        non_lora_trainable = False
        for name, param in model.named_parameters():
             if param.requires_grad:
                 trainable_params += param.numel()
                 # Check if any non-LoRA param is accidentally trainable
                 is_lora_param = False
                 if '.lora_q.' in name or '.lora_v.' in name:
                     if '.lora_components_per_stage.' in name and ('lora_a' in name or 'lora_b' in name):
                          is_lora_param = True
                 if not is_lora_param:
                     print(f"Warning: Non-LoRA parameter '{name}' is trainable!")
                     non_lora_trainable = True
        # Note: With pre-initialization and requires_grad=True for all LoRA,
        # trainable_params will NOT be 0 initially if freeze_base=True.
        # It will be the total number of parameters in all LoRA components across all stages.
        print(f"Total trainable parameters (LoRA components): {trainable_params}")
        if not non_lora_trainable:
             print("Freezing verification successful (only LoRA params are trainable).")
        else:
             print("Warning: Non-LoRA parameters are trainable despite freeze_base=True!")
    return model














































