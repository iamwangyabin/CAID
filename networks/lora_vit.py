import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block as TimmBlock
from functools import partial
from typing import List, Dict, Tuple, Optional

from .lora_block import ParallelDynamicLoRABlock
from .lora_classifier import LoRAPathClassifier
from utils.registry import MODELS

@MODELS.register_module()
class LoRADetector(nn.Module):
    def __init__(self,
                 backbone_name='vit_base_patch16_224',
                 backbone_pretrained=True,
                 num_stages=5,
                 max_rank_potential=8,
                 rank_dropout_p=0.0,
                 num_classes=1,
                 **kwargs):
        super().__init__()

        self.max_rank_potential = max_rank_potential
        self.num_stages = num_stages
        self.num_classes = num_classes

        self.base_model = timm.create_model(
            backbone_name,
            pretrained=backbone_pretrained,
            **kwargs
        )
        if hasattr(self.base_model, 'embed_dim'):
            self.embed_dim = self.base_model.embed_dim
        elif hasattr(self.base_model, 'patch_embed') and hasattr(self.base_model.patch_embed, 'proj') and hasattr(self.base_model.patch_embed.proj, 'out_channels'):
            self.embed_dim = self.base_model.patch_embed.proj.out_channels
        elif hasattr(self.base_model, 'norm') and hasattr(self.base_model.norm, 'normalized_shape'):
            norm_shape = self.base_model.norm.normalized_shape
            self.embed_dim = norm_shape[0] if isinstance(norm_shape, (list, tuple)) else norm_shape
        elif hasattr(self.base_model, 'blocks') and len(self.base_model.blocks) > 0 and hasattr(self.base_model.blocks[0].attn, 'dim'):
            self.embed_dim = self.base_model.blocks[0].attn.dim
        elif hasattr(self.base_model, 'head') and hasattr(self.base_model.head, 'in_features'):
            self.embed_dim = self.base_model.head.in_features
        else:
            raise ValueError("Cannot automatically determine embed_dim for the base model. Please check model structure or provide it explicitly.")

        original_blocks = list(self.base_model.blocks)
        new_blocks = []
        for i, original_block in enumerate(original_blocks):
            if isinstance(original_block, TimmBlock):
                lora_block = ParallelDynamicLoRABlock(
                    original_block=original_block,
                    embed_dim=self.embed_dim,
                    max_rank_potential=max_rank_potential,
                    num_stages=num_stages,
                    rank_dropout_p=rank_dropout_p
                )
                new_blocks.append(lora_block)
            else:
                new_blocks.append(original_block)

        if isinstance(self.base_model.blocks, nn.Sequential):
            self.base_model.blocks = nn.Sequential(*new_blocks)
        elif isinstance(self.base_model.blocks, nn.ModuleList):
            self.base_model.blocks = nn.ModuleList(new_blocks)
        else:
            self.base_model.blocks = nn.ModuleList(new_blocks)

        self.stage_classifiers = nn.ModuleDict()
        for s in range(self.num_stages):
            stage_key = f'stage_{s}'
            self.stage_classifiers[stage_key] = LoRAPathClassifier(self.embed_dim, self.num_classes)

        self.current_stage = -1
        self.current_stage_active_rank_count = 0

    def start_new_stage(self, stage_index: int, initial_active_rank: int = 1):
        if not (0 <= stage_index < self.num_stages):
            raise ValueError(f"无效的阶段索引 {stage_index}。必须在 0 到 {self.num_stages - 1} 之间。")
        self.current_stage = stage_index
        self.current_stage_active_rank_count = min(initial_active_rank, self.max_rank_potential)
        print(f"开始新阶段 {self.current_stage}，初始激活秩: {self.current_stage_active_rank_count}")

        for block in self.base_model.blocks:
            if isinstance(block, ParallelDynamicLoRABlock):
                block.set_current_stage_and_rank(self.current_stage, self.current_stage_active_rank_count)

    def increment_active_rank(self, increment: int = 1):
        if self.current_stage < 0:
            print("警告: 无法增加秩，没有活动的阶段。")
            return
        new_active_count = min(self.current_stage_active_rank_count + increment, self.max_rank_potential)

        if new_active_count > self.current_stage_active_rank_count:
            self.current_stage_active_rank_count = new_active_count
            print(f"阶段 {self.current_stage}: 正在将激活秩增加到 {self.current_stage_active_rank_count}")
            for block in self.base_model.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                    block.set_current_stage_and_rank(self.current_stage, self.current_stage_active_rank_count)
        else:
            print(f"阶段 {self.current_stage}: 秩已达到最大潜力 ({self.max_rank_potential})。")

    def get_params_for_current_stage(self) -> List[nn.Parameter]:
        params = []
        if self.current_stage >= 0:
            for block in self.base_model.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                    params.extend(block.get_params_for_stage(self.current_stage))
            stage_key = f'stage_{self.current_stage}'
            if stage_key in self.stage_classifiers:
                print(f"为阶段 {self.current_stage} 添加分类器参数: {stage_key}")
                params.extend(list(self.stage_classifiers[stage_key].parameters()))
            else:
                print(f"警告: 在 get_params_for_current_stage 中未找到阶段 {stage_key} 的分类器。")
        return list(dict.fromkeys(params))

    def set_ranks_for_stage_blocks(self, stage_index: int, ranks: List[int]):
        if not (0 <= stage_index < self.num_stages):
            raise ValueError(f"无效的阶段索引 {stage_index}。必须在 0 和 {self.num_stages - 1} 之间。")

        lora_blocks = [block for block in self.base_model.blocks if isinstance(block, ParallelDynamicLoRABlock)]
        num_lora_blocks = len(lora_blocks)

        if len(ranks) != num_lora_blocks:
            raise ValueError(f"提供的 ranks 列表长度 ({len(ranks)}) 与 LoRA Block 的数量 ({num_lora_blocks}) 不匹配。")

        print(f"正在为阶段 {stage_index} 的 {num_lora_blocks} 个 LoRA Block 设置秩: {ranks}")
        for i, block in enumerate(lora_blocks):
            rank = ranks[i]
            if not (0 <= rank <= self.max_rank_potential):
                raise ValueError(f"为 Block {i} 提供的秩 {rank} 无效。必须在 0 和 {self.max_rank_potential} 之间。")
            block.set_rank_for_stage(stage_index, rank)
 
    def set_trainable_stage(self, stage_idx: int):
        """
        Freezes all model parameters and then unfreezes only the parameters
        belonging to the specified stage (LoRA blocks and classifier).
        """
        if not (0 <= stage_idx < self.num_stages):
            raise ValueError(f"Invalid stage index {stage_idx}. Must be between 0 and {self.num_stages - 1}.")

        print(f"Setting trainable parameters for stage {stage_idx}...")

        # 1. Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False

        # 2. Unfreeze LoRA parameters for the target stage
        num_lora_params_unfrozen = 0
        for block in self.base_model.blocks:
            if isinstance(block, ParallelDynamicLoRABlock):
                # Assuming get_params_for_stage exists and returns parameters
                try:
                    stage_params = block.get_params_for_stage(stage_idx)
                    for param in stage_params:
                        param.requires_grad = True
                        num_lora_params_unfrozen += param.numel()
                except Exception as e:
                     print(f"Warning: Could not get or set params for stage {stage_idx} in block {type(block)}. Error: {e}")


        # 3. Unfreeze the classifier for the target stage
        stage_key = f'stage_{stage_idx}'
        num_classifier_params_unfrozen = 0
        if stage_key in self.stage_classifiers:
            for param in self.stage_classifiers[stage_key].parameters():
                param.requires_grad = True
                num_classifier_params_unfrozen += param.numel()
            print(f"Unfroze {num_classifier_params_unfrozen} parameters in classifier {stage_key}.")
        else:
            print(f"Warning: Classifier for stage {stage_idx} (key: {stage_key}) not found. Cannot unfreeze.")

        total_unfrozen = num_lora_params_unfrozen + num_classifier_params_unfrozen
        print(f"Total parameters unfrozen for stage {stage_idx}: {total_unfrozen} ({num_lora_params_unfrozen} LoRA + {num_classifier_params_unfrozen} Classifier)")

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]:
        x = self.base_model.patch_embed(x)
        if hasattr(self.base_model, 'cls_token') and self.base_model.cls_token is not None:
            cls_tokens = self.base_model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if hasattr(self.base_model, 'pos_embed') and self.base_model.pos_embed is not None:
            x = x + self.base_model.pos_embed
        x = self.base_model.pos_drop(x)

        num_blocks = len(self.base_model.blocks)
        outputs_by_stage: Dict[str, List[Optional[torch.Tensor]]] = \
            {f'stage_{s}': [None] * num_blocks for s in range(self.num_stages)}

        current_input = x
        for block_idx, blk in enumerate(self.base_model.blocks):
            if isinstance(blk, ParallelDynamicLoRABlock):
                x_base_output, block_stage_outputs_dict = blk(current_input)
                current_input = x_base_output
                for stage_key, stage_block_output in block_stage_outputs_dict.items():
                    if stage_key in outputs_by_stage:
                        outputs_by_stage[stage_key][block_idx] = stage_block_output
                    else:
                        print(f"警告: 在 Block {block_idx} 中遇到意外的阶段键 '{stage_key}'。已忽略。")
            else:
                current_input = blk(current_input)

        x_base_norm = self.base_model.norm(current_input)

        return x_base_norm, outputs_by_stage

    def extract_lora_cls_tokens(self, outputs_by_stage: Dict[str, List[Optional[torch.Tensor]]]) -> Dict[str, Optional[List[torch.Tensor]]]:
        if not hasattr(self.base_model, 'cls_token') or self.base_model.cls_token is None:
            return {stage_key: None for stage_key in outputs_by_stage}

        cls_token_lists_by_stage: Dict[str, Optional[List[torch.Tensor]]] = {}
        for stage_key, block_outputs_list in outputs_by_stage.items():
            stage_cls_tokens_list = []
            for block_output in block_outputs_list:
                if block_output is not None and block_output.ndim > 1:
                    stage_cls_tokens_list.append(block_output[:, 0])

            if not stage_cls_tokens_list:
                cls_token_lists_by_stage[stage_key] = None
            else:
                cls_token_lists_by_stage[stage_key] = stage_cls_tokens_list

        return cls_token_lists_by_stage

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        x_base_norm, outputs_by_stage = self.forward_features(x)

        cls_token_lists_by_stage = self.extract_lora_cls_tokens(outputs_by_stage)

        stage_logits = {}
        for stage_idx, (stage_key, token_list) in enumerate(cls_token_lists_by_stage.items()):
            stacked_tokens = torch.stack(token_list, dim=1)
            logits = self.stage_classifiers[stage_key](stacked_tokens)
            stage_logits[stage_key] = logits


        return {
            "current_logits": logits,
            "stage_logits": stage_logits
        }
