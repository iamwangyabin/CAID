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


    def start_new_stage(self, stage_index, initial_active_rank=1):
        if not (0 <= stage_index < self.num_stages):
             raise ValueError(f"Invalid stage_index {stage_index}. Must be between 0 and {self.num_stages - 1}.")
        self.current_stage = stage_index
        self.current_stage_active_rank_count = min(initial_active_rank, self.max_rank_potential)

        for block in self.base_model.blocks:
            if isinstance(block, ParallelDynamicLoRABlock):
                block.set_current_stage_and_rank(self.current_stage, self.current_stage_active_rank_count)

    def increment_active_rank(self, increment=1):
        if self.current_stage < 0:
            print("Warning: Cannot increment rank, no stage active.")
            return
        new_active_count = min(self.current_stage_active_rank_count + increment, self.max_rank_potential)
        if new_active_count > self.current_stage_active_rank_count:
            self.current_stage_active_rank_count = new_active_count
            print(f"Stage {self.current_stage}: Incrementing active rank to {self.current_stage_active_rank_count}")
            for block in self.base_model.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                     block.set_current_stage_and_rank(self.current_stage, self.current_stage_active_rank_count)
        else:
             print(f"Stage {self.current_stage}: Rank already at max potential ({self.max_rank_potential}).")


    def get_params_for_current_stage(self):
        params = []
        if self.current_stage >= 0:
            for block in self.base_model.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                    params.extend(block.get_params_for_stage(self.current_stage))
        return list(dict.fromkeys(params))

    def set_ranks_for_stage_blocks(self, stage_index: int, ranks: List[int]):
        """
        为指定阶段内的每个LoRA Block设置不同的rank。

        Args:
            stage_index (int): 要修改的阶段索引。必须在 0 到 num_stages - 1 之间。
            ranks (List[int]): 一个整数列表，列表长度必须等于网络中LoRA Block的数量。
                               列表中的每个元素指定对应Block在此阶段的rank。
                               每个rank值必须在 0 到 max_rank_potential 之间。
        """
        if not (0 <= stage_index < self.num_stages):
            raise ValueError(f"Invalid stage_index {stage_index}. Must be between 0 and {self.num_stages - 1}.")

        lora_blocks = [block for block in self.base_model.blocks if isinstance(block, ParallelDynamicLoRABlock)]
        num_lora_blocks = len(lora_blocks)

        if len(ranks) != num_lora_blocks:
            raise ValueError(f"Provided ranks list length ({len(ranks)}) does not match the number of LoRA Blocks ({num_lora_blocks}).")

        print(f"Setting Ranks for {num_lora_blocks} LoRA Blocks in Stage {stage_index}: {ranks}")
        for i, block in enumerate(lora_blocks):
            rank = ranks[i]
            if not (0 <= rank <= self.max_rank_potential):
                raise ValueError(f"Invalid rank {rank} for Block {i}. Must be between 0 and {self.max_rank_potential}.")
            block.set_rank_for_stage(stage_index, rank)


    def forward_features(self, x) -> Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]:
        x = self.base_model.patch_embed(x)
        if hasattr(self.base_model, 'cls_token') and self.base_model.cls_token is not None:
             cls_tokens = self.base_model.cls_token.expand(x.shape[0], -1, -1)
             x = torch.cat((cls_tokens, x), dim=1)
        if hasattr(self.base_model, 'pos_embed') and self.base_model.pos_embed is not None:
             x = x + self.base_model.pos_embed
        x = self.base_model.pos_drop(x)

        num_blocks = len(self.base_model.blocks)
        outputs_by_stage = {f'stage_{s}': [None] * num_blocks for s in range(self.num_stages)}

        current_input = x
        for block_idx, blk in enumerate(self.base_model.blocks):
            if isinstance(blk, ParallelDynamicLoRABlock):
                x_base_output, block_stage_outputs_dict = blk(current_input)
                current_input = x_base_output
                for stage_key, stage_block_output in block_stage_outputs_dict.items():
                    if stage_key in outputs_by_stage:
                         outputs_by_stage[stage_key][block_idx] = stage_block_output
                    else:
                        print(f"Warning: Encountered unexpected stage key '{stage_key}' in block {block_idx}. Ignoring.")
            else:
                current_input = blk(current_input)
                # No LoRA output for this block, None placeholders are already initialized
                current_input = blk(current_input)

        x_base_norm = self.base_model.norm(current_input)

        return x_base_norm, outputs_by_stage


    def extract_lora_cls_tokens(self, outputs_by_stage: Dict[str, List[Optional[torch.Tensor]]]) -> Dict[str, Optional[List[torch.Tensor]]]:
        """ Extracts a list of CLS tokens for each stage separately. """
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

    def forward(self, x) -> Dict:
        """
        Performs forward pass, extracting base features and stage-wise LoRA features,
        and classifies the LoRA features for each stage.
        """
        x_base_norm, outputs_by_stage = self.forward_features(x)

        final_base_cls = None
        if hasattr(self.base_model, 'cls_token') and self.base_model.cls_token is not None:
            if x_base_norm.ndim > 1:
                final_base_cls = x_base_norm[:, 0]
            else:
                print(f"Warning: Cannot extract base CLS token from x_base_norm with shape {x_base_norm.shape}.")

        cls_token_lists_by_stage = self.extract_lora_cls_tokens(outputs_by_stage)

        stage_logits = {}
        for stage_key, token_list in cls_token_lists_by_stage.items():
            if token_list is not None and len(token_list) > 0:
                 try:
                     first_shape = token_list[0].shape
                     if not all(t.shape == first_shape for t in token_list):
                          raise ValueError(f"Inconsistent token shapes in list for {stage_key}. Shapes: {[t.shape for t in token_list]}")

                     stacked_tokens = torch.stack(token_list, dim=1)

                     if stage_key in self.stage_classifiers:
                         logits = self.stage_classifiers[stage_key](stacked_tokens)
                         stage_logits[stage_key] = logits
                     else:
                          stage_logits[stage_key] = None

                 except (RuntimeError, ValueError) as e:
                     print(f"Error processing or stacking tokens for {stage_key}: {e}")
                     if 'token_list' in locals() and isinstance(token_list, list):
                         print(f"Token list shapes: {[t.shape for t in token_list if hasattr(t, 'shape')]}")
                     stage_logits[stage_key] = None

            else:
                stage_logits[stage_key] = None

        return {
            "base_cls_token": final_base_cls,
            "stage_logits": stage_logits
        }
