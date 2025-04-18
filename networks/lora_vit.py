import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block as TimmBlock
from functools import partial
from typing import List, Dict, Tuple, Optional

from .lora_block import ParallelDynamicLoRABlock
from .lora_classifier import LoRAPathAvgPoolClassifier
from utils.registry import MODELS 


@MODELS.register_module()
class LoRADetector(nn.Module):
    def __init__(self,
                 backbone_name='vit_base_patch16_224',
                 backbone_pretrained=True,
                 backbone_num_classes=0,
                 num_loras=1,
                 max_rank_potential=8,
                 num_stages=1,
                 rank_dropout_p=0.0,
                 freeze_backbone_base=True,
                 lora_classifier_output_dim=1, # Output dim for LoRA path classifiers (if used)
                 detector_head_output_dim=4 + 1,
                 **kwargs):
        super().__init__()

        print(f"Creating base model: {backbone_name} (pretrained={backbone_pretrained})")
        self.base_model = timm.create_model(
            backbone_name,
            pretrained=backbone_pretrained,
            num_classes=backbone_num_classes,
            **kwargs
        )
        print("Base model created.")

        print(f"Replacing {len(self.base_model.blocks)} blocks with ParallelDynamicLoRABlock...")
        original_blocks = list(self.base_model.blocks)
        new_blocks = []
        for i, original_block in enumerate(original_blocks):
            if not isinstance(original_block, TimmBlock):
                 print(f"Warning: Block {i} is not a TimmBlock, skipping replacement.")
                 new_blocks.append(original_block)
                 continue

            lora_block = ParallelDynamicLoRABlock(
                original_block=original_block,
                num_loras=num_loras,
                max_rank_potential=max_rank_potential,
                num_stages=num_stages,
                rank_dropout_p=rank_dropout_p
            )
            new_blocks.append(lora_block)

        if isinstance(self.base_model.blocks, nn.Sequential):
             self.base_model.blocks = nn.Sequential(*new_blocks)
        elif isinstance(self.base_model.blocks, nn.ModuleList):
             self.base_model.blocks = nn.ModuleList(new_blocks)
        else:
             print("Warning: Unknown block container type. Replacing attribute directly.")
             self.base_model.blocks = nn.ModuleList(new_blocks)
        print("Blocks replaced.")

        self.num_loras = num_loras
        self.max_rank_potential = 0
        self.num_stages = 0
        if self.num_loras > 0 and isinstance(self.base_model.blocks[0], ParallelDynamicLoRABlock):
             self.max_rank_potential = self.base_model.blocks[0].attn.max_rank_potential
             self.num_stages = self.base_model.blocks[0].attn.num_stages

        self.current_stage = -1
        self.current_stage_active_rank_count = 0


        self.lora_path_classifiers = nn.ModuleList(
            [LoRAPathAvgPoolClassifier(self.base_model.embed_dim, num_classes=lora_classifier_output_dim)
             for _ in range(self.num_loras)]
        ) if self.num_loras > 0 else nn.ModuleList()


        # Define a simple detection head using CLS tokens
        # This is a simplified example. A real detector needs spatial features.
        # We'll concatenate the base CLS token and the aggregated LoRA CLS token (if available)
        # Or you could process each LoRA CLS token separately and combine results.
        # For simplicity, let's use the base_cls_token and the list of lora_cls_tokens returned by the modified forward.

        # Calculate input dimension for the detection head
        # Base CLS token dim + (Num LoRAs * LoRA CLS token dim)
        # Assuming base_cls_token and lora_cls_tokens have the same dimension (embed_dim)
        detector_head_input_dim = self.base_model.embed_dim
        if self.num_loras > 0:
             detector_head_input_dim += self.num_loras * self.base_model.embed_dim

        # Simple linear detection head
        self.detector_head = nn.Linear(detector_head_input_dim, detector_head_output_dim)

        # 4. Freeze parameters if requested
        if freeze_backbone_base:
            print("Freezing base parameters...")
            self.freeze_base_parameters()
            print("Base parameters frozen.")

            trainable_params = 0
            total_params = 0
            trainable_names = []
            for name, param in self.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    trainable_names.append(name)
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")


    def freeze_base_parameters(self):
        """Freezes non-LoRA parameters in the base model and the LoRA classifiers."""
        for name, param in self.base_model.named_parameters():
            is_lora_param = False
            if '.attn.lora_q.' in name or '.attn.lora_v.' in name:
                 if '.lora_components_per_stage.' in name and ('lora_a' in name or 'lora_b' in name):
                      is_lora_param = True

            if not is_lora_param:
                param.requires_grad = False
            else:
                 param.requires_grad = True

        # Freeze LoRA classifiers initially (or based on training strategy)
        # Typically, these are trained along with LoRA weights.
        # If they should be frozen, uncomment below:
        # for param in self.lora_path_classifiers.parameters():
        #     param.requires_grad = False

        # Explicitly ensure base model components (patch_embed, cls_token, etc.) are frozen
        # This might overlap with the loop above but ensures critical parts are frozen.
        components_to_freeze = [
            self.base_model.patch_embed,
            self.base_model.cls_token,
            self.base_model.pos_embed,
            self.base_model.norm,
            self.base_model.head,
            getattr(self.base_model, 'head_dist', None)
        ]
        for component in components_to_freeze:
            if component is not None:
                if isinstance(component, nn.Parameter):
                     component.requires_grad = False
                else:
                     for param in component.parameters():
                          param.requires_grad = False

        # Ensure the detector head is trainable
        for param in self.detector_head.parameters():
             param.requires_grad = True


    def start_new_stage(self, stage_index, initial_active_rank=1):
        if not (0 <= stage_index < self.num_stages):
             raise ValueError(f"Invalid stage_index {stage_index}. Must be between 0 and {self.num_stages - 1}.")
        self.current_stage = stage_index
        self.current_stage_active_rank_count = min(initial_active_rank, self.max_rank_potential)

        for block in self.base_model.blocks:
            if isinstance(block, ParallelDynamicLoRABlock):
                block.set_current_stage_and_rank_for_all(self.current_stage, self.current_stage_active_rank_count)

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
                    block.set_current_stage_and_rank_for_all(self.current_stage, self.current_stage_active_rank_count)
        else:
             print(f"Stage {self.current_stage}: Rank already at max potential ({self.max_rank_potential}).")


    def get_params_for_current_stage(self):
        params = []
        if self.current_stage >= 0:
            for block in self.base_model.blocks:
                if isinstance(block, ParallelDynamicLoRABlock):
                    params.extend(block.get_params_for_stage(self.current_stage))
        for classifier in self.lora_path_classifiers:
             params.extend(classifier.parameters())

        return list(dict.fromkeys(params))

    def set_inference_configuration(self, stage_index: int, rank_config: List[List[int]]):
         if not (0 <= stage_index < self.num_stages):
             raise ValueError(f"Invalid stage_index {stage_index}. Must be between 0 and {self.num_stages - 1}.")

         if not isinstance(rank_config, list) or len(rank_config) != len(self.base_model.blocks):
             raise ValueError(f"rank_config must be a list of length {len(self.base_model.blocks)} (model depth).")

         self.current_stage = stage_index

         for block_idx, block in enumerate(self.base_model.blocks):
             if isinstance(block, ParallelDynamicLoRABlock):
                 block_rank_cfg = rank_config[block_idx]
                 if not isinstance(block_rank_cfg, list) or len(block_rank_cfg) != block.attn.num_loras:
                      raise ValueError(f"Element {block_idx} in rank_config must be a list of length {block.attn.num_loras} (num_loras).")

                 if hasattr(block.attn, 'set_inference_rank_config'):
                      block.attn.set_inference_rank_config(stage_index, block_rank_cfg)
                 else:
                      print(f"Warning: ParallelDynamicLoRAAttention does not have set_inference_rank_config method.")


    def forward_features(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.base_model.patch_embed(x)
        if self.base_model.cls_token is not None:
             cls_tokens = self.base_model.cls_token.expand(x.shape[0], -1, -1)
             x = torch.cat((cls_tokens, x), dim=1)
        x = self.base_model.pos_drop(x + self.base_model.pos_embed)

        collected_lora_features_per_lora = [[] for _ in range(self.num_loras)]

        for blk in self.base_model.blocks:
            if isinstance(blk, ParallelDynamicLoRABlock):
                x, lora_combined_features_block = blk(x)
                if self.num_loras > 0 and lora_combined_features_block:
                    for i in range(min(self.num_loras, len(lora_combined_features_block))):
                         collected_lora_features_per_lora[i].append(lora_combined_features_block[i])
            else:
                x = blk(x)

        x_base_norm = self.base_model.norm(x)

        final_lora_feature_tensors = []
        if self.num_loras > 0:
             if all(len(p) == len(self.base_model.blocks) for p in collected_lora_features_per_lora):
                  for features_one_lora in collected_lora_features_per_lora:
                      final_lora_feature_tensors.append(torch.stack(features_one_lora, dim=1))
             else:
                  pass

        return x_base_norm, final_lora_feature_tensors

    def extract_lora_cls_tokens(self, list_of_lora_feature_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extracts the CLS token feature from the stacked block features for each LoRA path."""
        list_of_lora_cls_token_tensors = []
        if not list_of_lora_feature_tensors or self.base_model.cls_token is None:
            return []

        for feature_tensor in list_of_lora_feature_tensors:
            cls_token_tensor = feature_tensor[:, -1, 0, :]
            list_of_lora_cls_token_tensors.append(cls_token_tensor)

        return list_of_lora_cls_token_tensors


    def forward(self, x) -> torch.Tensor:
        x_base_norm, list_of_lora_feature_tensors = self.forward_features(x)

        base_cls_token = None
        if self.base_model.cls_token is not None:
            base_cls_token = x_base_norm[:, 0]

        list_of_lora_cls_token_tensors = self.extract_lora_cls_tokens(list_of_lora_feature_tensors)

        features_for_head = []
        if base_cls_token is not None:
             features_for_head.append(base_cls_token)

        if list_of_lora_cls_token_tensors:
             stacked_lora_cls = torch.stack(list_of_lora_cls_token_tensors, dim=1)
             flattened_lora_cls = stacked_lora_cls.view(stacked_lora_cls.size(0), -1)
             features_for_head.append(flattened_lora_cls)

        if not features_for_head:
             if hasattr(self.base_model, 'global_pool') and self.base_model.global_pool == 'avg':
                  base_pooled = x_base_norm.mean(dim=1)
                  features_for_head.append(base_pooled)
                  raise NotImplementedError("Detector head requires CLS tokens or a different pooling strategy for models without CLS.")
             else:
                  raise ValueError("No CLS tokens or suitable features available for the detection head.")


        combined_features = torch.cat(features_for_head, dim=1)

        detection_output = self.detector_head(combined_features)

        return detection_output

if __name__ == '__main__':
    print("Testing LoRADetector creation...")
    detector = LoRADetector(
        backbone_name='vit_tiny_patch16_224',
        backbone_pretrained=True,
        num_loras=2,
        max_rank_potential=4,
        num_stages=2,
        freeze_backbone_base=True,
        lora_classifier_output_dim=1,
        detector_head_output_dim=5
    )
    print("LoRADetector created successfully.")

    print("Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    detector.eval()
    with torch.no_grad():
         detector.start_new_stage(stage_index=0, initial_active_rank=2)
         detection_output = detector(dummy_input)

    print("Forward pass successful.")
    print("Detection output shape:", detection_output.shape)

    print("\nTesting parameter freezing...")
    for name, param in detector.named_parameters():
         is_lora_component = '.attn.lora_q.' in name or '.attn.lora_v.' in name
         if is_lora_component:
              is_lora_component = '.lora_components_per_stage.' in name and ('lora_a' in name or 'lora_b' in name)

         is_lora_classifier = name.startswith("lora_path_classifiers")
         is_detector_head = name.startswith("detector_head")

         if is_lora_component or is_lora_classifier or is_detector_head:
              if not param.requires_grad:
                   print(f"Error: Trainable parameter frozen: {name}")
         elif param.requires_grad:
              print(f"Error: Base parameter not frozen: {name}")
    print("Parameter freezing check complete (check output for errors).")
