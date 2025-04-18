import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TimmBlock # Rename to avoid conflict
from typing import List, Tuple

# Import the refactored Attention class
from .lora_attention import ParallelDynamicLoRAAttention

# Inherit from nn.Module, not TimmBlock
class ParallelDynamicLoRABlock(nn.Module):
    """
    Wraps a timm Block and replaces its Attention module with ParallelDynamicLoRAAttention.
    Handles feature propagation for the base path and all parallel LoRA paths using the original block's components.
    """
    def __init__(
            self,
            original_block: TimmBlock, # Accept the original timm Block
            num_loras=1,
            max_rank_potential=8,
            num_stages=1,
            rank_dropout_p=0.0):
        super().__init__() # Call nn.Module's init
        self.original_block = original_block # Store the original block
        self.num_loras = num_loras

        # Instantiate the refactored Attention, passing the original block's attention
        self.attn = ParallelDynamicLoRAAttention(
            original_attn=self.original_block.attn,
            dim=self.original_block.attn.dim, # Get dim from original attn
            num_heads=self.original_block.attn.num_heads, # Get num_heads from original attn
            num_loras=num_loras,
            max_rank_potential=max_rank_potential,
            num_stages=num_stages,
            rank_dropout_p=rank_dropout_p
        )

        # We will use the original block's mlp, norm, and drop_path directly
        # No need to redefine self.norm1, self.mlp, self.norm2, self.drop_path

    def set_current_stage_and_rank_for_all(self, stage_index, active_rank_count):
        """Passes stage and rank settings down to the internal LoRA Attention module."""
        self.attn.set_current_stage_and_rank_for_all(stage_index, active_rank_count)

    def get_params_for_stage(self, stage_index):
        """Gets active parameters for the specified stage from the internal LoRA Attention module."""
        return self.attn.get_params_for_stage(stage_index)

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass using the original block's components and the wrapped LoRA attention.
        Returns:
            - x_output: Output of the base path after passing through the entire wrapped Block.
            - lora_combined_features: List of features for each parallel LoRA path processed through the Block structure.
        """
        # Use original_block's norm1
        x_norm1 = self.original_block.norm1(x)

        # Call the refactored self.attn
        x_base_attn_proj, list_x_lora_attn_proj = self.attn(x_norm1)

        # --- Base Path ---
        # Use original_block's drop_path, norm2, mlp
        x_base_residual = x + self.original_block.drop_path(x_base_attn_proj)
        x_base_mlp = self.original_block.mlp(self.original_block.norm2(x_base_residual))
        x_output = x_base_residual + self.original_block.drop_path(x_base_mlp)

        # --- LoRA Paths ---
        lora_combined_features = []
        if self.num_loras > 0 and list_x_lora_attn_proj:
            for x_lora_attn_proj_i in list_x_lora_attn_proj:
                # Use original_block's components for LoRA paths as well
                # Note: The residual connection adds back the original input 'x'
                x_lora_residual_i = x + self.original_block.drop_path(x_lora_attn_proj_i)
                x_lora_mlp_i = self.original_block.mlp(self.original_block.norm2(x_lora_residual_i))
                lora_combined_feature_i = x_lora_residual_i + self.original_block.drop_path(x_lora_mlp_i)
                lora_combined_features.append(lora_combined_feature_i)

        return x_output, lora_combined_features