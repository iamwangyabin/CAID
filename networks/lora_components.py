import torch
import torch.nn as nn
import math

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
    def __init__(self, in_features, out_features, max_rank_potential, rank_dropout_p=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank_potential = max_rank_potential
        self.rank_dropout_p = rank_dropout_p
        if not (0.0 <= self.rank_dropout_p <= 1.0):
            raise ValueError(f"rank_dropout_p must be between 0 and 1, but got {rank_dropout_p}")

        self.lora_components = nn.ModuleList()
        for _ in range(self.max_rank_potential):
            self.lora_components.append(Rank1LoRAComponent(self.in_features, self.out_features))

        self.active_rank_count = 0

    def set_active_rank(self, active_rank_count):
        new_rank = min(int(active_rank_count), self.max_rank_potential)
        if new_rank < 0:
             print(f"Warning: Attempted to set negative active rank ({active_rank_count}). Setting to 0.")
             new_rank = 0
        self.active_rank_count = new_rank

    def forward(self, x):
        lora_delta = 0.0

        target_rank = self.active_rank_count # Use internal state
        effective_rank = target_rank

        if self.training and self.rank_dropout_p > 0 and target_rank > 0:
            if torch.rand(1).item() < self.rank_dropout_p:
                effective_rank = torch.randint(0, target_rank, (1,)).item()

        if effective_rank > 0:
            for i in range(min(effective_rank, len(self.lora_components))):
                lora_delta = lora_delta + self.lora_components[i](x)

        return lora_delta


    def get_active_params(self):
        params = []
        active_rank = self.active_rank_count
        for i in range(min(active_rank, len(self.lora_components))):
             params.extend(self.lora_components[i].parameters())
        return params