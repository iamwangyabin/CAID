from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair


class E3ConvBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: str | int,
        activation: str,
    ) -> None:
        super().__init__()
        act = str(activation).lower()
        if act not in {"tanh", "relu"}:
            raise ValueError(f"Unsupported activation for E3 MISLNet: {activation!r}")
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = nn.Tanh() if act == "tanh" else nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.act(self.bn(self.conv(x))))


class E3DenseBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, activation: str) -> None:
        super().__init__()
        act = str(activation).lower()
        if act not in {"tanh", "relu"}:
            raise ValueError(f"Unsupported activation for E3 MISLNet: {activation!r}")
        self.fc = nn.Linear(in_chans, out_chans)
        self.act = nn.Tanh() if act == "tanh" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class MISLNetBackbone(nn.Module):
    """Official E3/MISLNet feature extractor.

    This ports the released `models/mislnet.py` backbone path and exposes the
    200-D dense embedding used by E3 experts.
    """

    arch = {
        256: {
            "conv1": (-1, 96, 7, 2, "valid", "tanh"),
            "conv2": (96, 64, 5, 1, "same", "tanh"),
            "conv3": (64, 64, 5, 1, "same", "tanh"),
            "conv4": (64, 128, 1, 1, "same", "tanh"),
            "fc1": (6 * 6 * 128, 200, "relu"),
            "fc2": (200, 200, "relu"),
        },
        128: {
            "conv1": (-1, 96, 7, 2, "valid", "tanh"),
            "conv2": (96, 64, 5, 1, "same", "tanh"),
            "conv3": (64, 64, 5, 1, "same", "tanh"),
            "conv4": (64, 128, 1, 1, "same", "tanh"),
            "fc1": (2 * 2 * 128, 200, "tanh"),
            "fc2": (200, 200, "tanh"),
        },
        96: {
            "conv1": (-1, 96, 7, 2, "valid", "tanh"),
            "conv2": (96, 64, 5, 1, "same", "tanh"),
            "conv3": (64, 64, 5, 1, "same", "tanh"),
            "conv4": (64, 128, 1, 1, "same", "tanh"),
            "fc1": (8 * 4 * 64, 200, "tanh"),
            "fc2": (200, 200, "tanh"),
        },
        64: {
            "conv1": (-1, 96, 7, 2, "valid", "tanh"),
            "conv2": (96, 64, 5, 1, "same", "tanh"),
            "conv3": (64, 64, 5, 1, "same", "tanh"),
            "conv4": (64, 128, 1, 1, "same", "tanh"),
            "fc1": (2 * 4 * 64, 200, "tanh"),
            "fc2": (200, 200, "tanh"),
        },
    }

    def __init__(
        self,
        patch_size: int = 256,
        num_filters: int = 6,
        constrained_conv: bool = True,
        save_features: bool = False,
    ) -> None:
        super().__init__()
        if patch_size not in self.arch:
            raise KeyError(f"Unsupported MISLNet patch_size={patch_size}; expected one of {sorted(self.arch)}")
        self.out_dim = 200
        self.patch_size = int(patch_size)
        self.num_filters = int(num_filters)
        self.constrained_conv = bool(constrained_conv)
        self.save_features = bool(save_features)
        self.features: torch.Tensor | None = None
        self.dense: torch.Tensor | None = None
        chosen_arch = self.arch[self.patch_size]

        self.weights_cstr = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.num_filters, 3, 5, 5)))
        self.conv_blocks = nn.Sequential(
            *[
                E3ConvBlock(
                    in_chans=self.num_filters if chosen_arch[f"conv{i}"][0] == -1 else chosen_arch[f"conv{i}"][0],
                    out_chans=chosen_arch[f"conv{i}"][1],
                    kernel_size=chosen_arch[f"conv{i}"][2],
                    stride=chosen_arch[f"conv{i}"][3],
                    padding=chosen_arch[f"conv{i}"][4],
                    activation=chosen_arch[f"conv{i}"][5],
                )
                for i in [1, 2, 3, 4]
            ]
        )
        self.fc_blocks = nn.Sequential(
            *[
                E3DenseBlock(
                    in_chans=chosen_arch[f"fc{i}"][0],
                    out_chans=chosen_arch[f"fc{i}"][1],
                    activation=chosen_arch[f"fc{i}"][2],
                )
                for i in [1, 2]
            ]
        )
        self._init_weights()
        self._constrain_conv()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    @torch.no_grad()
    def _constrain_conv(self) -> None:
        w = self.weights_cstr * 10000.0
        w[:, :, 2, 2] = 0
        w = w.reshape(self.num_filters, 3, 1, 25)
        w = w / w.sum(3, keepdim=True)
        w = w.reshape(self.num_filters, 3, 5, 5)
        w[:, :, 2, 2] = -1
        self.weights_cstr.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.constrained_conv:
            self._constrain_conv()
        constr_conv = F.conv2d(x.float(), self.weights_cstr, padding="valid")
        constr_conv = F.pad(constr_conv, (2, 3, 2, 3))
        if self.save_features:
            self.features = constr_conv[:, 0, :, :].detach()
        conv_out = self.conv_blocks(constr_conv)
        conv_out = conv_out.permute(0, 2, 3, 1).flatten(1, -1)
        dense_out = self.fc_blocks(conv_out)
        self.dense = dense_out.detach()
        return dense_out


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] | None = None,
        bias: bool | Sequence[bool] = True,
        drop_prob: float | Sequence[float] = 0.0,
        use_conv: bool = False,
        drop: float | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        out_features = int(out_features or in_features)
        hidden_features = int(hidden_features or in_features)
        bias_pair = _pair(bias)
        drop_values = drop if drop is not None else drop_prob
        drop_pair = _pair(drop_values)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias_pair[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_pair[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias_pair[1])
        self.drop2 = nn.Dropout(drop_pair[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x = x.transpose(1, 2).reshape(bsz, seq_len, dim)
        x = self.proj(x)
        return self.proj_drop(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = bool(inplace)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ParallelScalingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        mlp_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, mlp_hidden_dim + 3 * dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer("qkv_bias", None)
            self.register_parameter("mlp_bias", None)
        else:
            self.register_buffer("qkv_bias", torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = float(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)
        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)
        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)
        q = self.q_norm(q.view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x_attn = x_attn.transpose(1, 2).reshape(bsz, seq_len, dim)
        x_attn = self.attn_out_proj(x_attn)
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)
        y = self.drop_path(self.ls(x_attn + x_mlp))
        return x + y


class SpatioTempIncModule(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        input_chans: int = 768,
        output_chans: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        pre_norm: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        block_fn: Callable[..., nn.Module] = ParallelScalingBlock,
        mlp_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.embed_len = self.input_size
        self.embed_dim = int(embed_dim)
        self.input_chans = int(input_chans)
        self.output_chans = int(output_chans)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, self.embed_dim) * 0.02)
        self.norm_pre = norm_layer(self.embed_dim) if pre_norm else nn.Identity()
        if self.embed_dim != self.input_chans:
            self.pre_proj = nn.Sequential(nn.Linear(self.input_chans, self.embed_dim), nn.GELU())
        else:
            self.pre_proj = nn.Identity()
        if self.embed_dim != self.output_chans:
            self.post_proj = nn.Linear(self.embed_dim, self.output_chans)
        else:
            self.post_proj = nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_proj(x)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        x = self.blocks(x)
        return self.post_proj(x)


class E3ClassifierHead(nn.Module):
    def __init__(self, n_features: int, num_outputs: int = 2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class OfficialE3FusionNetwork(nn.Module):
    """Official E3 EKFN path from `models/mixture_transformer.py`."""

    def __init__(
        self,
        expert_n_features: int,
        num_experts: int,
        embed_dim: int = 200,
        depth: int = 5,
        num_heads: int = 8,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.expert_n_features = int(expert_n_features)
        self.transformer = SpatioTempIncModule(
            input_size=self.num_experts,
            input_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            output_chans=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        if self.expert_n_features != self.embed_dim:
            self.down_size = nn.ModuleList([nn.Linear(self.expert_n_features, self.embed_dim) for _ in range(self.num_experts)])
        else:
            self.down_size = nn.ModuleList([nn.Identity() for _ in range(self.num_experts)])
        self.classifier_head = E3ClassifierHead(self.num_experts * self.embed_dim, num_outputs=num_classes)

    def forward(self, expert_embeddings: torch.Tensor) -> torch.Tensor:
        if expert_embeddings.ndim != 3:
            raise ValueError(f"OfficialE3FusionNetwork expects [B,E,D], got shape={tuple(expert_embeddings.shape)}")
        bsz, num_experts, feat_dim = expert_embeddings.shape
        if num_experts != self.num_experts:
            raise ValueError(f"EKFN was built for {self.num_experts} experts, got {num_experts}")
        if feat_dim != self.expert_n_features:
            raise ValueError(f"EKFN expected expert_n_features={self.expert_n_features}, got {feat_dim}")
        features = []
        for idx, down_size in enumerate(self.down_size):
            features.append(down_size(expert_embeddings[:, idx, :]).unsqueeze(1))
        features = torch.cat(features, dim=1)
        weights = self.transformer(features)
        features = weights * features
        return self.classifier_head(features.flatten(1, -1))
