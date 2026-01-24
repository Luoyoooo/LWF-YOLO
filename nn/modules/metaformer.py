# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Minimal DCGFormer blocks used by custom YOLO11 modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("LayerNormGeneral", "LayerNormWithoutBias", "DCGFormerBlock")


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Scale(nn.Module):
    """Scale vector by element multiplications."""

    def __init__(self, dim: int, init_value: float = 1.0, trainable: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LayerNormGeneral(nn.Module):
    """General LayerNorm for different input formats."""

    def __init__(self, affine_shape, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5) -> None:
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """LayerNorm without bias for speed."""

    def __init__(self, normalized_shape, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class DynamicConvGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, reduction=16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = max(int(2 * hidden_features / 3), 1)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        reduction_channels = max(hidden_features // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_features, reduction_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, hidden_features, 1, bias=False),
            nn.Sigmoid(),
        )

        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer(),
        )

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        gate_weight = self.gate(v)
        v = v * gate_weight
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class DCGFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        token_mixer=nn.Identity,
        mlp=DynamicConvGLU,
        norm_layer=LayerNormWithoutBias,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x.permute(0, 3, 1, 2)) + self.layer_scale2(
            self.drop_path2(self.mlp(self.norm2(x).permute(0, 3, 1, 2)))
        )
        return x
