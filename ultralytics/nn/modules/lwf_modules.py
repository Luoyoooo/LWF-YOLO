# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom LWF-YOLO modules extracted from the mixed Ultralytics module files."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k, C3k2, DyReLU, build_activation_layer, constant_init, normal_init
from .conv import Conv
from .head import Detect
from .ops_dcnv3.modules import DCNv3_DyHead

__all__ = (
    "SobelConv",
    "ScaleEdge",
    "EdgeFusion",
    "GetIndexOutput",
    "MSEFE",
    "LayerNormGeneral",
    "DynamicConvGLU",
    "DCGFormerBlock",
    "DCGFormerC3k",
    "DCGFormer",
    "DyDCNBlock",
    "Detect_DyDCN",
    "DyDCN",
)


# ── MSEFE: SobelConv / ScaleEdge / EdgeFusion / GetIndexOutput


class SobelConv(nn.Module):
    """Fixed Sobel edge extractor used by the multi-scale edge enhancement branch."""

    def __init__(self, channel) -> None:
        super().__init__()
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[
            :, :, 0
        ]


class ScaleEdge(nn.Module):
    """Generate multi-scale edge features from a shallow backbone feature map."""

    def __init__(self, inc, oucs=None) -> None:
        super().__init__()
        self.single_output = oucs is None
        oucs = [inc] if oucs is None else oucs
        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)

    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        return outputs[0] if self.single_output else outputs


class EdgeFusion(nn.Module):
    """Fuse multi-scale edge priors back into the main convolutional branch."""

    def __init__(self, inc, ouc) -> None:
        super().__init__()
        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k=1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)

    def forward(self, x, edge=None):
        if edge is not None:
            x = list(x) if isinstance(x, (list, tuple)) else [x]
            edge = list(edge) if isinstance(edge, (list, tuple)) else [edge]
            x = x + edge
        x = torch.cat(x, dim=1)
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x


class GetIndexOutput(nn.Module):
    """Select a single tensor from a list of multi-scale outputs."""

    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


# ── DCGFormer: DCGFormerC3k / DCGFormerBlock / DynamicConvGLU / DCGFormer


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
    """Dynamic channel-mixing GLU block used inside the DCGFormer feed-forward branch."""

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
    """Core DCGFormer block with token mixing, dynamic gating, and dual residual scaling."""

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


class DCGFormerC3k(C3k):
    """C3k wrapper that replaces the inner bottlenecks with DCGFormer blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(
                DCGFormerBlock(
                    dim=c_,
                    token_mixer=nn.Identity,
                    norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
                )
                for _ in range(n)
            )
        )


class DCGFormer(C3k2):
    """C3k2-style module whose inner blocks are replaced by DCGFormer units."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            DCGFormerC3k(self.c, self.c, n, shortcut, g)
            if c3k
            else DCGFormerBlock(
                dim=self.c,
                token_mixer=nn.Identity,
                norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
            )
            for _ in range(n)
        )


# ── DyDCN: DyDCNBlock / Detect_DyDCN


class DyDCNBlock(nn.Module):
    """DyDCN feature aggregation block using DCNv3-based spatial sampling and dynamic attention."""

    def __init__(
        self,
        in_channels,
        norm_type="GN",
        zero_init_offset=True,
        act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
    ):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        self.offset_and_mask_dim = 3 * 4 * 3 * 3
        self.offset_dim = 2 * 4 * 3 * 3

        self.dw_conv_high = Conv(in_channels, in_channels, 3, g=in_channels)
        self.dw_conv_mid = Conv(in_channels, in_channels, 3, g=in_channels)
        self.dw_conv_low = Conv(in_channels, in_channels, 3, g=in_channels)

        self.spatial_conv_high = DCNv3_DyHead(in_channels)
        self.spatial_conv_mid = DCNv3_DyHead(in_channels)
        self.spatial_conv_low = DCNv3_DyHead(in_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, 3, padding=1, groups=4)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
        )
        self.task_attn_module = DyReLU(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat_ = self.dw_conv_mid(x[level])
            offset_and_mask = self.spatial_conv_offset(mid_feat_)
            offset = offset_and_mask[:, : self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat_ = self.dw_conv_low(x[level - 1])
                offset, mask = self.get_offset_mask(low_feat_)
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                high_feat_ = self.dw_conv_high(x[level + 1])
                offset, mask = self.get_offset_mask(high_feat_)
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs

    def get_offset_mask(self, x):
        n, _, h, w = x.size()
        dtype = x.dtype
        offset_and_mask = self.spatial_conv_offset(x).permute(0, 2, 3, 1)
        offset = offset_and_mask[..., : self.offset_dim]
        mask = offset_and_mask[..., self.offset_dim :].reshape(n, h, w, 4, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(n, h, w, -1).type(dtype)
        return offset, mask


class Detect_DyDCN(Detect):
    """Detection head that inserts DyDCN blocks before the standard YOLO prediction heads."""

    def __init__(self, nc: int = 80, hidc: int = 256, block_num: int = 2, reg_max: int = 16, end2end=False, ch=()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=[hidc] * len(ch))
        self.conv = nn.ModuleList(Conv(x, hidc, 1) for x in ch)
        self.dyhead = nn.Sequential(*[DyDCNBlock(hidc) for _ in range(block_num)])

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        x = [self.conv[i](xi) for i, xi in enumerate(x)]
        x = self.dyhead(x)
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)


# -- Paper-name aliases
class MSEFE(torch.nn.Module):
    """
    Unified MSEFE entry point from Section 3.2.

    Combines ScaleEdge for edge extraction and EdgeFusion for feature fusion.
    This allows external references to use the paper name directly.
    """

    def __init__(self, in_channels, edge_channels=None, out_channels=None):
        super().__init__()
        self.edge_channels = list(edge_channels) if edge_channels is not None else [max(in_channels // 2, 1), in_channels, in_channels * 2]
        self.out_channels = out_channels
        self.scale_edge = ScaleEdge(in_channels, self.edge_channels)
        fusion_out_channels = self.out_channels or self.edge_channels[-1]
        self.edge_fusion = nn.ModuleList(
            EdgeFusion([fusion_out_channels, edge_channel], fusion_out_channels) for edge_channel in self.edge_channels
        )

    def _select_edge_feature(self, backbone_feat, edge_features):
        target_size = backbone_feat.shape[-2:]
        for idx, feat in enumerate(edge_features):
            if feat.shape[-2:] == target_size:
                return idx, feat

        size_distance = [
            abs(feat.shape[-2] * feat.shape[-1] - target_size[0] * target_size[1]) for feat in edge_features
        ]
        idx = min(range(len(edge_features)), key=size_distance.__getitem__)
        feat = F.interpolate(edge_features[idx], size=target_size, mode="bilinear", align_corners=False)
        return idx, feat

    def forward(self, x, backbone_feat=None):
        if backbone_feat is None:
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise TypeError("MSEFE expects either (x, backbone_feat) or a two-element input list/tuple.")
            x, backbone_feat = x
        if isinstance(backbone_feat, (list, tuple)):
            if len(backbone_feat) != 1:
                raise TypeError("MSEFE expects backbone_feat to be a tensor or a single-element list/tuple.")
            backbone_feat = backbone_feat[0]
        edge = self.scale_edge(x)
        idx, matched_edge = self._select_edge_feature(backbone_feat, edge)
        return self.edge_fusion[idx](backbone_feat, matched_edge)


# DyDCNBlock is the core implementation; Detect_DyDCN is the detection-head wrapper.
DyDCN = DyDCNBlock  # Section 3.4
