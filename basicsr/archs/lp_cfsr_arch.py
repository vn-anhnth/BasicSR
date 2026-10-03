import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class HeightAwareMod(nn.Module):
    """
    Height-Aware Modulation Module (H-Mod).
    Inspired by EfficientRCTCDecoder's Height-wise Average Pooling:
    Instead of heavy 2D square large kernels, we capture the character strokes of license plates
    via Height-wise strip pooling (H -> 1) combined with depthwise 1D horizontal context.
    This sharply reduces FLOPs on edge devices (Jetson Nano) while prioritizing vertical alphanumeric strokes.
    """

    def __init__(self, dim, dw_size=7):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # Branch 1: Value projection
        self.v = nn.Conv2d(dim, dim, 1)

        # Branch 2: Height-Aware Attention Gating
        # 1. Height-wise average pooling compresses vertical dimension to capture horizontal stroke sequences
        # 2. 1D Depthwise Conv along width (W) to correlate neighboring characters
        self.w_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, dw_size), stride=1, padding=(0, dw_size // 2), groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        # Local 3x3 depthwise refinement for fine edge details
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        
        # Output 1x1 projection
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        res = x
        x = self.norm(x)
        
        # Height-wise pooling: [B, C, H, W] -> [B, C, 1, W]
        h_profile = x.mean(dim=2, keepdim=True)
        h_gate = torch.sigmoid(self.w_conv(h_profile))
        
        # Modulate value features with height-aware gate + local details
        val = self.v(x)
        attn = (val * h_gate) + self.local_conv(val)
        
        out = self.proj(attn)
        return out


class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP with Edge-Aware Filtering (B-MLP).
    Inspired by EfficientRCTCDecoder's 1x1 Bottleneck Channel Fusion:
    Instead of calculating Sobel/Laplacian gradient operators on all expanded channels (e.g. 192 channels),
    we compress features into a compact edge bottleneck (e.g. 16 or 32 channels), compute sharp character edges,
    and fuse them back. This dramatically cuts activation memory and FLOPs on Jetson Nano.
    """

    def __init__(self, dim, mlp_ratio=2, bottleneck_dim=16):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * mlp_ratio)
        self.bottleneck_dim = min(bottleneck_dim, self.hidden_dim)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, self.hidden_dim, 1)
        self.act = nn.GELU()

        # 1x1 Bottleneck for edge distillation (reduces params & memory)
        self.edge_bottleneck = nn.Conv2d(self.hidden_dim, self.bottleneck_dim, 1)
        self.edge_fusion = nn.Conv2d(self.bottleneck_dim, self.hidden_dim, 1)

        # Predefined edge filters on compact bottleneck channels
        # 1. Sobel X
        self.scale_sobel_x = nn.Parameter(torch.randn(self.bottleneck_dim, 1, 1, 1) * 1e-3)
        self.sobel_x_bias = nn.Parameter(torch.zeros(self.bottleneck_dim))
        mask_sobel_x = torch.zeros((self.bottleneck_dim, 1, 3, 3), dtype=torch.float32)
        for i in range(self.bottleneck_dim):
            mask_sobel_x[i, 0, 0, 0] = 1.0
            mask_sobel_x[i, 0, 1, 0] = 2.0
            mask_sobel_x[i, 0, 2, 0] = 1.0
            mask_sobel_x[i, 0, 0, 2] = -1.0
            mask_sobel_x[i, 0, 1, 2] = -2.0
            mask_sobel_x[i, 0, 2, 2] = -1.0
        self.register_buffer('mask_sobel_x', mask_sobel_x)

        # 2. Sobel Y
        self.scale_sobel_y = nn.Parameter(torch.randn(self.bottleneck_dim, 1, 1, 1) * 1e-3)
        self.sobel_y_bias = nn.Parameter(torch.zeros(self.bottleneck_dim))
        mask_sobel_y = torch.zeros((self.bottleneck_dim, 1, 3, 3), dtype=torch.float32)
        for i in range(self.bottleneck_dim):
            mask_sobel_y[i, 0, 0, 0] = 1.0
            mask_sobel_y[i, 0, 0, 1] = 2.0
            mask_sobel_y[i, 0, 0, 2] = 1.0
            mask_sobel_y[i, 0, 2, 0] = -1.0
            mask_sobel_y[i, 0, 2, 1] = -2.0
            mask_sobel_y[i, 0, 2, 2] = -1.0
        self.register_buffer('mask_sobel_y', mask_sobel_y)

        # 3. Laplacian
        self.scale_laplacian = nn.Parameter(torch.randn(self.bottleneck_dim, 1, 1, 1) * 1e-3)
        self.laplacian_bias = nn.Parameter(torch.zeros(self.bottleneck_dim))
        mask_laplacian = torch.zeros((self.bottleneck_dim, 1, 3, 3), dtype=torch.float32)
        for i in range(self.bottleneck_dim):
            mask_laplacian[i, 0, 0, 0] = 1.0
            mask_laplacian[i, 0, 1, 0] = 1.0
            mask_laplacian[i, 0, 1, 2] = 1.0
            mask_laplacian[i, 0, 2, 1] = 1.0
            mask_laplacian[i, 0, 1, 1] = -4.0
        self.register_buffer('mask_laplacian', mask_laplacian)

        # Learnable 3x3 depthwise convolution
        self.pos = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, groups=self.hidden_dim)
        self.fc2 = nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)

        # Spatial pos response
        out = self.pos(x)

        # Compact edge extraction
        edge_feat = self.edge_bottleneck(x)
        edge_x = F.conv2d(edge_feat, self.scale_sobel_x * self.mask_sobel_x,
                          self.sobel_x_bias, stride=1, padding=1, groups=self.bottleneck_dim)
        edge_y = F.conv2d(edge_feat, self.scale_sobel_y * self.mask_sobel_y,
                          self.sobel_y_bias, stride=1, padding=1, groups=self.bottleneck_dim)
        edge_lap = F.conv2d(edge_feat, self.scale_laplacian * self.mask_laplacian,
                            self.laplacian_bias, stride=1, padding=1, groups=self.bottleneck_dim)

        edge_combined = self.edge_fusion(edge_x + edge_y + edge_lap)
        out = out + edge_combined

        x = x + self.act(out)
        x = self.fc2(x)
        return x


class LPCFSRBlock(nn.Module):
    """
    License Plate CFSR Block (LP-Block) combining Height-Aware Modulation and Bottleneck MLP.
    """

    def __init__(self, dim, dw_size=7, mlp_ratio=2, bottleneck_dim=16):
        super().__init__()
        self.attn = HeightAwareMod(dim, dw_size)
        self.mlp = BottleneckMLP(dim, mlp_ratio, bottleneck_dim)
        
        self.layer_scale_1 = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.view(1, -1, 1, 1) * self.attn(x)
        x = x + self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(x)
        return x


class LPCFSRLayer(nn.Module):
    def __init__(self, dim, depth, dw_size=7, mlp_ratio=2, bottleneck_dim=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            LPCFSRBlock(dim, dw_size, mlp_ratio, bottleneck_dim) for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        res = x
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x) + res
        return x


class UpsampleOneStep(nn.Sequential):
    """Direct PixelShuffle Upsampling."""

    def __init__(self, scale, num_feat, num_out_ch):
        m = [
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale)
        ]
        super(UpsampleOneStep, self).__init__(*m)


@ARCH_REGISTRY.register()
class LPCFSR(nn.Module):
    """
    License-Plate-Tailored CFSR (LP-CFSR) for Efficient Super-Resolution on Edge Devices (Jetson Nano).
    
    Key Innovations:
    1. Height-Aware Modulation (H-Mod): Exploits plate geometry (H << W) via Height-wise Average Pooling,
       replacing bulky 2D large kernels with strip-gating and 1D horizontal convolutions.
    2. Bottleneck Gradient Distillation (B-MLP): Employs 1x1 channel bottleneck fusion to extract Sobel/Laplacian
       character stroke gradients efficiently with minimal memory footprint.
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=48,
        depths=(6, 6),
        dw_size=7,
        mlp_ratio=2,
        bottleneck_dim=16,
        upscale=2,
        img_range=1.,
        **kwargs
    ):
        super(LPCFSR, self).__init__()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 1, 1, 1))

        self.upscale = upscale
        self.num_layers = len(depths)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for d in depths:
            layer = LPCFSRLayer(embed_dim, d, dw_size, mlp_ratio, bottleneck_dim)
            self.layers.append(layer)

        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        h, w = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_feat = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_feat)) + x_feat
        out = self.upsample(res)

        out = out / self.img_range + self.mean
        return out
