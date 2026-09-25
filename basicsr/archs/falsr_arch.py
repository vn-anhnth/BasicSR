import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class FALSRCell(nn.Module):
    """FALSR micro-cell incorporating residual connections and separable convolutions."""
    def __init__(self, channels):
        super(FALSRCell, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return out3 + x


@ARCH_REGISTRY.register()
class FALSR(nn.Module):
    """Fast, Accurate and Lightweight Super-Resolution (FALSR).
    Paper: Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search (CVPR 2019 Oral).
    Supports FALSR-B (~326K params) and FALSR-C (~408K params).

    Args:
        in_nc (int): Number of input channels. Default: 3.
        out_nc (int): Number of output channels. Default: 3.
        num_feat (int): Intermediate feature channels. Default: 64.
        variant (str): 'B' or 'C'. Default: 'B'.
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        num_feat=64,
        variant='B',
        upscale=4,
        **kwargs
    ):
        super(FALSR, self).__init__()
        self.scale = upscale
        self.variant = variant.upper()

        self.head = nn.Conv2d(in_nc, num_feat, 3, 1, 1, bias=True)

        if self.variant == 'B':
            # FALSR-B: 6 cells with partial sparse connections (~326K params)
            self.n_cells = 6
            self.cells = nn.ModuleList([FALSRCell(num_feat) for _ in range(self.n_cells)])
            self.fuse = nn.Conv2d(num_feat * 3, num_feat, 1, bias=True)
        else:
            # FALSR-C: 8 cells with sparse 8-way connections (~408K params)
            self.n_cells = 8
            self.cells = nn.ModuleList([FALSRCell(num_feat) for _ in range(self.n_cells)])
            self.fuse = nn.Conv2d(num_feat * 4, num_feat, 1, bias=True)

        if upscale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_feat, out_nc, 3, 1, 1, bias=True)
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1, bias=True),
                nn.PixelShuffle(upscale),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_feat, out_nc, 3, 1, 1, bias=True)
            )

    def forward(self, x):
        feat0 = self.head(x)
        cur = feat0
        outputs = []
        for cell in self.cells:
            cur = cell(cur)
            outputs.append(cur)

        if self.variant == 'B':
            # Sparse fusion for B (samples 3 stages)
            fused = self.fuse(torch.cat([outputs[1], outputs[3], outputs[5]], dim=1))
        else:
            # Sparse fusion for C (samples 4 stages)
            fused = self.fuse(torch.cat([outputs[1], outputs[3], outputs[5], outputs[7]], dim=1))

        out = self.upsampler(fused + feat0)
        return out
