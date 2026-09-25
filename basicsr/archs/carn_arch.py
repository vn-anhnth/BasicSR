import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.body(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        return F.relu(out + x, inplace=True)


class EResidualBlock(nn.Module):
    """Efficient Residual Block with grouped convolutions used in CARN-M."""
    def __init__(self, in_channels, out_channels, group=1):
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        return F.relu(out + x, inplace=True)


class CascadingBlock(nn.Module):
    def __init__(self, channels=64, group=1, efficient=False):
        super(CascadingBlock, self).__init__()
        if efficient:
            self.b1 = EResidualBlock(channels, channels, group=group)
        else:
            self.b1 = ResidualBlock(channels, channels)
        self.c1 = BasicBlock(channels * 2, channels, 1, 1, 0)
        self.c2 = BasicBlock(channels * 3, channels, 1, 1, 0)
        self.c3 = BasicBlock(channels * 4, channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                    nn.ReLU(inplace=True),
                    nn.PixelShuffle(2)
                ]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(3)
            ]
        else:
            raise NotImplementedError(f'Scale {scale} is not supported')
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class CARN(nn.Module):
    """Cascading Residual Network (CARN & CARN-M).
    Paper: Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network (ECCV 2018).
    
    Args:
        in_nc (int): Input channel count. Default: 3.
        out_nc (int): Output channel count. Default: 3.
        num_feat (int): Intermediate feature channels. Default: 64.
        group (int): Group count for group convolutions (CARN-M uses group=4, CARN uses group=1).
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
        efficient (bool): If True, uses CARN-M architecture (EResidualBlock). Default: False.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        num_feat=64,
        group=1,
        upscale=4,
        efficient=False
    ):
        super(CARN, self).__init__()
        self.entry = nn.Conv2d(in_nc, num_feat, 3, 1, 1)

        self.b1 = CascadingBlock(num_feat, group=group, efficient=efficient)
        self.b2 = CascadingBlock(num_feat, group=group, efficient=efficient)
        self.b3 = CascadingBlock(num_feat, group=group, efficient=efficient)

        self.c1 = BasicBlock(num_feat * 2, num_feat, 1, 1, 0)
        self.c2 = BasicBlock(num_feat * 3, num_feat, 1, 1, 0)
        self.c3 = BasicBlock(num_feat * 4, num_feat, 1, 1, 0)

        self.upsample = UpsampleBlock(num_feat, scale=upscale, group=group)
        self.exit = nn.Conv2d(num_feat, out_nc, 3, 1, 1)

    def forward(self, x):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3)
        return self.exit(out)
