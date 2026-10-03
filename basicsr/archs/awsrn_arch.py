import math
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


class Scale(nn.Module):
    """Adaptive learnable scale factor."""
    def __init__(self, init_value=1e-3):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class AWRU(nn.Module):
    """Adaptive Weighted Residual Unit."""
    def __init__(self, n_feats, kernel_size, block_feats, wn, res_scale=1.0, act=None):
        super(AWRU, self).__init__()
        if act is None:
            act = nn.ReLU(True)
        self.res_scale = Scale(res_scale)
        self.x_scale = Scale(1.0)
        
        body = [
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2)),
            act,
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2))
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.res_scale(self.body(x)) + self.x_scale(x)


class AWMS(nn.Module):
    """Adaptive Weighted Multi-scale Module for reconstruction."""
    def __init__(self, scale, in_channels, out_channels, wn):
        super(AWMS, self).__init__()
        out_feats = scale * scale * out_channels
        self.tail_k3 = wn(nn.Conv2d(in_channels, out_feats, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(in_channels, out_feats, 5, padding=5 // 2, dilation=1))
        self.tail_k7 = wn(nn.Conv2d(in_channels, out_feats, 7, padding=7 // 2, dilation=1))
        self.tail_k9 = wn(nn.Conv2d(in_channels, out_feats, 9, padding=9 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.25)
        self.scale_k5 = Scale(0.25)
        self.scale_k7 = Scale(0.25)
        self.scale_k9 = Scale(0.25)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))
        x2 = self.pixelshuffle(self.scale_k7(self.tail_k7(x)))
        x3 = self.pixelshuffle(self.scale_k9(self.tail_k9(x)))
        return x0 + x1 + x2 + x3


class LFB(nn.Module):
    """Local Fusion Block with n_awru AWRU units and local residual fusion."""
    def __init__(self, n_feats, kernel_size, block_feats, n_awru=4, wn=lambda x: x, act=None):
        super(LFB, self).__init__()
        if act is None:
            act = nn.ReLU(True)
        self.n_awru = n_awru
        self.blocks = nn.ModuleList([
            AWRU(n_feats, kernel_size, block_feats, wn=wn, res_scale=1.0, act=act)
            for _ in range(n_awru)
        ])
        self.reduction = wn(nn.Conv2d(n_feats * n_awru, n_feats, 3, padding=3 // 2))
        self.res_scale = Scale(1.0)
        self.x_scale = Scale(1.0)

    def forward(self, x):
        features = []
        cur = x
        for b in self.blocks:
            cur = b(cur)
            features.append(cur)
        res = self.reduction(torch.cat(features, dim=1))
        return self.res_scale(res) + self.x_scale(x)


@ARCH_REGISTRY.register()
class AWSRN(nn.Module):
    """Adaptive Weighted Super-Resolution Network (AWSRN).
    Supports AWSRN-S, AWSRN-SD, AWSRN-M, and AWSRN.
    
    Args:
        in_nc (int): Channel number of input images. Default: 3.
        out_nc (int): Channel number of output images. Default: 3.
        n_feats (int): Intermediate feature channel count. Default: 32.
        block_feats (int): Inner block feature channel count in AWRU. Default: 128.
        n_blocks (int): Number of LFB blocks. Default: 3 (AWSRN-M).
        n_awru (int): Number of AWRU units per LFB. Default: 4.
        upscale (int): Upscaling factor (2, 3, 4, etc.). Default: 4.
        use_weight_norm (bool): Whether to use weight normalization. Default: True.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        n_feats=32,
        block_feats=128,
        n_blocks=3,
        n_awru=4,
        upscale=4,
        use_weight_norm=True
    ):
        super(AWSRN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)

        if use_weight_norm:
            wn = lambda x: torch.nn.utils.weight_norm(x)
        else:
            wn = lambda x: x

        # Shallow feature extraction (Head)
        self.head = wn(nn.Conv2d(in_nc, n_feats, 3, padding=3 // 2))

        # Deep feature extraction (Body)
        body = []
        for _ in range(n_blocks):
            body.append(LFB(n_feats, kernel_size, block_feats, n_awru=n_awru, wn=wn, act=act))
        self.body = nn.Sequential(*body)

        # Reconstruction module (Tail)
        self.tail = AWMS(upscale, n_feats, out_nc, wn=wn)

        # Global residual skip connection with PixelShuffle
        out_feats = upscale * upscale * out_nc
        self.skip = nn.Sequential(
            wn(nn.Conv2d(in_nc, out_feats, 3, padding=3 // 2)),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        s = self.skip(x)
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        return out + s
