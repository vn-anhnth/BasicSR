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


class HeightAwareAWRU(nn.Module):
    """Height-Aware Adaptive Weighted Residual Unit (H-AWRU).
    
    Inspired by EfficientRCTCDecoder:
    1. First Conv: projects n_feats -> block_feats (3x3).
    2. Height-Aware 1D horizontal Conv (1x5) on Height-wise pooled features (mean(dim=2)).
       Captures inter-character profiles with negligible parameter overhead (0.01M).
    3. Output Conv: projects block_feats -> n_feats (3x3).
    """
    def __init__(self, n_feats, kernel_size, block_feats, wn, res_scale=1.0, act=None):
        super(HeightAwareAWRU, self).__init__()
        if act is None:
            act = nn.ReLU(True)
        self.res_scale = Scale(res_scale)
        self.x_scale = Scale(1.0)

        self.conv1 = wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2))
        self.act = act

        # Height-Aware 1D horizontal stream (1x5)
        # Using depthwise separable / 1D conv for maximum parameter efficiency
        self.h_conv = wn(nn.Conv2d(block_feats, block_feats, kernel_size=(1, 5), padding=(0, 2), groups=block_feats))

        self.conv2 = wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        feat = self.act(self.conv1(x))
        # Height-wise average pooling
        h_pool = feat.mean(dim=2, keepdim=True)
        h_feat = self.h_conv(h_pool)
        # Residual fusion
        out = self.conv2(feat + h_feat)
        return self.res_scale(out) + self.x_scale(x)


class LP_AWMS(nn.Module):
    """License Plate Adaptive Weighted Multi-scale Module (LP-AWMS).
    
    Replaces heavy 2D square large kernels (7x7, 9x9) with efficient anisotropic kernels:
    - 3x3 standard kernel
    - 5x5 standard kernel
    - 1x7 horizontal kernel (capturing license plate character widths efficiently)
    - 1x9 horizontal kernel
    Dramatically reduces FLOPs and memory footprint on Jetson Nano while preserving high-frequency edges.
    """
    def __init__(self, scale, in_channels, out_channels, wn):
        super(LP_AWMS, self).__init__()
        out_feats = scale * scale * out_channels
        self.tail_k3 = wn(nn.Conv2d(in_channels, out_feats, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(in_channels, out_feats, 5, padding=5 // 2, dilation=1))
        # Anisotropic 1D kernels along horizontal dimension (License plate text orientation)
        self.tail_k7 = wn(nn.Conv2d(in_channels, out_feats, kernel_size=(1, 7), padding=(0, 3)))
        self.tail_k9 = wn(nn.Conv2d(in_channels, out_feats, kernel_size=(1, 9), padding=(0, 4)))
        
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


class LP_LFB(nn.Module):
    """License Plate Local Fusion Block (LP-LFB) with HeightAwareAWRU units."""
    def __init__(self, n_feats, kernel_size, block_feats, n_awru=4, wn=lambda x: x, act=None):
        super(LP_LFB, self).__init__()
        if act is None:
            act = nn.ReLU(True)
        self.n_awru = n_awru
        self.blocks = nn.ModuleList([
            HeightAwareAWRU(n_feats, kernel_size, block_feats, wn=wn, res_scale=1.0, act=act)
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
class LPAWSRN(nn.Module):
    """License Plate Adaptive Weighted Super-Resolution Network (LP-AWSRN).

    Novel improvements over baseline AWSRN:
    1. Height-Aware AWRU (H-AWRU):
       - Uses Height-wise average pooling and 1D horizontal convolution (1x5)
         to model the consistent vertical stroke height of vehicle license plate characters.
    2. Anisotropic Multi-Scale Reconstruction (LP-AWMS):
       - Replaces FLOPs-heavy 7x7 and 9x9 isotropic convolutions with 1x7 and 1x9 horizontal convolutions,
         significantly reducing latency on Jetson Nano while matching character aspect ratio.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        n_feats=32,
        block_feats=128,
        n_blocks=3,
        n_awru=4,
        upscale=2,
        use_weight_norm=True
    ):
        super(LPAWSRN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)

        if use_weight_norm:
            wn = lambda x: torch.nn.utils.weight_norm(x)
        else:
            wn = lambda x: x

        # Shallow feature extraction
        self.head = wn(nn.Conv2d(in_nc, n_feats, 3, padding=3 // 2))

        # Deep feature extraction with Height-Aware LFBs
        body = []
        for _ in range(n_blocks):
            body.append(LP_LFB(n_feats, kernel_size, block_feats, n_awru=n_awru, wn=wn, act=act))
        self.body = nn.Sequential(*body)

        # Anisotropic Multi-Scale Reconstruction
        self.tail = LP_AWMS(upscale, n_feats, out_nc, wn=wn)

        # Global residual skip connection
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
