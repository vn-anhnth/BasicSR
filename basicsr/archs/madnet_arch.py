import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


class ChannelAttention(nn.Module):
    """Channel Attention Module (Squeeze-and-Excitation style)."""
    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, max(num_feat // reduction, 4), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(num_feat // reduction, 4), num_feat, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class RMAM(nn.Module):
    """Residual Multiscale Module with Attention Mechanism (RMAM).
    Combines multiscale parallel convolutions (3x3 and 5x5 equivalent / 3x3 dilated)
    with cross-scale feature concatenation and channel attention.
    """
    def __init__(self, num_feat):
        super(RMAM, self).__init__()
        mid_feat = num_feat // 2

        self.conv1_1 = nn.Conv2d(num_feat, mid_feat, 3, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(num_feat, mid_feat, 3, padding=2, dilation=2, bias=True)

        self.conv2_1 = nn.Conv2d(mid_feat * 2, mid_feat, 3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(mid_feat * 2, mid_feat, 3, padding=2, dilation=2, bias=True)

        self.fuse = nn.Conv2d(mid_feat * 2, num_feat, 1, bias=True)
        self.ca = ChannelAttention(num_feat)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        # Scale 1
        x1_1 = self.act(self.conv1_1(x))
        x1_2 = self.act(self.conv1_2(x))
        x1 = torch.cat([x1_1, x1_2], dim=1)

        # Scale 2
        x2_1 = self.act(self.conv2_1(x1))
        x2_2 = self.act(self.conv2_2(x1))
        x2 = torch.cat([x2_1, x2_2], dim=1)

        out = self.fuse(x2)
        out = self.ca(out)
        return out + x


class DRPB(nn.Module):
    """Dual Residual-Path Block (DRPB).
    Contains stacked RMAM modules with dual residual pathways and dense feature aggregation.
    """
    def __init__(self, num_feat, n_rmam=2):
        super(DRPB, self).__init__()
        self.blocks = nn.ModuleList([RMAM(num_feat) for _ in range(n_rmam)])
        self.fuse = nn.Conv2d(num_feat * (n_rmam + 1), num_feat, 1, bias=True)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        feats = [x]
        cur = x
        for b in self.blocks:
            cur = b(cur)
            feats.append(cur)
        out = self.act(self.fuse(torch.cat(feats, dim=1)))
        return out + x


@ARCH_REGISTRY.register()
class MADNet(nn.Module):
    """MADNet: Multi-scale Attention-based Dense Network for Single Image Super-Resolution.
    Paper: MADNet: A Fast and Lightweight Network for Single-Image Super Resolution (IEEE TCYB).
    
    Args:
        in_nc (int): Channel number of input images. Default: 3.
        out_nc (int): Channel number of output images. Default: 3.
        num_feat (int): Number of intermediate feature channels. Default: 48.
        num_blocks (int): Number of DRPB blocks. Default: 4.
        num_rmam (int): Number of RMAM modules per DRPB. Default: 2.
        upscale (int): Upscaling factor (2, 3, 4, etc.). Default: 4.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        num_feat=48,
        num_blocks=4,
        num_rmam=2,
        upscale=4
    ):
        super(MADNet, self).__init__()
        self.upscale = upscale

        # Shallow feature extraction
        self.head = nn.Conv2d(in_nc, num_feat, 3, padding=1, bias=True)

        # Efficient Feature Extraction Network (EFEN) with dense block connections
        self.drpbs = nn.ModuleList([DRPB(num_feat, n_rmam=num_rmam) for _ in range(num_blocks)])
        self.global_fuse = nn.Conv2d(num_feat * (num_blocks + 1), num_feat, 1, bias=True)
        self.global_conv = nn.Conv2d(num_feat, num_feat, 3, padding=1, bias=True)

        # Upsampling Network (UN) with PixelShuffle
        if upscale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(num_feat, num_feat * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(num_feat, out_nc, 3, padding=1, bias=True)
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, padding=1, bias=True),
                nn.PixelShuffle(upscale),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(num_feat, out_nc, 3, padding=1, bias=True)
            )

        # Global residual bicubic-like skip path
        self.skip_conv = nn.Conv2d(in_nc, out_nc * (upscale ** 2), 3, padding=1, bias=True)
        self.skip_ps = nn.PixelShuffle(upscale)

    def forward(self, x):
        skip = self.skip_ps(self.skip_conv(x))
        feat0 = self.head(x)
        
        dense_feats = [feat0]
        cur = feat0
        for block in self.drpbs:
            cur = block(cur)
            dense_feats.append(cur)

        feat_dense = self.global_fuse(torch.cat(dense_feats, dim=1))
        feat_res = self.global_conv(feat_dense) + feat0

        out = self.upsampler(feat_res)
        return out + skip
