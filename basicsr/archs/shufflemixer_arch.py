import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class SplitPointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super(SplitPointMlp, self).__init__()
        hidden_dim = int(dim // 2 * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim // 2, hidden_dim, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim // 2, 1, 1, 0),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.fc(x1)
        x = torch.cat([x1, x2], dim=1)
        # Channel shuffle with g=8 groups
        b, c, h, w = x.shape
        g = 8
        d = c // g
        x = x.view(b, g, d, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_3d = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        norm_3d = self.body(x_3d)
        return norm_3d.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class SMLayer(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super(SMLayer, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.spatial = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        self.mlp1 = SplitPointMlp(dim, mlp_ratio)
        self.mlp2 = SplitPointMlp(dim, mlp_ratio)

    def forward(self, x):
        x = self.mlp1(self.norm1(x)) + x
        x = self.spatial(x)
        x = self.mlp2(self.norm2(x)) + x
        return x


class FMBlock(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super(FMBlock, self).__init__()
        self.net = nn.Sequential(
            SMLayer(dim, kernel_size, mlp_ratio),
            SMLayer(dim, kernel_size, mlp_ratio),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim + 16, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim + 16, dim, 1, 1, 0)
        )

    def forward(self, x):
        x = self.net(x) + x
        x = self.conv(x) + x
        return x


@ARCH_REGISTRY.register()
class ShuffleMixer(nn.Module):
    """ShuffleMixer: An Efficient ConvNet for Image Super-Resolution.
    Paper: NeurIPS 2022 (Winner of NTIRE 2022 Efficient SR Challenge - Complexity Track).

    Args:
        in_nc (int): Number of input channels. Default: 3.
        out_nc (int): Number of output channels. Default: 3.
        n_feats (int): Number of channels. Default: 64 (32 for tiny model).
        kernel_size (int): Kernel size of Depthwise convolution. Default: 7.
        n_blocks (int): Number of feature mixing blocks. Default: 5.
        mlp_ratio (int): Expanding factor of point-wise MLP. Default: 2.
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        n_feats=64,
        kernel_size=7,
        n_blocks=5,
        mlp_ratio=2,
        upscale=4
    ):
        super(ShuffleMixer, self).__init__()
        self.scale = upscale

        self.to_feat = nn.Conv2d(in_nc, n_feats, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(
            *[FMBlock(n_feats, kernel_size, mlp_ratio) for _ in range(n_blocks)]
        )

        if self.scale == 4:
            self.upsampling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True)
            )
        else:
            self.upsampling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * (self.scale ** 2), 1, 1, 0),
                nn.PixelShuffle(self.scale),
                nn.SiLU(inplace=True)
            )

        self.tail = nn.Conv2d(n_feats, out_nc, 3, 1, 1)

    def forward(self, x):
        base = x
        feat = self.to_feat(x)
        feat = self.blocks(feat)
        feat = self.upsampling(feat)
        out = self.tail(feat)
        base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return out + base
