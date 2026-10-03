import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)
    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    return gumbels.softmax(dim)


class CALayer(nn.Module):
    """Channel Attention layer for SMSR."""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, max(channel // reduction, 4), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channel // reduction, 4), channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(self.avg_pool(x))
        return x * y


class SMB(nn.Module):
    """Sparse Mask Block (SMB)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=4):
        super(SMB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.tau = 1.0
        self.relu = nn.ReLU(True)

        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))
        body = []
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(n_layers - 1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)
        self.collect = nn.Conv2d(out_channels * n_layers, out_channels, 1, 1, 0)

    def forward(self, x):
        spa_mask = x[1]
        ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

        out = []
        fea = x[0]
        for i in range(self.n_layers):
            if i == 0:
                fea = self.body[i](fea)
                fea = fea * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea * ch_mask[:, :, i:i + 1, :1]
            else:
                fea_d = self.body[i](fea * ch_mask[:, :, i - 1:i, :1])
                fea_s = self.body[i](fea * ch_mask[:, :, i - 1:i, 1:])
                fea = (
                    fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask +
                    fea_d * ch_mask[:, :, i:i + 1, :1] +
                    fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask +
                    fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask
                )
            fea = self.relu(fea)
            out.append(fea)

        out = self.collect(torch.cat(out, 1))
        return out, ch_mask


class SMM(nn.Module):
    """Sparse Mask Module (SMM)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SMM, self).__init__()
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels // 4, 2, 3, 2, 1, output_padding=1),
        )
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=4)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        spa_mask = self.spa_mask(x)
        if self.training:
            spa_mask = gumbel_softmax(spa_mask, 1, 1.0)
            out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
            out = self.ca(out) + x
            return out, spa_mask[:, 1:, ...]
        else:
            spa_mask = (spa_mask[:, 1:, ...] > spa_mask[:, :1, ...]).float()
            out, ch_mask = self.body([x, spa_mask])
            out = self.ca(out) + x
            return out, spa_mask


@ARCH_REGISTRY.register()
class SMSR(nn.Module):
    """Exploring Sparsity in Image Super-Resolution for Efficient Inference (CVPR 2021).
    Paper: SMSR: Exploring Sparsity in Image Super-Resolution for Efficient Inference.

    Args:
        in_nc (int): Channel number of input images. Default: 3.
        out_nc (int): Channel number of output images. Default: 3.
        n_feats (int): Intermediate feature channels. Default: 64.
        n_modules (int): Number of SMM modules. Default: 5.
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        n_feats=64,
        n_modules=5,
        upscale=4,
        **kwargs
    ):
        super(SMSR, self).__init__()
        self.scale = upscale
        kernel_size = 3

        # Shallow feature extraction (Head)
        self.head = nn.Sequential(
            nn.Conv2d(in_nc, n_feats, kernel_size, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=1)
        )

        # Body modules
        self.n_modules = n_modules
        self.body = nn.ModuleList([SMM(n_feats, n_feats, kernel_size) for _ in range(n_modules)])

        # Collect fusion
        self.collect = nn.Sequential(
            nn.Conv2d(n_feats * n_modules, n_feats, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        # Upsampling tail
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_nc * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        feat0 = self.head(x)
        out_fea = []
        cur = feat0
        for m in self.body:
            cur, _ = m(cur)
            out_fea.append(cur)

        fused = self.collect(torch.cat(out_fea, 1)) + feat0
        out = self.tail(fused)
        base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return out + base
