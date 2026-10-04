import torch
import torch.nn as nn
from collections import OrderedDict

from basicsr.utils.registry import ARCH_REGISTRY


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    """Contrast-aware Channel Attention (CCA) layer."""
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class IMDModule(nn.Module):
    """Information Multi-Distillation Block (IMDB)."""
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


@ARCH_REGISTRY.register()
class IMDN(nn.Module):
    """Information Multi-Distillation Network for Lightweight Super-Resolution (AIM 2019 Champion).

    Paper: Lightweight Image Super-Resolution with Information Multi-distillation Network
    Reference: https://github.com/Zheng222/IMDN
    """
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN, self).__init__()
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.IMDBs = nn.ModuleList([IMDModule(in_channels=nf) for _ in range(num_modules)])
        self.c = nn.Sequential(
            conv_layer(nf * num_modules, nf, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B = out_fea
        features = []
        for imdb in self.IMDBs:
            out_B = imdb(out_B)
            features.append(out_B)

        out_fused = self.c(torch.cat(features, dim=1))
        out_lr = self.LR_conv(out_fused) + out_fea
        output = self.upsampler(out_lr)
        return output
