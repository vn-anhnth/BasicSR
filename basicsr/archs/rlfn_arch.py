# RLFN: Residual Local Feature Network for Efficient Super-Resolution
# NTIRE 2022 Efficient Super-Resolution Challenge Runtime Winner
# Official repository: https://github.com/bytedance/RLFN

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """Convolution layer with auto-calculated padding."""
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """Upsample features according to upscale_factor."""
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """Enhanced Spatial Attention (ESA) modified for RLFN."""
    def __init__(self, esa_channels, n_feats):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        # Handle small spatial sizes (e.g. 12x12 LQ in x4 SR with 48x48 GT)
        if c1.size(2) < 7 or c1.size(3) < 7:
            pad_h = max(0, 7 - c1.size(2))
            pad_w = max(0, 7 - c1.size(3))
            c1_pad = F.pad(c1, (0, pad_w, 0, pad_h), mode='replicate')
            v_max = F.max_pool2d(c1_pad, kernel_size=7, stride=3)
        else:
            v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """Residual Local Feature Block (RLFB)."""
    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16):
        super(RLFB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)

        out = self.c2_r(out)
        out = self.act(out)

        out = self.c3_r(out)
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))
        return out


@ARCH_REGISTRY.register()
class RLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Official NTIRE 2022 Runtime Track Winner
    """
    def __init__(self, in_channels=3, out_channels=3, feature_channels=52, num_blocks=6, upscale=4):
        super(RLFN, self).__init__()
        self.conv_1 = conv_layer(in_channels, feature_channels, kernel_size=3)

        self.blocks = nn.ModuleList([
            RLFB(feature_channels) for _ in range(num_blocks)
        ])

        self.conv_2 = conv_layer(feature_channels, feature_channels, kernel_size=3)
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out = out_feature
        for block in self.blocks:
            out = block(out)

        out_low_resolution = self.conv_2(out) + out_feature
        output = self.upsampler(out_low_resolution)
        return output
