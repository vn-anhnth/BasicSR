import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.utils.registry import ARCH_REGISTRY


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESA(nn.Module):
    """Enhanced Spatial Attention (ESA) block used in RFDN."""
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
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
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RFDB(nn.Module):
    """Residual Feature Distillation Block (RFDB)."""
    def __init__(self, in_channels):
        super(RFDB, self).__init__()
        self.dc = in_channels // 2
        self.rc = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.rc, self.dc, 1)
        self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c3_d = conv_layer(self.rc, self.dc, 1)
        self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c4 = conv_layer(self.rc, self.dc, 3)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.act(self.c1_r(input) + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    """Residual Feature Distillation Network for Lightweight Super-Resolution (AIM 2020 Champion).

    Paper: Residual Feature Distillation Network for Lightweight Image Super-Resolution
    Reference: https://github.com/njulj/RFDN
    """
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.c = nn.Sequential(
            conv_layer(nf * num_modules, nf, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        return output
