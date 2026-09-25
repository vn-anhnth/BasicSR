import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class PA(nn.Module):
    """Pixel Attention layer."""
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return torch.mul(x, y)


class PAConv(nn.Module):
    """Pixel Attention Convolution block."""
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.sigmoid(self.k2(x))
        out = torch.mul(self.k3(x), y)
        return self.k4(out)


class SCPA(nn.Module):
    """Self-Calibrated Pixel Attention block."""
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Conv2d(
            group_width, group_width, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False
        )
        self.PAConv = PAConv(group_width)
        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out_a = self.lrelu(self.conv1_a(x))
        out_b = self.lrelu(self.conv1_b(x))

        out_a = self.lrelu(self.k1(out_a))
        out_b = self.lrelu(self.PAConv(out_b))

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        return out + residual


@ARCH_REGISTRY.register()
class PAN(nn.Module):
    """Pixel Attention Network (PAN) for Efficient Image Super-Resolution.
    Paper: Efficient Image Super-Resolution Using Pixel Attention (ECCV 2020 Workshops).

    Args:
        in_nc (int): Number of input channels. Default: 3.
        out_nc (int): Number of output channels. Default: 3.
        nf (int): Number of trunk feature channels. Default: 40.
        unf (int): Number of upsampler feature channels. Default: 24.
        nb (int): Number of SCPA blocks. Default: 16.
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
    """
    def __init__(self, in_nc=3, out_nc=3, nf=40, unf=24, nb=16, upscale=4):
        super(PAN, self).__init__()
        self.upscale = upscale

        # First convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # Main trunk SCPA blocks
        self.SCPA_trunk = nn.Sequential(*[SCPA(nf=nf, reduction=2) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling module
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if upscale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        if self.upscale == 2 or self.upscale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.upscale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.upscale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)
        ilr = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return out + ilr
