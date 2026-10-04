import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class Conv3XC(nn.Module):
    """3x3 convolution composed with 1x1 convolutions for structural reparameterization."""
    def __init__(self, c_in, c_out, gain1=1, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        w_full = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b_full = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        sk_w = F.pad(sk_w, [1, 1, 1, 1])

        self.eval_conv.weight.data = w_full + sk_w
        self.eval_conv.bias.data = b_full + sk_b

    def forward(self, x):
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block."""
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1, bias=bias)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        return out, out1, sim_att


@ARCH_REGISTRY.register()
class SPAN(nn.Module):
    """Swift Parameter-free Attention Network (SPAN).
    Winner of NTIRE 2024 Efficient Super-Resolution Challenge.

    Args:
        in_nc (int): Channel number of input images. Default: 3.
        out_nc (int): Channel number of output images. Default: 3.
        feature_channels (int): Intermediate feature channels. Default: 48.
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
        bias (bool): Whether to use bias in Conv3XC. Default: True.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        feature_channels=48,
        upscale=4,
        bias=True
    ):
        super(SPAN, self).__init__()
        self.upscale = upscale

        self.conv_1 = Conv3XC(in_nc, feature_channels, gain1=2, s=1, bias=bias)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1, bias=bias)

        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, out_nc * (upscale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1, _, _ = self.block_1(out_feature)
        out_b2, _, _ = self.block_2(out_b1)
        out_b3, _, _ = self.block_3(out_b2)
        out_b4, _, _ = self.block_4(out_b3)
        out_b5, _, _ = self.block_5(out_b4)
        out_b6, out_b5_2, _ = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], dim=1))
        return self.upsampler(out)
