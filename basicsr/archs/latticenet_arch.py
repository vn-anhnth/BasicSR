import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


class CC(nn.Module):
    """Combination Coefficient (CC) module used in LatticeNet."""
    def __init__(self, channel, reduction=16):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_mean = self.conv_mean(self.avg_pool(x))
        m_batchsize, C, _, _ = x.size()
        x_dense = x.view(m_batchsize, C, -1)
        ca_std = torch.std(x_dense, dim=2, keepdim=True).view(m_batchsize, C, 1, 1)
        ca_var = self.conv_std(ca_std)
        cc = (ca_mean + ca_var) / 2.0
        return cc


class LatticeBlock(nn.Module):
    """Lattice Block (LB) with butterfly fusion."""
    def __init__(self, nFeat=64, nDiff=16):
        super(LatticeBlock, self).__init__()
        self.conv_block0 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(nFeat - nDiff, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(nFeat - nDiff, nFeat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05, inplace=True)
        )
        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(nFeat - nDiff, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(nFeat - nDiff, nFeat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05, inplace=True)
        )
        self.fea_ca2 = CC(nFeat)
        self.x_ca2 = CC(nFeat)

        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x_feature_shot = self.conv_block0(x)
        fea_ca1 = self.fea_ca1(x_feature_shot)
        x_ca1 = self.x_ca1(x)

        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        out = torch.cat((p3z, q3z), 1)
        out = self.compress(out)
        return out


@ARCH_REGISTRY.register()
class LatticeNet(nn.Module):
    """LatticeNet: Towards Lightweight Image Super-resolution with Lattice Block (ECCV 2020).

    Reference: https://github.com/ymff0592/super-resolution
    """
    def __init__(self, in_nc=3, out_nc=3, num_feat=64, num_diff=16, upscale=4):
        super(LatticeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_nc, num_feat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=True)

        self.body_unit1 = LatticeBlock(num_feat, num_diff)
        self.body_unit2 = LatticeBlock(num_feat, num_diff)
        self.body_unit3 = LatticeBlock(num_feat, num_diff)
        self.body_unit4 = LatticeBlock(num_feat, num_diff)

        self.T_tdm1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.L_tdm1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True))

        self.T_tdm2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.L_tdm2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True))

        self.T_tdm3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.L_tdm3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True))

        self.tail = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(num_feat, out_nc * (upscale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        x_head = self.conv2(self.conv1(x))

        res1 = self.body_unit1(x_head)
        res2 = self.body_unit2(res1)
        res3 = self.body_unit3(res2)
        res4 = self.body_unit4(res3)

        T_tdm1 = self.T_tdm1(res4)
        L_tdm1 = self.L_tdm1(res3)
        out_TDM1 = torch.cat((T_tdm1, L_tdm1), 1)

        T_tdm2 = self.T_tdm2(out_TDM1)
        L_tdm2 = self.L_tdm2(res2)
        out_TDM2 = torch.cat((T_tdm2, L_tdm2), 1)

        T_tdm3 = self.T_tdm3(out_TDM2)
        L_tdm3 = self.L_tdm3(res1)
        out_TDM3 = torch.cat((T_tdm3, L_tdm3), 1)

        res = out_TDM3 + x_head
        out = self.tail(res)
        return out
