import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SpatialShift(nn.Module):
    """Parameter-free spatial shift operation for efficient feature interaction."""
    def __init__(self, n_div=4):
        super(SpatialShift, self).__init__()
        self.n_div = n_div

    def forward(self, x):
        c = x.size(1)
        g = c // self.n_div
        out = torch.zeros_like(x)
        # Shift in 4 directions
        out[:, 0:g, :, :-1] = x[:, 0:g, :, 1:]          # shift left
        out[:, g:2*g, :, 1:] = x[:, g:2*g, :, :-1]      # shift right
        out[:, 2*g:3*g, :-1, :] = x[:, 2*g:3*g, 1:, :]  # shift up
        out[:, 3*g:4*g, 1:, :] = x[:, 3*g:4*g, :-1, :]  # shift down
        out[:, 4*g:, :, :] = x[:, 4*g:, :, :]            # remaining identity
        return out


class EFFN(nn.Module):
    """Enhanced Feed-Forward Network with spatial shift operation."""
    def __init__(self, dim, mlp_ratio=2.0):
        super(EFFN, self).__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, bias=True)
        self.shift = SpatialShift(n_div=4)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, bias=True)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.shift(x)
        x = self.act(self.conv(x))
        x = self.fc2(x)
        return x + res


class TCSRBlock(nn.Module):
    """Transformer-Convolution Interaction Block."""
    def __init__(self, dim, mlp_ratio=2.0):
        super(TCSRBlock, self).__init__()
        self.norm = LayerNorm(dim)
        # Local & contextual interaction convs
        self.conv_local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.conv_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.effn = EFFN(dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        res = x
        x = self.norm(x)
        attn = torch.sigmoid(self.conv_proj(self.conv_local(x)))
        x = res * attn
        x = self.effn(x)
        return x


@ARCH_REGISTRY.register()
class TCSR(nn.Module):
    """TCSR: Transformer and CNN Interaction Network for Image Super-Resolution.
    Paper: Incorporating Transformer Designs into Convolutions for Lightweight Image Super-Resolution (IEEE TCYB 2024).
    Supports TCSR-B (~450K params).

    Args:
        in_nc (int): Channel number of input images. Default: 3.
        out_nc (int): Channel number of output images. Default: 3.
        embed_dim (int): Intermediate channel depth. Default: 48.
        depths (list): Number of blocks per stage. Default: [4, 4, 4, 4].
        mlp_ratio (float): MLP expanding ratio in EFFN. Default: 2.0.
        upscale (int): Upscaling factor (2, 3, 4). Default: 4.
    """
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        embed_dim=48,
        depths=(4, 4, 4, 4),
        mlp_ratio=2.0,
        upscale=4,
        **kwargs
    ):
        super(TCSR, self).__init__()
        self.scale = upscale
        self.head = nn.Conv2d(in_nc, embed_dim, 3, 1, 1, bias=True)

        self.stages = nn.ModuleList()
        for d in depths:
            stage = nn.Sequential(*[TCSRBlock(embed_dim, mlp_ratio=mlp_ratio) for _ in range(d)])
            self.stages.append(stage)

        self.fuse = nn.Conv2d(embed_dim * len(depths), embed_dim, 1, bias=True)
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)

        self.upsampler = nn.Sequential(
            nn.Conv2d(embed_dim, out_nc * (upscale ** 2), 3, 1, 1, bias=True),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        feat0 = self.head(x)
        cur = feat0
        stage_outputs = []
        for stage in self.stages:
            cur = stage(cur)
            stage_outputs.append(cur)

        fused = self.fuse(torch.cat(stage_outputs, dim=1))
        fused = self.conv_after(fused) + feat0

        out = self.upsampler(fused)
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return out + base
