"""Style-conditioned pix2pix: U-Net generator and PatchGAN discriminator (128x128)."""
from __future__ import annotations

import torch
import torch.nn as nn

N_STYLES = 3


def _style_map(emb: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return emb[:, :, None, None].expand(-1, -1, h, w)


class UNetGenerator(nn.Module):
    """y_hat = G(x, s).  The style s (0/1/2) is mapped through a *learned* embedding that is
    (i) tiled and concatenated to the input photograph and (ii) concatenated to the 1x1 bottleneck,
    so it can steer both low-level and global features. Seven stride-2 stages 128 -> 1, mirrored decoder with
    skip connections, dropout in the first three decoder stages, tanh output in [-1, 1]."""

    def __init__(self, base_ch: int = 64, style_dim: int = 16, dropout: float = 0.5):
        super().__init__()
        c = base_ch
        self.embed = nn.Embedding(N_STYLES, style_dim)
        enc_ch = [c, 2 * c, 4 * c, 8 * c, 8 * c, 8 * c, 8 * c]

        def down(cin, cout, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        def up(cin, cout, drop):
            layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
            if drop:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(True))
            return nn.Sequential(*layers)

        cin = 3 + style_dim
        self.downs = nn.ModuleList()
        for i, co in enumerate(enc_ch):
            self.downs.append(down(cin, co, norm=i not in (0, 6)))
            cin = co
        # decoder: input of stage k = previous output (+ style at bottleneck) concatenated with skip
        self.ups = nn.ModuleList([
            up(enc_ch[6] + style_dim, enc_ch[5], True),
            up(enc_ch[5] * 2, enc_ch[4], True),
            up(enc_ch[4] * 2, enc_ch[3], True),
            up(enc_ch[3] * 2, enc_ch[2], False),
            up(enc_ch[2] * 2, enc_ch[1], False),
            up(enc_ch[1] * 2, enc_ch[0], False),
        ])
        self.out = nn.Sequential(nn.ConvTranspose2d(enc_ch[0] * 2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        e = self.embed(style)
        h = torch.cat([x, _style_map(e, x.shape[2], x.shape[3])], 1)
        skips = []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = torch.cat([h, _style_map(e, 1, 1)], 1)  # inject style at the bottleneck
        for k, u in enumerate(self.ups):
            h = u(h)
            h = torch.cat([h, skips[5 - k]], 1)
        return self.out(h)


class PatchDiscriminator(nn.Module):
    """D(x, y, s): PatchGAN on the channel-wise concatenation [photo, sketch, tiled style embedding].
    Four 4x4 conv stages (three stride-2, one stride-1) + a final conv give a 14x14 patch map of real/fake logits
    (each logit judges a local receptive field)."""

    def __init__(self, base_ch: int = 64, style_dim: int = 16):
        super().__init__()
        c = base_ch
        self.embed = nn.Embedding(N_STYLES, style_dim)
        cin = 6 + style_dim

        def block(cin, cout, stride, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.net = nn.Sequential(*block(cin, c, 2, False), *block(c, 2 * c, 2), *block(2 * c, 4 * c, 2),
                                 *block(4 * c, 8 * c, 1), nn.Conv2d(8 * c, 1, 4, 1, 1))

    def forward(self, x, y, style):
        e = self.embed(style)
        return self.net(torch.cat([x, y, _style_map(e, x.shape[2], x.shape[3])], 1))
