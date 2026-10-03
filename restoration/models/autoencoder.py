"""Convolutional denoising autoencoder with a dense bottleneck (no skip connections)."""
from __future__ import annotations

import torch.nn as nn


def _down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))


def _up(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class DenoisingAE(nn.Module):
    """Convolutional denoising autoencoder with a genuine compressed latent and no skip connections.

    128x128x3 -> 4 stride-2 conv blocks -> 8x8x(8c) -> bottleneck -> 8x8x(8c) -> 4 transposed-conv blocks
    -> 3x3 conv -> sigmoid. Two bottleneck types (a design choice investigated with Optuna):

    * ``dense``   : flatten -> Linear -> latent vector of `latent_dim` numbers -> Linear -> reshape.
                    Every spatial location is mixed by the fully-connected layer (global code, blurry output).
    * ``spatial`` : 1x1 conv to `latent_ch` channels -> latent map latent_ch x 8 x 8 (= 64*latent_ch numbers)
                    -> 1x1 conv back. The code keeps its 8x8 spatial layout (convolutional prior), which preserves
                    local structure much better at the same compression.

    All information must pass through the latent: there are deliberately no skip connections.
    """

    def __init__(self, base_ch: int = 32, latent_dim: int = 512, dropout: float = 0.1,
                 latent_type: str = "dense", latent_ch: int = 16):
        super().__init__()
        assert latent_type in ("dense", "spatial")
        c = base_ch
        self.base_ch, self.latent_type, self.latent_ch = base_ch, latent_type, latent_ch
        self.latent_dim = latent_dim if latent_type == "dense" else latent_ch * 64
        self.encoder = nn.Sequential(_down(3, c), _down(c, 2 * c), _down(2 * c, 4 * c), _down(4 * c, 8 * c))
        self.flat = 8 * c * 8 * 8
        if latent_type == "dense":
            self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(self.flat, latent_dim), nn.LeakyReLU(0.2, True),
                                           nn.Dropout(dropout))
            self.from_latent = nn.Sequential(nn.Linear(latent_dim, self.flat), nn.ReLU(True))
        else:
            self.to_latent = nn.Sequential(nn.Conv2d(8 * c, latent_ch, 1), nn.BatchNorm2d(latent_ch),
                                           nn.LeakyReLU(0.2, True), nn.Dropout2d(dropout))
            self.from_latent = nn.Sequential(nn.Conv2d(latent_ch, 8 * c, 1), nn.BatchNorm2d(8 * c), nn.ReLU(True))
        self.decoder = nn.Sequential(_up(8 * c, 4 * c), _up(4 * c, 2 * c), _up(2 * c, c), _up(c, c))
        self.head = nn.Sequential(nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        h = self.from_latent(z)
        if self.latent_type == "dense":
            h = h.view(-1, 8 * self.base_ch, 8, 8)
        return self.head(self.decoder(h))

    def forward(self, x):
        return self.decode(self.encode(x))
