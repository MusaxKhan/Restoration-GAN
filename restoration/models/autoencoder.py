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
    """128x128x3 -> 4 stride-2 conv blocks -> 8x8x(8c) -> flatten -> Linear -> latent (d)
    -> Linear -> 8x8x(8c) -> 4 transposed-conv blocks -> 3x3 conv -> sigmoid.

    `latent_dim` is the genuine compressed representation (e.g. 512 numbers vs 49152 input values),
    `base_ch` the number of channels after the first encoder stage (doubled at each stage).
    There are deliberately no skip connections: all information must pass through the latent vector.
    """

    def __init__(self, base_ch: int = 32, latent_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        c = base_ch
        self.base_ch, self.latent_dim = base_ch, latent_dim
        self.encoder = nn.Sequential(_down(3, c), _down(c, 2 * c), _down(2 * c, 4 * c), _down(4 * c, 8 * c))
        self.flat = 8 * c * 8 * 8
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(self.flat, latent_dim), nn.LeakyReLU(0.2, True),
                                       nn.Dropout(dropout))
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, self.flat), nn.ReLU(True))
        self.decoder = nn.Sequential(_up(8 * c, 4 * c), _up(4 * c, 2 * c), _up(2 * c, c), _up(c, c))
        self.head = nn.Sequential(nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        h = self.from_latent(z).view(-1, 8 * self.base_ch, 8, 8)
        return self.head(self.decoder(h))

    def forward(self, x):
        return self.decode(self.encode(x))
