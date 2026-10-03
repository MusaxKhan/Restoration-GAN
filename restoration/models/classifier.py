"""Corruption classifier: clean / salt-and-pepper / blur / occlusion."""
from __future__ import annotations

import torch.nn as nn


class CorruptionClassifier(nn.Module):
    """Plain CNN: `depth` conv blocks (channels base, 2*base, ... capped at 8*base) with max-pooling,
    global average pooling, dropout, and a linear head producing 4 logits."""

    def __init__(self, base_ch: int = 32, depth: int = 4, dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        layers, cin = [], 3
        for i in range(depth):
            cout = min(base_ch * 2 ** i, base_ch * 8)
            layers += [nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                       nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                       nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, n_classes))
        self.base_ch, self.depth = base_ch, depth

    def forward(self, x):
        return self.head(self.features(x))
