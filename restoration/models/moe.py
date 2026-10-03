"""Soft mixture-of-experts restoration: gate + identity branch + 3 specialist autoencoders."""
from __future__ import annotations

import torch
import torch.nn as nn

from .autoencoder import DenoisingAE
from .classifier import CorruptionClassifier


class SoftMoE(nn.Module):
    """x_hat = w0 * x + w1 * A_salt(x) + w2 * A_blur(x) + w3 * A_occ(x),  w = softmax(G(x) / tau).

    Branch order matches the corruption labels: 0 clean (identity), 1 salt, 2 blur, 3 occlusion.
    forward() returns (reconstruction, gate_logits, weights, expert_outputs)."""

    def __init__(self, gate: CorruptionClassifier, experts: list[DenoisingAE], tau: float = 1.0):
        super().__init__()
        assert len(experts) == 3
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.tau = tau

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)
        outs = [e(x) for e in self.experts]
        recon = w[:, 0, None, None, None] * x
        for k, o in enumerate(outs):
            recon = recon + w[:, k + 1, None, None, None] * o
        return recon, logits, w, torch.stack(outs, 1)


class SoftMoEExport(nn.Module):
    """Inference wrapper for ONNX: input image -> (reconstruction, weights)."""

    def __init__(self, moe: SoftMoE):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        recon, _, w, _ = self.moe(x)
        return recon, w
