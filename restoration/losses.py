"""Differentiable SSIM, PSNR and the L1+SSIM restoration loss used in Tasks 1-3."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gauss_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.repeat(channels, 1, 1, 1).to(dtype or torch.float32)


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5,
         per_sample: bool = False) -> torch.Tensor:
    """Mean SSIM (Wang et al. 2004) for images in [0,1] of shape (B,C,H,W)."""
    c = x.shape[1]
    w = _gauss_window(window, sigma, c, x.device, x.dtype)
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1) if per_sample else s.mean()


def psnr(x: torch.Tensor, y: torch.Tensor, per_sample: bool = True) -> torch.Tensor:
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    p = 10 * torch.log10(1.0 / mse)
    return p if per_sample else p.mean()


class L1SSIMLoss(torch.nn.Module):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)."""

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        s = ssim(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * (1 - s), l1.detach(), s.detach()
