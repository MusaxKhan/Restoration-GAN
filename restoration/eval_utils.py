"""Selection of representative examples and failure cases for the report figures."""
from __future__ import annotations

import numpy as np

from .data import corruptions as C

LEVEL_NAMES = ["low", "medium", "high"]


def representative_indices(rows: np.ndarray, n_extra: int = 2, seed: int = 0) -> list[tuple[int, str]]:
    """10 'typical' examples (the median-SSIM entry of each condition group: clean + 3 types x 3 levels)
    plus `n_extra` seeded random ones. `rows` is the per-sample array from evaluate_restoration."""
    out = []
    lab, lev, ssim = rows[:, 6], rows[:, 7], rows[:, 1]
    groups = [(C.CLEAN, -1)] + [(t, l) for t in (C.SALT, C.BLUR, C.OCC) for l in range(3)]
    for t, l in groups:
        idx = np.where((lab == t) & (lev == l))[0]
        if len(idx):
            med = idx[np.argsort(ssim[idx])[len(idx) // 2]]
            out.append((int(med), f"{C.CLASS_NAMES[t]}" + (f"/{LEVEL_NAMES[l]}" if l >= 0 else "")))
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(rows), n_extra, replace=False):
        out.append((int(i), f"random {C.CLASS_NAMES[int(lab[i])]}"))
    return out


def failure_indices(rows: np.ndarray) -> list[tuple[int, str]]:
    """Worst (lowest-SSIM) example of each condition type: clean, salt, blur, occlusion."""
    out = []
    lab, ssim = rows[:, 6], rows[:, 1]
    for t in range(4):
        idx = np.where(lab == t)[0]
        if len(idx):
            out.append((int(idx[np.argmin(ssim[idx])]), f"worst {C.CLASS_NAMES[t]}"))
    return out
