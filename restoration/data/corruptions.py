"""Corruption definitions shared by training (runtime), validation and test (manifest).

A corruption is described by a small JSON-serialisable *spec* dict. `apply_corruption`
is a pure function of (image, spec), so the same code path serves both the random
runtime pipeline (spec sampled on the fly) and the deterministic manifests (spec stored).

Images are float32 numpy arrays, shape (H, W, 3), values in [0, 1].
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF

CLEAN, SALT, BLUR, OCC = 0, 1, 2, 3
CLASS_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]
LEVEL_NAMES = ["low", "medium", "high"]

# Fixed severity levels used for the final test set (assignment specification).
TEST_SALT_P = (0.03, 0.08, 0.15)
TEST_BLUR = ((3, 0.7), (5, 1.5), (7, 2.5))
TEST_OCC = ((1, 0.10), (2, 0.20), (3, 0.35))  # (number of rectangles, target coverage)


# --------------------------------------------------------------------------- operators
def salt_and_pepper(img: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Replace a fraction `p` of pixels by black or white (50/50)."""
    rng = np.random.default_rng(seed)
    h, w, _ = img.shape
    hit = rng.random((h, w)) < p
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[hit & white] = 1.0
    out[hit & ~white] = 0.0
    return out


def gaussian_blur(img: np.ndarray, k: int, sigma: float) -> np.ndarray:
    t = torch.from_numpy(img).permute(2, 0, 1)
    t = TF.gaussian_blur(t, [k, k], [sigma, sigma])
    return t.permute(1, 2, 0).numpy()


def occlude(img: np.ndarray, rects: list[list[int]]) -> np.ndarray:
    out = img.copy()
    for x0, y0, x1, y1 in rects:
        out[y0:y1, x0:x1, :] = 0.0
    return out


def make_rects(rng: np.random.Generator, n: int, coverage: float, size: int = 128) -> list[list[int]]:
    """Sample `n` non-overlapping black rectangles jointly covering ~`coverage` of the image.

    The target area is split randomly between the rectangles; each rectangle gets a random
    aspect ratio in [0.5, 2] and a random position. Placement is re-tried until rectangles
    do not overlap (so the achieved coverage equals the target up to integer rounding);
    if that fails after many attempts the overlapping layout is accepted.
    """
    total = coverage * size * size
    for _ in range(200):
        share = rng.dirichlet(np.full(n, 4.0)) if n > 1 else np.array([1.0])
        rects, mask = [], np.zeros((size, size), bool)
        ok = True
        for a in share * total:
            ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))  # width / height
            w = int(round(np.sqrt(a * ar)))
            h = int(round(a / max(w, 1)))
            w, h = min(max(w, 4), size), min(max(h, 4), size)
            x0 = int(rng.integers(0, size - w + 1))
            y0 = int(rng.integers(0, size - h + 1))
            if mask[y0:y0 + h, x0:x0 + w].any():
                ok = False
            mask[y0:y0 + h, x0:x0 + w] = True
            rects.append([x0, y0, x0 + w, y0 + h])
        if ok:
            return rects
    return rects


# --------------------------------------------------------------------------- specs
def coverage_of(rects: list[list[int]], size: int = 128) -> float:
    mask = np.zeros((size, size), bool)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def sample_train_spec(rng: np.random.Generator, ctype: int | None = None) -> dict:
    """Sample a runtime training corruption. `ctype=None` -> uniform over the 4 conditions."""
    if ctype is None:
        ctype = int(rng.integers(0, 4))
    if ctype == CLEAN:
        return {"type": CLEAN}
    if ctype == SALT:
        return {"type": SALT, "p": float(rng.uniform(0.02, 0.15)), "seed": int(rng.integers(2**31))}
    if ctype == BLUR:
        return {"type": BLUR, "k": int(rng.choice([3, 5, 7])), "sigma": float(rng.uniform(0.5, 2.5))}
    n = int(rng.integers(1, 4))
    rects = make_rects(rng, n, float(rng.uniform(0.10, 0.35)))
    return {"type": OCC, "rects": rects, "coverage": coverage_of(rects)}


def apply_corruption(img: np.ndarray, spec: dict) -> np.ndarray:
    t = spec["type"]
    if t == CLEAN:
        return img
    if t == SALT:
        return salt_and_pepper(img, spec["p"], spec["seed"])
    if t == BLUR:
        return gaussian_blur(img, spec["k"], spec["sigma"])
    if t == OCC:
        return occlude(img, spec["rects"])
    raise ValueError(f"unknown corruption type {t}")


def spec_from_ui(ctype: int, level: int, seed: int = 0) -> dict:
    """Spec for one of the three fixed test severities (used by the web application)."""
    if ctype == CLEAN:
        return {"type": CLEAN}
    if ctype == SALT:
        return {"type": SALT, "p": TEST_SALT_P[level], "seed": seed}
    if ctype == BLUR:
        return {"type": BLUR, "k": TEST_BLUR[level][0], "sigma": TEST_BLUR[level][1]}
    n, cov = TEST_OCC[level]
    rects = make_rects(np.random.default_rng(seed), n, cov)
    return {"type": OCC, "rects": rects, "coverage": coverage_of(rects)}
