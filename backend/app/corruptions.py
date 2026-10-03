"""Numpy/OpenCV re-implementation of the training corruptions (keeps the backend free of PyTorch).

Behaviour is identical to restoration/data/corruptions.py for the same spec; a unit test compares them.
Images: float32 (H, W, 3) in [0, 1]."""
from __future__ import annotations

import cv2
import numpy as np

CLEAN, SALT, BLUR, OCC = 0, 1, 2, 3
CLASS_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]
LEVELS = ["low", "medium", "high"]
TEST_SALT_P = (0.03, 0.08, 0.15)
TEST_BLUR = ((3, 0.7), (5, 1.5), (7, 2.5))
TEST_OCC = ((1, 0.10), (2, 0.20), (3, 0.35))


def salt_and_pepper(img, p, seed):
    rng = np.random.default_rng(seed)
    h, w, _ = img.shape
    hit = rng.random((h, w)) < p
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[hit & white] = 1.0
    out[hit & ~white] = 0.0
    return out


def gaussian_blur(img, k, sigma):
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def occlude(img, rects):
    out = img.copy()
    for x0, y0, x1, y1 in rects:
        out[y0:y1, x0:x1, :] = 0.0
    return out


def make_rects(rng, n, coverage, size=128):
    total = coverage * size * size
    for _ in range(200):
        share = rng.dirichlet(np.full(n, 4.0)) if n > 1 else np.array([1.0])
        rects, mask, ok = [], np.zeros((size, size), bool), True
        for a in share * total:
            ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
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


def spec_from_ui(ctype: int, level: int, seed: int = 0) -> dict:
    """One of the three fixed test severities (low / medium / high) of a corruption type."""
    if ctype == CLEAN:
        return {"type": CLEAN}
    if ctype == SALT:
        return {"type": SALT, "p": TEST_SALT_P[level], "seed": seed}
    if ctype == BLUR:
        return {"type": BLUR, "k": TEST_BLUR[level][0], "sigma": TEST_BLUR[level][1]}
    n, cov = TEST_OCC[level]
    return {"type": OCC, "rects": make_rects(np.random.default_rng(seed), n, cov), "coverage": cov}


def apply_corruption(img, spec):
    t = spec["type"]
    if t == CLEAN:
        return img
    if t == SALT:
        return salt_and_pepper(img, spec["p"], spec["seed"])
    if t == BLUR:
        return gaussian_blur(img, spec["k"], spec["sigma"])
    if t == OCC:
        return occlude(img, spec["rects"])
    raise ValueError(t)


def describe(spec: dict) -> dict:
    """Human-readable settings echoed back to the UI."""
    t = spec["type"]
    d = {"type": CLASS_NAMES[t]}
    if t == SALT:
        d["probability"] = round(spec["p"], 3)
    elif t == BLUR:
        d["kernel_size"], d["sigma"] = spec["k"], spec["sigma"]
    elif t == OCC:
        d["rectangles"] = len(spec["rects"])
        d["coverage_target"] = spec.get("coverage")
    return d
