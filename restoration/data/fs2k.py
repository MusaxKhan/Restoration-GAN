"""FS2K photo-sketch pairs for Task 4 (official train/test, stratified 85/15 train/val split, seed 42)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMG = 128
SEED = 42
ROOT = Path(os.environ.get("DATA_ROOT", "data"))
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]


def _pair_paths(item: dict) -> tuple[Path, Path]:
    d, n = item["image_name"].split("/")  # e.g. photo1/image0110
    base = ROOT / "FS2K"
    photo = base / "photo" / d / f"{n}.jpg"
    sk_dir = base / "sketch" / d.replace("photo", "sketch")
    stem = n.replace("image", "sketch")
    sketch = next(p for p in (sk_dir / f"{stem}.jpg", sk_dir / f"{stem}.png") if p.exists())
    return photo, sketch


def _cache(split: str):
    """Official split ('train'/'test') -> uint8 arrays photos (N,128,128,3), sketches (N,...), styles (N,)."""
    p = ROOT / "cache" / f"fs2k_{split}_{IMG}.npz"
    if p.exists():
        z = np.load(p)
        return z["photos"], z["sketches"], z["styles"]
    ann = json.loads((ROOT / "FS2K" / f"anno_{split}.json").read_text())
    photos, sketches, styles = [], [], []
    for it in ann:
        pp, sp = _pair_paths(it)
        photos.append(np.asarray(Image.open(pp).convert("RGB").resize((IMG, IMG), Image.BICUBIC)))
        sketches.append(np.asarray(Image.open(sp).convert("RGB").resize((IMG, IMG), Image.BICUBIC)))
        styles.append(int(it["style"]))
    arrs = (np.stack(photos), np.stack(sketches), np.array(styles))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, photos=arrs[0], sketches=arrs[1], styles=arrs[2])
    return arrs


def split_indices() -> dict[str, list[int]]:
    """15% of the official training portion held out for validation, stratified by sketch style."""
    _, _, styles = _cache("train")
    tr, va = train_test_split(np.arange(len(styles)), test_size=0.15, stratify=styles, random_state=SEED)
    return {"train": sorted(tr.tolist()), "val": sorted(va.tolist())}


class FS2KDataset(Dataset):
    """Returns (photo, sketch, style) with images in [-1, 1], CHW.

    Training augmentation is *paired*: the same random horizontal flip and the same random crop box
    (then resized back to 128x128) are applied to the photograph and its sketch."""

    def __init__(self, mode: str, subset: int | None = None):
        assert mode in ("train", "val", "test")
        self.mode = mode
        official = "test" if mode == "test" else "train"
        self.photos, self.sketches, self.styles = _cache(official)
        self.idx = np.arange(len(self.styles)) if mode == "test" else np.array(split_indices()[mode])
        if subset:
            self.idx = self.idx[:subset]

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def _t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 127.5 - 1.0

    def __getitem__(self, i):
        j = self.idx[i]
        x, y = self._t(self.photos[j]), self._t(self.sketches[j])
        if self.mode == "train":
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1).item() < 0.5:  # identical random crop of both members of the pair
                s = int(IMG * (0.8 + 0.2 * torch.rand(1).item()))
                top, left = torch.randint(0, IMG - s + 1, (2,)).tolist()
                x = F.interpolate(x[None, :, top:top + s, left:left + s], size=IMG, mode="bilinear", align_corners=False)[0]
                y = F.interpolate(y[None, :, top:top + s, left:left + s], size=IMG, mode="bilinear", align_corners=False)[0]
        return x, y, int(self.styles[j])
