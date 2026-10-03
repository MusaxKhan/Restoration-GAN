"""Oxford-IIIT Pet data for Tasks 1-3: cached 128x128 RGB arrays, fixed split, datasets."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from . import corruptions as C

IMG = 128
SEED = 42
ROOT = Path(os.environ.get("DATA_ROOT", "data"))
MANIFEST_DIR = Path(os.environ.get("MANIFEST_DIR", "manifests"))


def _cache_path(split: str) -> Path:
    return ROOT / "cache" / f"pets_{split}_{IMG}.npy"


def build_cache(split: str) -> np.ndarray:
    """Load an official split (trainval/test), convert to RGB, resize to 128x128, store uint8."""
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(str(ROOT / "oxford"), split=split, download=True)
    arr = np.empty((len(ds), IMG, IMG, 3), np.uint8)
    for i in range(len(ds)):
        img = ds[i][0].convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        arr[i] = np.asarray(img)
    _cache_path(split).parent.mkdir(parents=True, exist_ok=True)
    np.save(_cache_path(split), arr)
    return arr


def load_images(split: str) -> np.ndarray:
    p = _cache_path(split)
    return np.load(p, mmap_mode="r") if p.exists() else build_cache(split)


def get_split_indices() -> dict[str, list[int]]:
    """80/20 train/val split of the official trainval collection, random seed 42."""
    n = 3680
    perm = np.random.RandomState(SEED).permutation(n)
    n_train = int(round(0.8 * n))
    return {"train": sorted(perm[:n_train].tolist()), "val": sorted(perm[n_train:].tolist())}


# --------------------------------------------------------------------------- manifests
def build_val_manifest() -> list[dict]:
    """One fixed random corruption per validation image (type uniform over the 4 conditions,
    severity drawn from the training distribution). Fully determined by SEED."""
    idx = get_split_indices()["val"]
    rng = np.random.default_rng(SEED)
    out = []
    for i, img_idx in enumerate(idx):
        spec = C.sample_train_spec(rng)
        out.append({"id": i, "img": img_idx, "type": spec["type"], "spec": spec})
    return out


def build_test_manifest() -> list[dict]:
    """Every official test image x 10 conditions: clean + 3 types x 3 fixed severities."""
    rng = np.random.default_rng(SEED + 1)
    out = []
    for img_idx in range(3669):
        out.append({"img": img_idx, "type": C.CLEAN, "level": -1, "spec": {"type": C.CLEAN}})
        for level in range(3):
            seed = int(rng.integers(2**31))
            out.append({"img": img_idx, "type": C.SALT, "level": level,
                        "spec": {"type": C.SALT, "p": C.TEST_SALT_P[level], "seed": seed}})
            k, s = C.TEST_BLUR[level]
            out.append({"img": img_idx, "type": C.BLUR, "level": level,
                        "spec": {"type": C.BLUR, "k": k, "sigma": s}})
            n, cov = C.TEST_OCC[level]
            rects = C.make_rects(rng, n, cov)
            out.append({"img": img_idx, "type": C.OCC, "level": level,
                        "spec": {"type": C.OCC, "rects": rects, "coverage": C.coverage_of(rects)}})
    for i, e in enumerate(out):
        e["id"] = i
    return out


def load_manifest(name: str) -> list[dict]:
    p = MANIFEST_DIR / f"{name}_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m restoration.data.make_manifests`")
    return json.loads(p.read_text())


def severity_level(entry: dict) -> int:
    """Low/medium/high bin (0/1/2) of a manifest entry; -1 for clean.

    Test entries carry their fixed level. Validation entries use the continuous severity
    parameter split into thirds of the training range."""
    if "level" in entry:
        return entry["level"]
    s = entry["spec"]
    if s["type"] == C.CLEAN:
        return -1
    if s["type"] == C.SALT:
        v = (s["p"] - 0.02) / 0.13
    elif s["type"] == C.BLUR:
        v = (s["sigma"] - 0.5) / 2.0
    else:
        v = (s["coverage"] - 0.10) / 0.25
    return int(min(2, max(0, v * 3)))


# --------------------------------------------------------------------------- datasets
class PetsDataset(Dataset):
    """Returns (corrupted, clean, label, level) as float32 tensors in [0,1], CHW.

    mode='train': corruption sampled at runtime on every access (new type + severity each time).
                  `types` restricts the sampled condition (e.g. specialists); None = all 4, uniform.
                  An index may be a (idx, label) tuple (see BalancedBatchSampler) to force a class.
    mode='val'/'test': deterministic, driven by the stored manifest. `types` filters entries.
    """

    def __init__(self, mode: str, types: list[int] | None = None, subset: int | None = None):
        assert mode in ("train", "val", "test")
        self.mode, self.types = mode, types
        self._rng = None
        self.images = load_images("test" if mode == "test" else "trainval")
        if mode == "train":
            self.entries = get_split_indices()["train"]
        else:
            man = load_manifest(mode)
            self.entries = [e for e in man if types is None or e["type"] in types]
        if subset:
            self.entries = self.entries[:subset]

    def __len__(self):
        return len(self.entries)

    def _get_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def __getitem__(self, item):
        if self.mode == "train":
            forced = None
            if isinstance(item, tuple):
                item, forced = item
            img = self.images[self.entries[item]].astype(np.float32) / 255.0
            rng = self._get_rng()
            ctype = forced if forced is not None else (
                None if self.types is None else int(rng.choice(self.types)))
            spec = C.sample_train_spec(rng, ctype)
            level = -1
        else:
            e = self.entries[item]
            img = self.images[e["img"]].astype(np.float32) / 255.0
            spec, level = e["spec"], severity_level(e)
        corrupted = C.apply_corruption(img, spec)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()
        return to_t(corrupted), to_t(img), spec["type"], level


class BalancedBatchSampler(Sampler):
    """Batches containing exactly batch_size/4 samples of every condition (classifier training).

    Yields lists of (index, forced_class) tuples; one epoch = len(dataset) // batch_size batches.
    """

    def __init__(self, n: int, batch_size: int, seed: int = SEED):
        assert batch_size % 4 == 0, "batch size must be a multiple of 4 for balanced batches"
        self.n, self.bs, self.seed, self.epoch = n, batch_size, seed, 0

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.n)
        per = self.bs // 4
        labels = np.repeat(np.arange(4), per)
        for b in range(len(self)):
            idx = order[b * self.bs:(b + 1) * self.bs]
            yield [(int(i), int(c)) for i, c in zip(idx, rng.permutation(labels))]
