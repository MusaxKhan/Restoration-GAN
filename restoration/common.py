"""Shared helpers: seeding, device, loaders, evaluation, MLflow, figures, checkpoints."""
from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import corruptions as C  # noqa: E402
from .losses import psnr, ssim  # noqa: E402

LEVELS = ["low", "medium", "high"]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def make_loader(ds, batch_size, shuffle=False, workers=None, batch_sampler=None):
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, **kw)


# ------------------------------------------------------------------ evaluation
def objective_value(l1: float, ssim_v: float) -> float:
    """Validation objective (lower = better): L1 + (1 - SSIM).

    Deliberately independent of the loss weight alpha that Optuna tunes, so that different
    alpha values are compared on the same yardstick."""
    return l1 + (1.0 - ssim_v)


@torch.no_grad()
def evaluate_restoration(fn, loader, device, max_batches: int | None = None, return_rows: bool = False,
                        needs_labels: bool = False):
    """Run `fn(corrupted)->restored` over a manifest-driven loader and aggregate PSNR/SSIM/L1.

    Returns {"overall", "by_type", "by_type_level"} for the model output and "input_*" baselines
    (metrics of the unrestored corrupted image), so improvements can be quantified."""
    rows = []
    for b, (x, y, lab, lev) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast(device):
            out = fn(x, lab.to(device)) if needs_labels else fn(x)
        out = out.float().clamp(0, 1)
        l1 = (out - y).abs().flatten(1).mean(1)
        rows.append(torch.stack([psnr(out, y), ssim(out, y, per_sample=True), l1,
                                 psnr(x, y), ssim(x, y, per_sample=True), (x - y).abs().flatten(1).mean(1),
                                 lab.float().to(device), lev.float().to(device)], 1).cpu())
    r = torch.cat(rows).numpy()
    names = ["psnr", "ssim", "l1", "input_psnr", "input_ssim", "input_l1"]

    def agg(m):
        d = {n: float(r[m, i].mean()) for i, n in enumerate(names)}
        d["objective"] = objective_value(d["l1"], d["ssim"])
        d["n"] = int(m.sum())
        return d

    lab, lev = r[:, 6], r[:, 7]
    res = {"overall": agg(np.ones(len(r), bool)), "by_type": {}, "by_type_level": {}}
    for t, tn in enumerate(C.CLASS_NAMES):
        m = lab == t
        if m.any():
            res["by_type"][tn] = agg(m)
            if t != C.CLEAN:
                for li, ln in enumerate(LEVELS):
                    ml = m & (lev == li)
                    if ml.any():
                        res["by_type_level"][f"{tn}/{ln}"] = agg(ml)
    return (res, r) if return_rows else res


# ------------------------------------------------------------------ figures
def restoration_grid(fn, ds, indices, device, path, extra_fn=None):
    """Rows: corrupted input | restored | clean target | absolute error (x4 for visibility)."""
    xs, ys = zip(*[(ds[i][0], ds[i][1]) for i in indices])
    x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
    with torch.no_grad():
        out = fn(x).float().clamp(0, 1)
    err = ((out - y).abs() * 4).clamp(0, 1)
    n = len(indices)
    fig, ax = plt.subplots(4, n, figsize=(1.6 * n, 6.6))
    ax = np.atleast_2d(ax).reshape(4, n)
    for j in range(n):
        for i, (im, t) in enumerate([(x, "input"), (out, "restored"), (y, "clean"), (err, "|error|x4")]):
            ax[i, j].imshow(im[j].permute(1, 2, 0).cpu().numpy())
            ax[i, j].axis("off")
            if j == 0:
                ax[i, j].set_title(t, fontsize=8, loc="left")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    plt.close(fig)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ mlflow
def mlflow_setup(experiment: str):
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    return mlflow


def flat_metrics(prefix: str, res: dict) -> dict:
    out = {}
    for k, v in res["overall"].items():
        out[f"{prefix}/{k}"] = v
    for t, d in res["by_type"].items():
        out[f"{prefix}/{t}/psnr"] = d["psnr"]
        out[f"{prefix}/{t}/ssim"] = d["ssim"]
    return out
