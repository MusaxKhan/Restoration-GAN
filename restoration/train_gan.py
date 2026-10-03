"""Task 4: style-conditioned face-to-sketch cGAN (pix2pix-style U-Net + PatchGAN).

    L_D = 0.5 * [BCE(D(x,y,s), 1) + BCE(D(x,G(x,s),s), 0)]
    L_G = BCE(D(x,G(x,s),s), 1) + lambda_l1 * L1(y, G(x,s))

    python -m restoration.train_gan --config configs/gan_best.json --out runs/task4 --epochs 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import common as cm  # noqa: E402
from .data.fs2k import STYLE_NAMES, FS2KDataset  # noqa: E402
from .losses import psnr, ssim  # noqa: E402
from .models.gan import PatchDiscriminator, UNetGenerator  # noqa: E402

DEFAULTS = dict(g_lr=2e-4, d_lr=2e-4, batch_size=16, base_ch=64, dropout=0.5, style_dim=16, lambda_l1=100.0,
                epochs=100, subset_train=None, sample_every=10, workers=None)
BCE = F.binary_cross_entropy_with_logits


def to01(t):
    return (t.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def evaluate_gan(G, loader, device) -> dict:
    """L1 / PSNR / SSIM between generated and ground-truth sketches (images mapped to [0,1])."""
    G.eval()
    rows = []
    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)
        g = to01(G(x, s)); t = to01(y)
        rows.append(torch.stack([(g - t).abs().flatten(1).mean(1), psnr(g, t), ssim(g, t, per_sample=True),
                                 s.float()], 1).cpu())
    r = torch.cat(rows).numpy()

    def agg(m):
        d = {"l1": float(r[m, 0].mean()), "psnr": float(r[m, 1].mean()), "ssim": float(r[m, 2].mean()), "n": int(m.sum())}
        d["objective"] = d["l1"] + (1 - d["ssim"])
        return d

    res = {"overall": agg(np.ones(len(r), bool)), "by_style": {}}
    for k, n in enumerate(STYLE_NAMES):
        if (r[:, 3] == k).any():
            res["by_style"][n] = agg(r[:, 3] == k)
    return res


@torch.no_grad()
def sample_grid(G, ds, indices, device, path, all_styles=False):
    """Photo | generated sketch | ground-truth sketch for fixed validation photographs.
    With all_styles=True the generated column is repeated for Style 1/2/3 (conditioning check)."""
    G.eval()
    x = torch.stack([ds[i][0] for i in indices]).to(device)
    y = torch.stack([ds[i][1] for i in indices]).to(device)
    s_true = torch.tensor([ds[i][2] for i in indices], device=device)
    cols = [("photo", x)]
    if all_styles:
        for k in range(3):
            cols.append((f"G(x, {STYLE_NAMES[k]})", G(x, torch.full_like(s_true, k))))
    else:
        cols.append(("generated", G(x, s_true)))
    cols.append(("target", y))
    fig, ax = plt.subplots(len(indices), len(cols), figsize=(1.8 * len(cols), 1.8 * len(indices)), squeeze=False)
    for i in range(len(indices)):
        for j, (t, im) in enumerate(cols):
            ax[i, j].imshow(to01(im[i]).permute(1, 2, 0).cpu().numpy()); ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(t, fontsize=8)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90); plt.close(fig)


def build_G(cfg):
    return UNetGenerator(cfg["base_ch"], cfg["style_dim"], cfg["dropout"])


def train_gan(cfg, out_dir=None, trial=None, mlflow_experiment=None, resume=False, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    cm.seed_everything(42)
    device = cm.get_device()
    train_ds, val_ds = FS2KDataset("train", cfg["subset_train"]), FS2KDataset("val")
    workers = cfg["workers"] if cfg["workers"] is not None else min(2, __import__("os").cpu_count() or 1)
    kw = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, 32, **kw)

    G, D = build_G(cfg).to(device), PatchDiscriminator(cfg["base_ch"], cfg["style_dim"]).to(device)
    oG = torch.optim.Adam(G.parameters(), lr=cfg["g_lr"], betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=cfg["d_lr"], betas=(0.5, 0.999))
    out = Path(out_dir) if out_dir else None
    start, best_obj, best_state, best_res = 0, float("inf"), None, None
    if out and resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); oG.load_state_dict(ck["oG"]); oD.load_state_dict(ck["oD"])
        start, best_obj, best_state, best_res = ck["epoch"], ck["best_obj"], ck["best_state"], ck["best_res"]
        print(f"resumed from epoch {start}")
    fixed = list(range(0, len(val_ds), max(1, len(val_ds) // 6)))[:6]  # same validation photos every time

    mlf = None
    if mlflow_experiment:
        mlf = cm.mlflow_setup(mlflow_experiment)
        mlf.start_run(run_name=f"gan_{int(time.time())}")
        mlf.log_params({k: v for k, v in cfg.items() if v is not None})
    try:
        for epoch in range(start, cfg["epochs"]):
            G.train(); D.train()
            acc = dict(d_real=0.0, d_fake=0.0, g_adv=0.0, g_l1=0.0)
            nb, t0 = 0, time.time()
            for x, y, s in tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg['epochs']}", leave=False, disable=not verbose):
                x, y, s = x.to(device), y.to(device), s.to(device)
                fake = G(x, s)
                # --- discriminator
                pr, pf = D(x, y, s), D(x, fake.detach(), s)
                d_real, d_fake = BCE(pr, torch.ones_like(pr)), BCE(pf, torch.zeros_like(pf))
                oD.zero_grad(set_to_none=True)
                (0.5 * (d_real + d_fake)).backward()
                oD.step()
                # --- generator
                pf = D(x, fake, s)
                g_adv = BCE(pf, torch.ones_like(pf))
                g_l1 = F.l1_loss(fake, y)
                oG.zero_grad(set_to_none=True)
                (g_adv + cfg["lambda_l1"] * g_l1).backward()
                oG.step()
                for k, v in zip(acc, (d_real, d_fake, g_adv, g_l1)):
                    acc[k] += v.item()
                nb += 1
            acc = {k: v / nb for k, v in acc.items()}
            res = evaluate_gan(G, val_dl, device)
            obj = res["overall"]["objective"]
            if verbose:
                o = res["overall"]
                print(f"epoch {epoch + 1}: D_real {acc['d_real']:.3f} D_fake {acc['d_fake']:.3f} G_adv {acc['g_adv']:.3f} "
                      f"G_L1 {acc['g_l1']:.4f} | val L1 {o['l1']:.4f} PSNR {o['psnr']:.2f} SSIM {o['ssim']:.4f} | {time.time() - t0:.0f}s")
            if mlf:
                mlf.log_metrics({**{f"train/{k}": v for k, v in acc.items()},
                                 **{f"val/{k}": v for k, v in res["overall"].items() if k != "n"}}, step=epoch)
            if obj < best_obj:
                best_obj, best_res = obj, res
                best_state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
            if out:
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"cfg": cfg, "model": best_state}, out / "best.pt")
                torch.save({"cfg": cfg, "G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(),
                            "oD": oD.state_dict(), "epoch": epoch + 1, "best_obj": best_obj,
                            "best_state": best_state, "best_res": best_res}, out / "last.pt")
                if cfg["sample_every"] and ((epoch + 1) % cfg["sample_every"] == 0 or epoch == 0):
                    p = out / "samples" / f"epoch_{epoch + 1:03d}.png"
                    sample_grid(G, val_ds, fixed, device, p)
                    if mlf:
                        mlf.log_artifact(str(p), "samples")
            if trial is not None:
                trial.report(obj, epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        if out:
            cm.save_json({"cfg": cfg, "best_val": best_res}, out / "val_results.json")
            if mlf:
                mlf.log_artifact(str(out / "val_results.json")); mlf.log_artifact(str(out / "best.pt"))
    finally:
        if mlf:
            mlf.end_run()
    return best_res, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int); ap.add_argument("--subset-train", type=int)
    ap.add_argument("--mlflow-experiment"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = cm.load_json(a.config)["params"] if a.config else {}
    for k in ("epochs", "subset_train"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    train_gan(cfg, a.out, mlflow_experiment=a.mlflow_experiment, resume=a.resume)


if __name__ == "__main__":
    main()
