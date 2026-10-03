"""Final test evaluation of the face-to-sketch generator on the official FS2K test set.

    python -m restoration.evaluate_task4 --ckpt runs/task4/best.pt --out results/task4"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import common as cm
from .data.fs2k import FS2KDataset
from .train_gan import build_G, evaluate_gan, sample_grid, to01
from .losses import ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results/task4")
    ap.add_argument("--subset", type=int, default=None)
    a = ap.parse_args()
    device = cm.get_device()
    ck = torch.load(a.ckpt, map_location=device)
    G = build_G(ck["cfg"]).to(device)
    G.load_state_dict(ck["model"])
    G.eval()
    ds = FS2KDataset("test", a.subset)
    dl = DataLoader(ds, 32)
    res = evaluate_gan(G, dl, device)
    cm.save_json({"cfg": ck["cfg"], "test": res}, f"{a.out}/test_results.json")

    # quality ranking -> best / worst examples, and a style-conditioning check on the same photos
    with torch.no_grad():
        scores = []
        for x, y, s in dl:
            x, y, s = x.to(device), y.to(device), s.to(device)
            scores.append(ssim(to01(G(x, s)), to01(y), per_sample=True).cpu())
    sc = torch.cat(scores).numpy()
    order = np.argsort(sc)
    pick = lambda idx: [int(i) for i in idx]
    sample_grid(G, ds, pick(order[::-1][:6]), device, f"{a.out}/best_examples.png")
    sample_grid(G, ds, pick(order[:6]), device, f"{a.out}/failure_cases.png")
    rng = np.random.default_rng(0)
    sample_grid(G, ds, pick(rng.choice(len(ds), 8, replace=False)), device, f"{a.out}/style_conditioning.png",
                all_styles=True)
    cm.save_json({"worst_ssim": [float(sc[i]) for i in order[:6]], "best_ssim": [float(sc[i]) for i in order[::-1][:6]]},
                 f"{a.out}/ranked_examples.json")
    o = res["overall"]
    print(f"test: L1 {o['l1']:.4f}  PSNR {o['psnr']:.2f}  SSIM {o['ssim']:.4f}  n={o['n']}")
    for k, v in res["by_style"].items():
        print(f"  {k}: L1 {v['l1']:.4f} PSNR {v['psnr']:.2f} SSIM {v['ssim']:.4f} n={v['n']}")


if __name__ == "__main__":
    main()
