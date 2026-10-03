"""Final test evaluation of the universal autoencoder (Task 1) on the fixed test manifest.

    python -m restoration.evaluate_task1 --ckpt runs/task1/best.pt --out results/task1
"""
from __future__ import annotations

import argparse

import torch

from . import common as cm
from .data.pets import PetsDataset
from .eval_utils import failure_indices, representative_indices
from .train_ae import build_ae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results/task1")
    ap.add_argument("--subset", type=int, default=None, help="evaluate only the first N manifest entries")
    a = ap.parse_args()
    device = cm.get_device()
    ck = torch.load(a.ckpt, map_location=device)
    model = build_ae(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = PetsDataset("test", subset=a.subset)
    res, rows = cm.evaluate_restoration(model, cm.make_loader(ds, 128), device, return_rows=True)
    cm.save_json({"cfg": ck["cfg"], "test": res}, f"{a.out}/test_results.json")

    ex = representative_indices(rows)
    cm.restoration_grid(model, ds, [i for i, _ in ex], device, f"{a.out}/examples.png")
    cm.save_json([{"index": i, "group": g} for i, g in ex], f"{a.out}/examples.json")
    fl = failure_indices(rows)
    cm.restoration_grid(model, ds, [i for i, _ in fl], device, f"{a.out}/failures.png")
    cm.save_json([{"index": i, "group": g, "psnr": float(rows[i, 0]), "ssim": float(rows[i, 1]),
                   "spec": ds.entries[i]["spec"]} for i, g in fl], f"{a.out}/failures.json")
    o = res["overall"]
    print(f"test overall: PSNR {o['psnr']:.2f} dB (input {o['input_psnr']:.2f}), SSIM {o['ssim']:.4f} "
          f"(input {o['input_ssim']:.4f})")
    for k, v in res["by_type_level"].items():
        print(f"  {k:22s} PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.4f}  n={v['n']}")


if __name__ == "__main__":
    main()
