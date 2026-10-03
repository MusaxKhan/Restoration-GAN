"""Task 3: jointly trained soft mixture-of-experts restoration.

Gate = classifier from Task 2, experts = the three Task 2 specialists, plus an identity branch.
Stage 1 (warm-up): experts frozen, only the gate is trained.  Stage 2: everything is fine-tuned jointly
with a smaller learning rate.

    L = l_l1*L1 + l_ssim*(1-SSIM) + l_ce*CE(G(x)/tau, label) + l_bal*sum_k (mean_w_k - 1/4)^2

    python -m restoration.train_moe --cls runs/task2_cls/best.pt --salt ... --blur ... --occ ... \
        --config configs/moe_best.json --out runs/task3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import common as cm
from .data import corruptions as C
from .data.pets import BalancedBatchSampler, PetsDataset
from .evaluate_task2 import load_router
from .losses import ssim
from .models.moe import SoftMoE

DEFAULTS = dict(lr=3e-4, tau=1.0, lambda_ce=0.1, lambda_bal=0.01, alpha=0.8, warmup_epochs=2, epochs=10,
                batch_size=64, warmup_lr_mult=3.0, weight_decay=1e-5, subset_train=None, subset_val=None,
                workers=None)


def build_moe(paths: dict, tau: float, device) -> SoftMoE:
    r = load_router(paths["cls"], paths["salt"], paths["blur"], paths["occ"], device)
    return SoftMoE(r.classifier, list(r.experts), tau).to(device)


@torch.no_grad()
def collect_weights(moe, loader, device):
    """Routing weights (N,4) for every sample of a manifest loader, with labels and levels."""
    moe.eval()
    W, L, V = [], [], []
    for x, _, lab, lev in loader:
        W.append(moe(x.to(device))[2].float().cpu()); L.append(lab); V.append(lev)
    return torch.cat(W).numpy(), torch.cat(L).numpy(), torch.cat(V).numpy()


def balance_loss(w):
    """sum_k (mean_batch(w_k) - 1/4)^2  (Shazeer et al. 2017 style load balancing, squared-error form)."""
    return ((w.mean(0) - 0.25) ** 2).sum()


def joint_loss(recon, y, logits, w, lab, cfg):
    l1 = F.l1_loss(recon, y)
    s = ssim(recon, y)
    ce = F.cross_entropy(logits / cfg["tau"], lab)
    bal = balance_loss(w)
    a = cfg["alpha"]
    total = a * l1 + (1 - a) * (1 - s) + cfg["lambda_ce"] * ce + cfg["lambda_bal"] * bal
    return total, dict(l1=l1.item(), ssim=s.item(), ce=ce.item(), bal=bal.item())


def train_moe(paths: dict, cfg: dict, out_dir=None, trial=None, mlflow_experiment=None, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    cm.seed_everything(42)
    device = cm.get_device()
    moe = build_moe(paths, cfg["tau"], device)
    train_ds = PetsDataset("train", None, cfg["subset_train"])
    val_ds = PetsDataset("val", None, cfg["subset_val"])
    train_dl = cm.make_loader(train_ds, None, batch_sampler=BalancedBatchSampler(len(train_ds), cfg["batch_size"]),
                              workers=cfg["workers"])
    val_dl = cm.make_loader(val_ds, 128, workers=cfg["workers"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out = Path(out_dir) if out_dir else None

    def val_eval():
        res = cm.evaluate_restoration(lambda x: moe(x)[0], val_dl, device)
        w, lab, _ = collect_weights(moe, val_dl, device)
        res["mean_weights"] = w.mean(0).tolist()
        res["routing_accuracy"] = float((w.argmax(1) == lab).mean())
        return res

    init = val_eval()  # state right after initialisation from Task 2 (before any joint training)
    if verbose:
        print(f"init: PSNR {init['overall']['psnr']:.2f} SSIM {init['overall']['ssim']:.4f} "
              f"routing acc {init['routing_accuracy']:.3f}")
    mlf = None
    if mlflow_experiment:
        mlf = cm.mlflow_setup(mlflow_experiment)
        mlf.start_run(run_name=f"moe_{int(time.time())}")
        mlf.log_params({k: v for k, v in cfg.items() if v is not None})
        mlf.log_metrics({"init/psnr": init["overall"]["psnr"], "init/ssim": init["overall"]["ssim"]})

    best_obj, best_state, best_res = float("inf"), None, None
    opt = None
    try:
        for epoch in range(cfg["epochs"]):
            warm = epoch < cfg["warmup_epochs"]
            if epoch == 0 or epoch == cfg["warmup_epochs"]:
                for p in moe.experts.parameters():
                    p.requires_grad_(not warm)
                params = [p for p in moe.parameters() if p.requires_grad]
                lr = cfg["lr"] * (cfg["warmup_lr_mult"] if warm else 1.0)
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
                n_ep = cfg["warmup_epochs"] if warm else cfg["epochs"] - cfg["warmup_epochs"]
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_ep * len(train_dl)))
            moe.train()
            if warm:
                moe.experts.eval()  # frozen experts: keep BatchNorm statistics fixed too
            tot, parts, nb, t0 = 0.0, {}, 0, time.time()
            for x, y, lab, _ in tqdm(train_dl, desc=f"epoch {epoch + 1} ({'warm-up' if warm else 'joint'})",
                                     leave=False, disable=not verbose):
                x, y, lab = x.to(device, non_blocking=True), y.to(device, non_blocking=True), lab.to(device)
                with cm.autocast(device):
                    recon, logits, w, _ = moe(x)
                loss, p = joint_loss(recon.float(), y, logits.float(), w.float(), lab, cfg)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([q for q in moe.parameters() if q.requires_grad], 5.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item(); nb += 1
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v
            res = val_eval()
            obj = res["overall"]["objective"]
            mw = res["mean_weights"]
            if verbose:
                print(f"epoch {epoch + 1} [{'warm' if warm else 'joint'}]: loss {tot / nb:.4f} | val PSNR "
                      f"{res['overall']['psnr']:.2f} SSIM {res['overall']['ssim']:.4f} obj {obj:.4f} | "
                      f"route-acc {res['routing_accuracy']:.3f} w={np.round(mw, 3).tolist()} | {time.time() - t0:.0f}s")
            if mlf:
                mlf.log_metrics({"train/loss": tot / nb, **{f"train/{k}": v / nb for k, v in parts.items()},
                                 **cm.flat_metrics("val", res), "val/routing_accuracy": res["routing_accuracy"],
                                 **{f"val/mean_w_{n}": mw[i] for i, n in enumerate(C.CLASS_NAMES)}}, step=epoch)
            if obj < best_obj:
                best_obj, best_res = obj, res
                best_state = {k: v.detach().cpu().clone() for k, v in moe.state_dict().items()}
            if out and ((epoch + 1) % cfg.get("ckpt_every", 10) == 0 or epoch + 1 == cfg["epochs"]):
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"cfg": cfg, "paths": paths, "model": best_state,
                            "gate_cfg": torch.load(paths["cls"], map_location="cpu")["cfg"],
                            "expert_cfgs": [torch.load(paths[k], map_location="cpu")["cfg"]
                                            for k in ("salt", "blur", "occ")]}, out / "best.pt")
            if trial is not None:
                trial.report(obj, epoch)
                if min(mw) < 0.02 or max(mw) > 0.85:  # routing collapse: a branch unused / one branch dominates
                    import optuna
                    raise optuna.TrialPruned()
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        if out:
            cm.save_json({"cfg": cfg, "init_val": init, "best_val": best_res}, out / "val_results.json")
            if mlf:
                mlf.log_artifact(str(out / "val_results.json")); mlf.log_artifact(str(out / "best.pt"))
    finally:
        if mlf:
            mlf.end_run()
    return best_res, best_state, init


def main():
    ap = argparse.ArgumentParser()
    for k in ("cls", "salt", "blur", "occ"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--config"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int); ap.add_argument("--subset-train", type=int)
    ap.add_argument("--subset-val", type=int); ap.add_argument("--mlflow-experiment")
    a = ap.parse_args()
    cfg = cm.load_json(a.config)["params"] if a.config else {}
    for k in ("epochs", "subset_train", "subset_val"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    train_moe({k: getattr(a, k) for k in ("cls", "salt", "blur", "occ")}, cfg, a.out,
              mlflow_experiment=a.mlflow_experiment)


if __name__ == "__main__":
    main()
