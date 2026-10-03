"""Train a denoising autoencoder: the universal model (Task 1) or one specialist (Task 2).

    python -m restoration.train_ae --task universal --config configs/ae_universal_best.json --out runs/task1
    python -m restoration.train_ae --task salt --config configs/ae_specialist_best.json --out runs/task2_salt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from . import common as cm
from .data import corruptions as C
from .data.pets import PetsDataset
from .losses import L1SSIMLoss
from .models.autoencoder import DenoisingAE

TASK_TYPES = {"universal": None, "salt": [C.SALT], "blur": [C.BLUR], "occ": [C.OCC]}

DEFAULTS = dict(task="universal", base_ch=32, latent_dim=512, latent_type="dense", latent_ch=16, dropout=0.1, lr=1e-3, batch_size=64,
                alpha=0.8, weight_decay=1e-5, epochs=30, subset_train=None, subset_val=None, workers=None)


def build_ae(cfg: dict) -> DenoisingAE:
    return DenoisingAE(cfg["base_ch"], cfg.get("latent_dim", 512), cfg.get("dropout", 0.0),
                       cfg.get("latent_type", "dense"), cfg.get("latent_ch", 16))


def train_ae(cfg: dict, out_dir: str | None = None, trial=None, mlflow_experiment: str | None = None,
             resume: bool = False, verbose: bool = True):
    """Train and return (best_val_result, best_state_dict). Saves best.pt/last.pt when out_dir given."""
    cfg = {**DEFAULTS, **cfg}
    cm.seed_everything(42)
    device = cm.get_device()
    types = TASK_TYPES[cfg["task"]]
    train_ds = PetsDataset("train", types, cfg["subset_train"])
    val_ds = PetsDataset("val", types, cfg["subset_val"])
    # specialists' validation sets are small (~180 images): evaluate on the whole filtered manifest
    train_dl = cm.make_loader(train_ds, cfg["batch_size"], shuffle=True, workers=cfg["workers"])
    val_dl = cm.make_loader(val_ds, 128, workers=cfg["workers"])

    model = build_ae(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = L1SSIMLoss(cfg["alpha"])

    out = Path(out_dir) if out_dir else None
    start_epoch, best_obj, best_state, best_res = 0, float("inf"), None, None
    if out and resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best_obj, best_state, best_res = ck["epoch"], ck["best_obj"], ck["best_state"], ck["best_res"]
        print(f"resumed from epoch {start_epoch}")

    mlf = run = None
    if mlflow_experiment:
        mlf = cm.mlflow_setup(mlflow_experiment)
        run = mlf.start_run(run_name=f"{cfg['task']}_{int(time.time())}")
        mlf.log_params({k: v for k, v in cfg.items() if v is not None})
        mlf.log_param("n_params", sum(p.numel() for p in model.parameters()))

    try:
        for epoch in range(start_epoch, cfg["epochs"]):
            model.train()
            t0, tot, nb = time.time(), 0.0, 0
            it = tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg['epochs']}", leave=False, disable=not verbose)
            for x, y, _, _ in it:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with cm.autocast(device):
                    pred = model(x)
                loss, l1, s = loss_fn(pred.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item(); nb += 1
            model.eval()
            res = cm.evaluate_restoration(model, val_dl, device)
            obj = res["overall"]["objective"]
            if verbose:
                o = res["overall"]
                print(f"epoch {epoch + 1}: train_loss {tot / nb:.4f} | val psnr {o['psnr']:.2f} ssim {o['ssim']:.4f} "
                      f"obj {obj:.4f} | {time.time() - t0:.0f}s")
            if mlf:
                mlf.log_metrics({"train/loss": tot / nb, "lr": sched.get_last_lr()[0],
                                 **cm.flat_metrics("val", res)}, step=epoch)
            if obj < best_obj:
                best_obj, best_res = obj, res
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out and ((epoch + 1) % cfg.get("ckpt_every", 10) == 0 or epoch + 1 == cfg["epochs"]):
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"cfg": cfg, "model": best_state}, out / "best.pt")
                torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": epoch + 1, "best_obj": best_obj,
                            "best_state": best_state, "best_res": best_res}, out / "last.pt")
            if trial is not None:
                trial.report(obj, epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

        if out:
            model.load_state_dict(best_state)
            model.eval()
            idx = list(range(0, min(len(val_ds), 96), 8))[:12]
            cm.restoration_grid(model, val_ds, idx, device, out / "val_samples.png")
            cm.save_json({"cfg": cfg, "best_val": best_res}, out / "val_results.json")
            if mlf:
                mlf.log_artifact(str(out / "val_samples.png"))
                mlf.log_artifact(str(out / "val_results.json"))
                mlf.log_artifact(str(out / "best.pt"))
    finally:
        if mlf:
            mlf.end_run()
    return best_res, best_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_TYPES), required=True)
    p.add_argument("--config", help="JSON file with hyperparameters (e.g. Optuna best)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--subset-train", type=int)
    p.add_argument("--subset-val", type=int)
    p.add_argument("--mlflow-experiment")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = cm.load_json(a.config)["params"] if a.config else {}
    cfg["task"] = a.task
    if a.epochs:
        cfg["epochs"] = a.epochs
    if a.subset_train:
        cfg["subset_train"] = a.subset_train
    if a.subset_val:
        cfg["subset_val"] = a.subset_val
    train_ae(cfg, a.out, mlflow_experiment=a.mlflow_experiment, resume=a.resume)


if __name__ == "__main__":
    main()
