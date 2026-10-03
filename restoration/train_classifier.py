"""Task 2: corruption classifier (clean / salt-and-pepper / blur / occlusion).

Training batches are exactly class-balanced (BalancedBatchSampler); labels come from the runtime
corruption pipeline. Loss: multiclass cross-entropy.

    python -m restoration.train_classifier --config configs/cls_best.json --out runs/task2_cls
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from . import common as cm
from .data import corruptions as C
from .data.pets import BalancedBatchSampler, PetsDataset
from .models.classifier import CorruptionClassifier

DEFAULTS = dict(base_ch=32, depth=4, dropout=0.3, lr=1e-3, batch_size=64, weight_decay=1e-4, epochs=15,
                subset_train=None, subset_val=None, workers=None)


def build_classifier(cfg):
    return CorruptionClassifier(cfg["base_ch"], cfg["depth"], cfg["dropout"])


@torch.no_grad()
def predict(model, loader, device):
    """Return (probabilities, labels, levels) over a manifest loader."""
    model.eval()
    P, L, V = [], [], []
    for x, _, lab, lev in loader:
        P.append(torch.softmax(model(x.to(device)).float(), 1).cpu()); L.append(lab); V.append(lev)
    return torch.cat(P).numpy(), torch.cat(L).numpy(), torch.cat(V).numpy()


def classification_report(probs, labels, levels) -> dict:
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(labels, pred, labels=[0, 1, 2, 3], zero_division=0)
    cmat = confusion_matrix(labels, pred, labels=[0, 1, 2, 3], normalize="true")
    rep = {"accuracy": float((pred == labels).mean()), "macro_precision": float(p.mean()),
           "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
           "per_class": {C.CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                            "support": int(s[i])} for i in range(4)},
           "confusion_matrix_normalized": cmat.tolist(), "accuracy_by_type_level": {}}
    for t in range(1, 4):
        for li in range(3):
            m = (labels == t) & (levels == li)
            if m.any():
                rep["accuracy_by_type_level"][f"{C.CLASS_NAMES[t]}/{cm.LEVELS[li]}"] = float((pred[m] == t).mean())
    return rep


def train_classifier(cfg, out_dir=None, trial=None, mlflow_experiment=None, resume=False, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    cm.seed_everything(42)
    device = cm.get_device()
    train_ds = PetsDataset("train", None, cfg["subset_train"])
    val_ds = PetsDataset("val", None, cfg["subset_val"])
    sampler = BalancedBatchSampler(len(train_ds), cfg["batch_size"])
    train_dl = cm.make_loader(train_ds, None, batch_sampler=sampler, workers=cfg["workers"])
    val_dl = cm.make_loader(val_ds, 128, workers=cfg["workers"])

    model = build_classifier(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(train_dl),
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out = Path(out_dir) if out_dir else None
    start, best_f1, best_state, best_rep = 0, -1.0, None, None
    if out and resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_f1, best_state, best_rep = ck["epoch"], ck["best_f1"], ck["best_state"], ck["best_rep"]

    mlf = None
    if mlflow_experiment:
        mlf = cm.mlflow_setup(mlflow_experiment)
        mlf.start_run(run_name=f"classifier_{int(time.time())}")
        mlf.log_params({k: v for k, v in cfg.items() if v is not None})
    try:
        for epoch in range(start, cfg["epochs"]):
            model.train()
            tot, nb, t0 = 0.0, 0, time.time()
            for x, _, lab, _ in tqdm(train_dl, desc=f"epoch {epoch + 1}", leave=False, disable=not verbose):
                x, lab = x.to(device, non_blocking=True), lab.to(device, non_blocking=True)
                with cm.autocast(device):
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), lab)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item(); nb += 1
            rep = classification_report(*predict(model, val_dl, device))
            if verbose:
                print(f"epoch {epoch + 1}: loss {tot / nb:.4f} | val acc {rep['accuracy']:.4f} "
                      f"macroF1 {rep['macro_f1']:.4f} | {time.time() - t0:.0f}s")
            if mlf:
                mlf.log_metrics({"train/loss": tot / nb, "val/accuracy": rep["accuracy"],
                                 "val/macro_f1": rep["macro_f1"], "val/macro_precision": rep["macro_precision"],
                                 "val/macro_recall": rep["macro_recall"]}, step=epoch)
            if rep["macro_f1"] > best_f1:
                best_f1, best_rep = rep["macro_f1"], rep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out:
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"cfg": cfg, "model": best_state}, out / "best.pt")
                torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": epoch + 1, "best_f1": best_f1,
                            "best_state": best_state, "best_rep": best_rep}, out / "last.pt")
            if trial is not None:
                trial.report(1 - rep["macro_f1"], epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        if out:
            cm.save_json({"cfg": cfg, "best_val": best_rep}, out / "val_results.json")
            if mlf:
                mlf.log_artifact(str(out / "val_results.json")); mlf.log_artifact(str(out / "best.pt"))
    finally:
        if mlf:
            mlf.end_run()
    return best_rep, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int); ap.add_argument("--subset-train", type=int)
    ap.add_argument("--subset-val", type=int); ap.add_argument("--mlflow-experiment"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = cm.load_json(a.config)["params"] if a.config else {}
    for k in ("epochs", "subset_train", "subset_val"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    train_classifier(cfg, a.out, mlflow_experiment=a.mlflow_experiment, resume=a.resume)


if __name__ == "__main__":
    main()
