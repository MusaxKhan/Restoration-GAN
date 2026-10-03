"""Optuna study for the face-to-sketch cGAN (Task 4), run with a reduced epoch budget.

    python -m restoration.optuna_gan --trials 12 --epochs 10
Objective: minimise validation (L1 + (1 - SSIM)) of generated vs. ground-truth sketches."""
from __future__ import annotations

import argparse

from .optuna_utils import make_study, run_study, save_report
from .train_gan import train_gan

SPACE = {
    "g_lr": "loguniform[5e-5, 1e-3]", "d_lr": "loguniform[5e-5, 1e-3]", "batch_size": "categorical{8, 16, 32}",
    "base_ch": "categorical{32, 48, 64}", "dropout": "uniform[0.0, 0.5]",
    "style_dim": "categorical{8, 16, 32, 64}", "lambda_l1": "loguniform[10, 300]",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--subset-train", type=int, default=None)
    ap.add_argument("--out", default="runs/optuna")
    ap.add_argument("--config-out", default="configs/gan_best.json")
    a = ap.parse_args()
    study = make_study("gan", a.out, n_startup=3, warmup=3)

    def objective(trial):
        cfg = dict(g_lr=trial.suggest_float("g_lr", 5e-5, 1e-3, log=True),
                   d_lr=trial.suggest_float("d_lr", 5e-5, 1e-3, log=True),
                   batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
                   base_ch=trial.suggest_categorical("base_ch", [32, 48, 64]),
                   dropout=trial.suggest_float("dropout", 0.0, 0.5),
                   style_dim=trial.suggest_categorical("style_dim", [8, 16, 32, 64]),
                   lambda_l1=trial.suggest_float("lambda_l1", 10, 300, log=True),
                   epochs=a.epochs, subset_train=a.subset_train, sample_every=0)
        res, _ = train_gan(cfg, trial=trial, mlflow_experiment="optuna_gan", verbose=False)
        return res["overall"]["objective"]

    run_study(study, objective, a.trials)
    rep = save_report(study, SPACE, a.config_out, {"epochs_per_trial": a.epochs,
                                                   "objective": "val (L1 + (1 - SSIM)) on [0,1] sketches"})
    print(f"best value {rep['best_value']:.4f} params {rep['params']} -> {a.config_out}")


if __name__ == "__main__":
    main()
