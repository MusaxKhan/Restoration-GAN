"""Optuna study for the joint fine-tuning of the soft mixture-of-experts (Task 3).

    python -m restoration.optuna_moe --cls ... --salt ... --blur ... --occ ... --trials 15
Trials are pruned on poor intermediate objective or on routing collapse (see train_moe)."""
from __future__ import annotations

import argparse

from .optuna_utils import make_study, run_study, save_report
from .train_moe import train_moe

SPACE = {
    "lr": "loguniform[1e-5, 1e-3]", "tau": "loguniform[0.5, 2.0]", "lambda_ce": "loguniform[0.01, 1.0]",
    "lambda_bal": "loguniform[1e-3, 0.1]", "alpha (L1 weight; SSIM weight = 1 - alpha)": "uniform[0.5, 0.95]",
}


def main():
    ap = argparse.ArgumentParser()
    for k in ("cls", "salt", "blur", "occ"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--subset-train", type=int, default=1500)
    ap.add_argument("--subset-val", type=int, default=None)
    ap.add_argument("--out", default="runs/optuna")
    ap.add_argument("--config-out", default="configs/moe_best.json")
    a = ap.parse_args()
    paths = {k: getattr(a, k) for k in ("cls", "salt", "blur", "occ")}
    study = make_study("moe", a.out, n_startup=3, warmup=1)

    def objective(trial):
        cfg = dict(lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                   tau=trial.suggest_float("tau", 0.5, 2.0, log=True),
                   lambda_ce=trial.suggest_float("lambda_ce", 0.01, 1.0, log=True),
                   lambda_bal=trial.suggest_float("lambda_bal", 1e-3, 0.1, log=True),
                   alpha=trial.suggest_float("alpha", 0.5, 0.95),
                   epochs=a.epochs, warmup_epochs=a.warmup_epochs, subset_train=a.subset_train,
                   subset_val=a.subset_val)
        res, _, _ = train_moe(paths, cfg, trial=trial, mlflow_experiment="optuna_moe", verbose=False)
        return res["overall"]["objective"]

    run_study(study, objective, a.trials)
    rep = save_report(study, SPACE, a.config_out,
                      {"epochs_per_trial": a.epochs, "warmup_epochs": a.warmup_epochs,
                       "subset_train": a.subset_train, "objective": "val (L1 + (1 - SSIM)); prune on collapse"})
    print(f"best value {rep['best_value']:.4f} params {rep['params']} -> {a.config_out}")


if __name__ == "__main__":
    main()
