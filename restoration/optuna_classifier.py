"""Optuna study for the corruption classifier (Task 2).

    python -m restoration.optuna_classifier --trials 20
Objective: minimise (1 - macro-F1) on the validation manifest."""
from __future__ import annotations

import argparse

from .optuna_utils import make_study, run_study, save_report
from .train_classifier import train_classifier

SPACE = {
    "lr": "loguniform[1e-4, 3e-3]", "batch_size": "categorical{32, 64, 128}",
    "base_ch": "categorical{16, 32, 48}", "depth": "categorical{3, 4, 5}",
    "dropout": "uniform[0.0, 0.5]", "weight_decay": "loguniform[1e-6, 1e-2]",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset-train", type=int, default=1500)
    ap.add_argument("--subset-val", type=int, default=None)
    ap.add_argument("--out", default="runs/optuna")
    ap.add_argument("--config-out", default="configs/cls_best.json")
    a = ap.parse_args()
    study = make_study("classifier", a.out)

    def objective(trial):
        p = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                 batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                 base_ch=trial.suggest_categorical("base_ch", [16, 32, 48]),
                 depth=trial.suggest_categorical("depth", [3, 4, 5]),
                 dropout=trial.suggest_float("dropout", 0.0, 0.5),
                 weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        rep, _ = train_classifier(dict(epochs=a.epochs, subset_train=a.subset_train, subset_val=a.subset_val, **p),
                                  trial=trial, mlflow_experiment="optuna_classifier", verbose=False)
        return 1.0 - rep["macro_f1"]

    run_study(study, objective, a.trials)
    rep = save_report(study, SPACE, a.config_out, {"epochs_per_trial": a.epochs, "subset_train": a.subset_train,
                                                   "objective": "1 - macro-F1 (validation)"})
    print(f"best (1-F1) {rep['best_value']:.4f} params {rep['params']} -> {a.config_out}")


if __name__ == "__main__":
    main()
