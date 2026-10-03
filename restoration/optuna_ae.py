"""Optuna study for the autoencoders.

    Task 1 (universal AE):            python -m restoration.optuna_ae --mode universal --trials 20
    Task 2 (shared specialist arch):  python -m restoration.optuna_ae --mode specialist --trials 12

For the specialists a single shared search is run: every trial trains the three specialists
(salt / blur / occlusion) independently with the sampled hyper-parameters, and the objective is the
mean of their validation objectives. The best architecture is then used to train the final
specialists with `train_ae` (see assignment text).
"""
from __future__ import annotations

import argparse

import optuna

from . import common as cm
from .optuna_utils import make_study, run_study, save_report
from .train_ae import train_ae

SPACE_UNIVERSAL = {
    "lr": "loguniform[1e-4, 3e-3]", "batch_size": "categorical{32, 64, 128}",
    "latent_type": "categorical{dense, spatial}",
    "latent_dim (dense)": "categorical{128, 256, 512, 1024}", "latent_ch (spatial; dim = 64 x latent_ch)": "categorical{4, 8, 16, 32}",
    "base_ch": "categorical{16, 32, 48, 64}",
    "dropout": "uniform[0.0, 0.3]", "alpha": "uniform[0.5, 0.95]",
}
SPACE_SPECIALIST = {
    "lr": "loguniform[1e-4, 3e-3]", "batch_size": "categorical{32, 64, 128}",
    "latent_type": "categorical{dense, spatial}",
    "latent_dim (dense)": "categorical{128, 256, 512, 1024}", "latent_ch (spatial; dim = 64 x latent_ch)": "categorical{4, 8, 16, 32}",
    "base_ch": "categorical{16, 32, 48, 64}",
    "alpha": "uniform[0.5, 0.95]", "(fixed) dropout": 0.1,
}


def suggest(trial, mode):
    p = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
             batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
             latent_type=trial.suggest_categorical("latent_type", ["dense", "spatial"]),
             base_ch=trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
             alpha=trial.suggest_float("alpha", 0.5, 0.95))
    if p["latent_type"] == "dense":
        p["latent_dim"] = trial.suggest_categorical("latent_dim", [128, 256, 512, 1024])
    else:
        p["latent_ch"] = trial.suggest_categorical("latent_ch", [4, 8, 16, 32])
    p["dropout"] = trial.suggest_float("dropout", 0.0, 0.3) if mode == "universal" else 0.1
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["universal", "specialist"], required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=6, help="epochs per trial (reduced budget)")
    ap.add_argument("--subset-train", type=int, default=1500)
    ap.add_argument("--subset-val", type=int, default=None)
    ap.add_argument("--out", default="runs/optuna")
    ap.add_argument("--config-out")
    a = ap.parse_args()

    name = f"ae_{a.mode}"
    study = make_study(name, a.out)

    def objective(trial):
        p = suggest(trial, a.mode)
        base = dict(epochs=a.epochs, subset_train=a.subset_train, subset_val=a.subset_val, **p)
        exp = f"optuna_{name}"
        if a.mode == "universal":
            res, _ = train_ae({**base, "task": "universal"}, trial=trial, mlflow_experiment=exp, verbose=False)
            return res["overall"]["objective"]
        objs = []
        for i, task in enumerate(["salt", "blur", "occ"]):
            res, _ = train_ae({**base, "task": task}, trial=trial if i == 0 else None,
                              mlflow_experiment=exp, verbose=False)
            objs.append(res["overall"]["objective"])
            trial.set_user_attr(f"obj_{task}", objs[-1])
        return sum(objs) / 3

    run_study(study, objective, a.trials)
    space = SPACE_UNIVERSAL if a.mode == "universal" else SPACE_SPECIALIST
    path = a.config_out or f"configs/{name}_best.json"
    rep = save_report(study, space, path,
                      {"epochs_per_trial": a.epochs, "subset_train": a.subset_train,
                       "objective": "mean val (L1 + (1 - SSIM))"})
    print(f"best value {rep['best_value']:.4f} params {rep['params']} -> {path}")


if __name__ == "__main__":
    main()
