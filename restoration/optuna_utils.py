"""Optuna helpers shared by all four tasks: study creation (resumable) and report export."""
from __future__ import annotations

from pathlib import Path

import optuna

from . import common as cm


def make_study(name: str, out_dir: str, direction: str = "minimize", n_startup: int = 3, warmup: int = 2):
    """SQLite-backed study so an interrupted Colab session can resume (`load_if_exists`)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=name, storage=f"sqlite:///{Path(out_dir) / (name + '.db')}", direction=direction,
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=warmup))


def run_study(study, objective, n_trials: int):
    """Run until the study holds `n_trials` finished trials (counting those loaded from disk)."""
    done = len([t for t in study.trials if t.state.is_finished()])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, gc_after_trial=True)


def save_report(study, search_space: dict, config_path: str, extra: dict | None = None):
    """Write best params + the full search space + trial table + plots to disk."""
    states = [t.state.name for t in study.trials]
    best = study.best_trial
    report = {
        "study": study.study_name, "direction": study.direction.name, "search_space": search_space,
        "n_trials": len(states), "n_complete": states.count("COMPLETE"), "n_pruned": states.count("PRUNED"),
        "n_failed": states.count("FAIL"), "best_trial": best.number, "best_value": best.value,
        "params": best.params, "trials": [
            {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
            for t in study.trials],
    }
    if extra:
        report.update(extra)
    cm.save_json(report, config_path)
    stem = str(Path(config_path).with_suffix(""))
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        plot_optimization_history(study); plt.tight_layout(); plt.savefig(stem + "_history.png", dpi=100); plt.close()
        plot_param_importances(study); plt.tight_layout(); plt.savefig(stem + "_importance.png", dpi=100); plt.close()
    except Exception as e:  # importance needs >=2 completed trials
        print("could not draw optuna plots:", e)
    return report
