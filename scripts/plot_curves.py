"""Plot training/validation curves of the final runs from the MLflow SQLite database.

    python scripts/plot_curves.py --db experiments/mlflow.db --out report/figures
"""
import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(db, experiment):
    """Metrics of the latest run of an experiment: {key: [(step, value), ...]}."""
    c = sqlite3.connect(db)
    row = c.execute("select e.experiment_id from experiments e where e.name=?", (experiment,)).fetchone()
    if not row:
        print("missing experiment", experiment)
        return {}
    run = c.execute("select run_uuid from runs where experiment_id=? order by start_time desc limit 1", (row[0],)).fetchone()
    m = defaultdict(list)
    for k, v, s in c.execute("select key, value, step from metrics where run_uuid=? order by step", (run[0],)):
        m[k].append((s, v))
    return m


def line(ax, m, key, label=None, **kw):
    if key in m:
        xs, ys = zip(*m[key])
        ax.plot(xs, ys, label=label or key, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="experiments/mlflow.db")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    m = load(a.db, "v2_task1_universal_ae")
    line(ax[0], m, "train/loss", "train loss"); ax[0].set_title("Task 1: training loss"); ax[0].set_xlabel("epoch")
    line(ax[1], m, "val/psnr", "val PSNR (dB)"); ax[1].set_title("Task 1: validation PSNR"); ax[1].set_xlabel("epoch")
    line(ax[2], m, "val/ssim", "val SSIM"); ax[2].set_title("Task 1: validation SSIM"); ax[2].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(out / "curves_task1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    m = load(a.db, "task2_classifier")
    line(ax[0], m, "train/loss", "train CE loss"); ax[0].set_title("Task 2 classifier: loss"); ax[0].set_xlabel("epoch")
    line(ax[1], m, "val/accuracy", "accuracy"); line(ax[1], m, "val/macro_f1", "macro-F1")
    ax[1].legend(); ax[1].set_title("Task 2 classifier: validation"); ax[1].set_xlabel("epoch")
    for t in ("salt", "blur", "occ"):
        line(ax[2], load(a.db, f"v2_task2_specialist_{t}"), "val/psnr", t)
    ax[2].legend(); ax[2].set_title("Task 2 specialists: val PSNR (dB)"); ax[2].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(out / "curves_task2.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    m = load(a.db, "v2_task3_soft_moe")
    line(ax[0], m, "train/loss", "total"); line(ax[0], m, "train/l1", "L1"); line(ax[0], m, "train/ce", "CE")
    ax[0].legend(); ax[0].set_title("Task 3: training losses"); ax[0].set_xlabel("epoch (1 = warm-up, then joint)")
    line(ax[1], m, "val/psnr", "val PSNR (dB)"); ax[1].set_title("Task 3: validation PSNR"); ax[1].set_xlabel("epoch")
    for k, n in enumerate(["clean", "salt_pepper", "blur", "occlusion"]):
        line(ax[2], m, f"val/mean_w_{n}", ["identity", "salt expert", "blur expert", "occ. expert"][k])
    ax[2].legend(fontsize=7); ax[2].set_title("Task 3: mean gate weights (val)"); ax[2].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(out / "curves_task3.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    m = load(a.db, "task4_face_to_sketch")
    line(ax[0], m, "train/d_real", "D real"); line(ax[0], m, "train/d_fake", "D fake"); line(ax[0], m, "train/g_adv", "G adversarial")
    ax[0].legend(fontsize=7); ax[0].set_title("Task 4: adversarial losses"); ax[0].set_xlabel("epoch")
    line(ax[1], m, "train/g_l1", "G L1 (train)"); line(ax[1], m, "val/l1", "L1 (val)")
    ax[1].legend(); ax[1].set_title("Task 4: reconstruction"); ax[1].set_xlabel("epoch")
    line(ax[2], m, "val/ssim", "val SSIM", color="tab:blue"); ax[2].set_ylabel("SSIM", color="tab:blue")
    ax2 = ax[2].twinx(); line(ax2, m, "val/psnr", "val PSNR (dB)", color="tab:orange"); ax2.set_ylabel("PSNR (dB)", color="tab:orange")
    ax[2].set_title("Task 4: validation quality"); ax[2].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(out / "curves_task4.png", dpi=130); plt.close(fig)
    print("curves written to", out)


if __name__ == "__main__":
    main()
