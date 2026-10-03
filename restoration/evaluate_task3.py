"""Final test evaluation of the soft MoE (Task 3): restoration quality and gate behaviour.

    python -m restoration.evaluate_task3 --ckpt runs/task3/best.pt --out results/task3
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import common as cm  # noqa: E402
from .data import corruptions as C  # noqa: E402
from .data.pets import PetsDataset  # noqa: E402
from .eval_utils import failure_indices, representative_indices  # noqa: E402
from .models.autoencoder import DenoisingAE  # noqa: E402
from .models.classifier import CorruptionClassifier  # noqa: E402
from .models.moe import SoftMoE  # noqa: E402
from .train_moe import collect_weights  # noqa: E402

BRANCHES = ["identity", "salt-expert", "blur-expert", "occ-expert"]


def load_moe(ckpt_path, device) -> SoftMoE:
    ck = torch.load(ckpt_path, map_location=device)
    g = ck["gate_cfg"]
    gate = CorruptionClassifier(g["base_ch"], g["depth"], g["dropout"])
    experts = [DenoisingAE(c["base_ch"], c["latent_dim"], c.get("dropout", 0.0)) for c in ck["expert_cfgs"]]
    moe = SoftMoE(gate, experts, ck["cfg"]["tau"])
    moe.load_state_dict(ck["model"])
    return moe.to(device).eval()


def weight_table(w, lab, lev):
    """Average routing weights for every true type (and every severity level)."""
    tab = {}
    for t in range(4):
        m = lab == t
        tab[C.CLASS_NAMES[t]] = w[m].mean(0).tolist()
        if t != C.CLEAN:
            for li in range(3):
                ml = m & (lev == li)
                if ml.any():
                    tab[f"{C.CLASS_NAMES[t]}/{cm.LEVELS[li]}"] = w[ml].mean(0).tolist()
    return tab


def plot_heatmap(tab, path):
    keys = list(tab)
    mat = np.array([tab[k] for k in keys])
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(keys) + 1.4))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(BRANCHES, rotation=25, ha="right")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8)
    for i in range(len(keys)):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 0.6 else "black")
    ax.set_title("Mean routing weight per true condition (test)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def plot_weight_examples(ds, w, indices, titles, path):
    fig, ax = plt.subplots(2, len(indices), figsize=(2.6 * len(indices), 5), squeeze=False,
                           gridspec_kw={"height_ratios": [3, 2]})
    for j, i in enumerate(indices):
        ax[0, j].imshow(ds[i][0].permute(1, 2, 0).numpy()); ax[0, j].axis("off")
        ax[0, j].set_title(titles[j], fontsize=8)
        ax[1, j].bar(range(4), w[i], color=["#888", "#d62728", "#1f77b4", "#2ca02c"])
        ax[1, j].set_ylim(0, 1); ax[1, j].set_xticks(range(4)); ax[1, j].set_xticklabels(["id", "s&p", "blur", "occ"], fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results/task3")
    ap.add_argument("--subset", type=int, default=None)
    a = ap.parse_args()
    device = cm.get_device()
    moe = load_moe(a.ckpt, device)
    ds = PetsDataset("test", subset=a.subset)
    dl = cm.make_loader(ds, 128)

    res, rows = cm.evaluate_restoration(lambda x: moe(x)[0], dl, device, return_rows=True)
    cm.save_json({"test": res}, f"{a.out}/test_results.json")
    w, lab, lev = collect_weights(moe, dl, device)
    tab = weight_table(w, lab, lev)
    dominant = {C.CLASS_NAMES[t]: [float((w[lab == t].argmax(1) == k).mean()) for k in range(4)] for t in range(4)}
    mean_all = w.mean(0)
    health = {"mean_weight_per_branch": mean_all.tolist(),
              "inactive_branches": [BRANCHES[k] for k in range(4) if mean_all[k] < 0.05],
              "routing_accuracy": float((w.argmax(1) == lab).mean()),
              "share_of_samples_where_branch_is_argmax": [float((w.argmax(1) == k).mean()) for k in range(4)],
              "argmax_share_by_true_type": dominant,
              "mean_entropy_nats": float(-(w * np.log(w + 1e-9)).sum(1).mean())}
    cm.save_json({"weights_by_condition": tab, "gate_health": health}, f"{a.out}/routing_analysis.json")
    plot_heatmap(tab, f"{a.out}/routing_heatmap.png")

    # examples: one expert clearly dominates vs weight spread over several branches
    ent = -(w * np.log(w + 1e-9)).sum(1)
    dom_idx = np.argsort(ent)[: 3]
    spread_idx = np.argsort(-ent)[: 3]
    idx = [int(i) for i in list(dom_idx) + list(spread_idx)]
    titles = [f"dominant: {C.CLASS_NAMES[int(lab[i])]}" for i in dom_idx] + \
             [f"spread: {C.CLASS_NAMES[int(lab[i])]}" for i in spread_idx]
    plot_weight_examples(ds, w, idx, titles, f"{a.out}/weight_examples.png")

    ex = representative_indices(rows)
    cm.restoration_grid(lambda x: moe(x)[0], ds, [i for i, _ in ex], device, f"{a.out}/examples.png")
    fl = failure_indices(rows)
    cm.restoration_grid(lambda x: moe(x)[0], ds, [i for i, _ in fl], device, f"{a.out}/failures.png")
    cm.save_json([{"index": i, "group": g, "psnr": float(rows[i, 0]), "ssim": float(rows[i, 1]),
                   "weights": w[i].tolist(), "spec": ds.entries[i]["spec"]} for i, g in fl],
                 f"{a.out}/failures.json")
    o = res["overall"]
    print(f"test: PSNR {o['psnr']:.2f} SSIM {o['ssim']:.4f}; routing acc {health['routing_accuracy']:.3f}; "
          f"inactive: {health['inactive_branches']}")


if __name__ == "__main__":
    main()
