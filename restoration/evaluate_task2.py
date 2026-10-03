"""Final test evaluation of Task 2: classifier, oracle-routed and predicted-routed restoration.

    python -m restoration.evaluate_task2 --cls runs/task2_cls/best.pt \
        --salt runs/task2_salt/best.pt --blur runs/task2_blur/best.pt --occ runs/task2_occ/best.pt \
        --out results/task2
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
from .routing import HardRouter  # noqa: E402
from .train_ae import build_ae  # noqa: E402
from .train_classifier import build_classifier, classification_report, predict  # noqa: E402


def load_router(cls_p, salt_p, blur_p, occ_p, device) -> HardRouter:
    ck = torch.load(cls_p, map_location=device)
    cls = build_classifier(ck["cfg"]).to(device)
    cls.load_state_dict(ck["model"])
    experts = []
    for p in (salt_p, blur_p, occ_p):
        c = torch.load(p, map_location=device)
        m = build_ae(c["cfg"]).to(device)
        m.load_state_dict(c["model"])
        experts.append(m)
    return HardRouter(cls, experts).eval()


def plot_confusion(cmat, path, title="Normalized confusion matrix (test)"):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cmat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(C.CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(C.CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmat[i][j]:.2f}", ha="center", va="center",
                    color="white" if cmat[i][j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    for k in ("cls", "salt", "blur", "occ"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default="results/task2")
    ap.add_argument("--subset", type=int, default=None)
    a = ap.parse_args()
    device = cm.get_device()
    router = load_router(a.cls, a.salt, a.blur, a.occ, device)
    ds = PetsDataset("test", subset=a.subset)
    dl = cm.make_loader(ds, 128)

    # --- classifier
    probs, labels, levels = predict(router.classifier, dl, device)
    rep = classification_report(probs, labels, levels)
    cm.save_json(rep, f"{a.out}/classifier_test.json")
    plot_confusion(np.array(rep["confusion_matrix_normalized"]), f"{a.out}/confusion_matrix.png")
    print(f"classifier: acc {rep['accuracy']:.4f} macroF1 {rep['macro_f1']:.4f}")

    # --- restoration, both routing modes
    res_o, rows_o = cm.evaluate_restoration(lambda x, l: router(x, route=l)[0], dl, device,
                                            return_rows=True, needs_labels=True)
    res_p, rows_p = cm.evaluate_restoration(lambda x: router(x)[0], dl, device, return_rows=True)
    cm.save_json({"oracle": res_o, "predicted": res_p}, f"{a.out}/restoration_test.json")
    print(f"oracle    PSNR {res_o['overall']['psnr']:.2f} SSIM {res_o['overall']['ssim']:.4f}")
    print(f"predicted PSNR {res_p['overall']['psnr']:.2f} SSIM {res_p['overall']['ssim']:.4f}"
          f"  (input PSNR {res_p['overall']['input_psnr']:.2f})")

    # --- routing failures: misrouted samples with the largest quality loss versus oracle routing
    pred = probs.argmax(1)
    drop = rows_o[:, 0] - rows_p[:, 0]
    wrong = np.where(pred != labels)[0]
    worst = wrong[np.argsort(-drop[wrong])][:4]
    info = [{"index": int(i), "true": C.CLASS_NAMES[labels[i]], "predicted": C.CLASS_NAMES[pred[i]],
             "psnr_oracle": float(rows_o[i, 0]), "psnr_predicted": float(rows_p[i, 0]),
             "p_true": float(probs[i, labels[i]]), "spec": ds.entries[i]["spec"]} for i in worst]
    cm.save_json({"n_misrouted": int(len(wrong)), "mean_psnr_drop_when_misrouted": float(drop[wrong].mean())
                  if len(wrong) else 0.0, "worst_cases": info}, f"{a.out}/routing_failures.json")
    if len(worst):
        xs = torch.stack([ds[int(i)][0] for i in worst]).to(device)
        ys = torch.stack([ds[int(i)][1] for i in worst]).to(device)
        lab_t = torch.tensor(labels[worst], device=device)
        out_o, out_p = router(xs, route=lab_t)[0], router(xs)[0]
        fig, ax = plt.subplots(4, len(worst), figsize=(2.6 * len(worst), 10), squeeze=False)
        for j in range(len(worst)):
            for i, (im, t) in enumerate([(xs, "input"), (out_o, "oracle routing"), (out_p, "predicted routing"),
                                         (ys, "clean")]):
                ax[i, j].imshow(im[j].permute(1, 2, 0).clamp(0, 1).cpu().numpy()); ax[i, j].axis("off")
                ax[i, j].set_title(t if i != 2 else f"{t}\n({C.CLASS_NAMES[pred[worst[j]]]})", fontsize=8)
        plt.tight_layout(); plt.savefig(f"{a.out}/routing_failures.png", dpi=90); plt.close(fig)

    # --- examples (predicted routing)
    from .eval_utils import representative_indices
    ex = representative_indices(rows_p)
    cm.restoration_grid(lambda x: router(x)[0], ds, [i for i, _ in ex], device, f"{a.out}/examples.png")


if __name__ == "__main__":
    main()
