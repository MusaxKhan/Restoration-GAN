"""Architecture diagrams for the report (dimensions taken from restoration/models/*.py for the selected configs).

    python scripts/make_arch_figures.py     -> report/figures/arch_ae.png, arch_routing.png, arch_gan.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "report" / "figures"
BLUE, GREEN, ORANGE, GREY, PURPLE = "#cfe2ff", "#d1f2d9", "#ffe5b4", "#e9ecef", "#e5d4f7"


def box(ax, x, y, w, h, text, color=GREY, fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=color, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, ls="-", color="#333", lw=0.9, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def fig_ae():
    fig, ax = canvas(7.3, 2.6, 19.5, 6.2)
    enc = [("Input\n128x128x3", BLUE), ("Down 1\n64x64x32", GREEN), ("Down 2\n32x32x64", GREEN),
           ("Down 3\n16x16x128", GREEN), ("Down 4\n8x8x256", GREEN)]
    xs = [0.2 + 2.5 * i for i in range(5)]
    for (t, c), x in zip(enc, xs):
        box(ax, x, 3.4, 2.1, 1.6, t, c)
    for i in range(4):
        arrow(ax, xs[i] + 2.1, 4.2, xs[i + 1], 4.2)
    box(ax, 12.7, 3.4, 3.2, 1.6, "1x1 conv latent\n8x8x32\n= 2,048 numbers", ORANGE, bold=True)
    arrow(ax, xs[-1] + 2.1, 4.2, 12.7, 4.2)
    box(ax, 16.9, 3.4, 2.3, 1.6, "1x1 conv\n8x8x256", GREEN)
    arrow(ax, 15.9, 4.2, 16.9, 4.2)
    dec = [("Up 1\n16x16x128", GREEN), ("Up 2\n32x32x64", GREEN), ("Up 3\n64x64x32", GREEN), ("Up 4\n128x128x32", GREEN),
           ("3x3 conv +\nsigmoid", BLUE)]
    dx = [16.9, 14.1, 11.3, 8.5, 5.7]
    for (t, c), x in zip(dec, dx):
        box(ax, x, 0.9, 2.3, 1.6, t, c)
    arrow(ax, 18.05, 3.4, 18.05, 2.5)
    for i in range(4):
        arrow(ax, dx[i], 1.7, dx[i + 1] + 2.3, 1.7)
    box(ax, 2.9, 0.9, 2.1, 1.6, "Restored\nx_hat\n128x128x3", BLUE, bold=True)
    arrow(ax, dx[-1], 1.7, 5.0, 1.7)
    ax.text(0.2, 5.65, "Encoder stage: 4x4 stride-2 conv + 3x3 conv, BatchNorm, LeakyReLU", fontsize=6.3, style="italic")
    ax.text(0.2, 0.15, "Decoder stage: 4x4 stride-2 transposed conv + 3x3 conv, BatchNorm, ReLU.   No skip connections: everything "
            "passes through the latent (24x compression of 49,152 values).", fontsize=6.3, style="italic")
    fig.savefig(OUT / "arch_ae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_routing():
    fig, ax = canvas(7.3, 5.0, 19, 12.4)
    ax.text(0.1, 11.9, "(a) Task 2 - hard routing", fontsize=8, fontweight="bold")
    box(ax, 0.3, 8.4, 2.3, 1.8, "Damaged\nimage x~", BLUE)
    box(ax, 3.6, 8.4, 3.4, 1.8, "CNN classifier\n5 conv blocks\n-> p (4 classes)", ORANGE)
    box(ax, 7.8, 8.4, 2.6, 1.8, "r = argmax p", GREY, bold=True)
    arrow(ax, 2.6, 9.3, 3.6, 9.3)
    arrow(ax, 7.0, 9.3, 7.8, 9.3)
    exps = [("r = clean: identity (bypass)", 10.6, GREY), ("r = salt: A_salt", 9.2, GREEN), ("r = blur: A_blur", 7.8, GREEN),
            ("r = occlusion: A_occ", 6.4, GREEN)]
    for t, y, c in exps:
        box(ax, 11.4, y - 0.1, 4.6, 1.0, t, c, fs=6.2)
        arrow(ax, 10.4, 9.3, 11.4, y + 0.4)
    box(ax, 17.0, 8.4, 1.9, 1.8, "Restored\nx_hat", BLUE, bold=True)
    for _, y, _ in exps:
        arrow(ax, 16.0, y + 0.4, 17.0, 9.3)
    ax.text(0.1, 5.2, "(b) Task 3 - soft mixture of experts (jointly fine-tuned)", fontsize=8, fontweight="bold")
    box(ax, 0.3, 1.9, 2.3, 1.8, "Damaged\nimage x~", BLUE)
    box(ax, 3.4, 3.6, 5.0, 1.2, "Gate G (= Task-2 classifier)\nw = softmax(G(x~) / tau)", ORANGE, fs=6.2)
    arrow(ax, 1.4, 3.7, 3.4, 4.2, color="#777")
    ys = [3.9, 2.9, 1.9, 0.9]
    labels = ["identity x~", "A_salt(x~)", "A_blur(x~)", "A_occ(x~)"]
    for t, y in zip(labels, ys):
        box(ax, 9.4, y - 0.2, 2.8, 0.8, t, GREEN if t != "identity x~" else GREY, fs=6.2)
        arrow(ax, 2.6, 2.8, 9.4, y + 0.2, color="#777")
    box(ax, 13.2, 1.5, 3.4, 2.2, "x_hat =\nsum_k w_k * branch_k", PURPLE, fs=6.2, bold=True)
    for y in ys:
        arrow(ax, 12.2, y + 0.2, 13.2, 2.6)
    arrow(ax, 8.4, 4.4, 13.2, 3.5, color="#c0392b", ls="--")
    ax.text(9.2, 4.9, "weights w0..w3 (sum to 1)", fontsize=6.3, color="#c0392b")
    box(ax, 17.2, 1.7, 1.7, 1.8, "Restored\nx_hat", BLUE, fs=6.3, bold=True)
    arrow(ax, 16.6, 2.6, 17.2, 2.6)
    ax.text(0.3, -0.5, "Warm-up: experts frozen, only the gate trains; then all parts are fine-tuned jointly with a smaller "
            "learning rate.\nLoss = L1 + SSIM + cross-entropy(gate, true corruption) + balance term sum_k (mean w_k - 1/4)^2.",
            fontsize=6.3, style="italic")
    fig.savefig(OUT / "arch_routing.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_gan():
    fig, ax = canvas(7.4, 4.8, 22, 11)
    ax.text(0.1, 10.5, "Generator G(x, s): U-Net with learned style embedding", fontsize=8, fontweight="bold")
    box(ax, 0.3, 7.6, 2.4, 1.8, "Photo x\n128x128x3", BLUE)
    box(ax, 0.3, 5.0, 2.4, 1.8, "Style s (1/2/3)\nEmbedding\n(32-d, learned)", ORANGE, fs=5.8)
    box(ax, 3.7, 6.7, 3.2, 1.9, "concat photo +\ntiled style\n(3 + 32 channels)", GREY, fs=5.8)
    arrow(ax, 2.7, 8.5, 3.7, 8.0)
    arrow(ax, 2.7, 6.0, 3.7, 7.3)
    box(ax, 7.9, 6.7, 4.2, 1.9, "Encoder: 7 stride-2 stages\n128 -> 1 px\nchannels 64 ... 512", GREEN, fs=5.8)
    arrow(ax, 6.9, 7.65, 7.9, 7.65)
    box(ax, 13.1, 6.7, 3.6, 1.9, "1x1 bottleneck (512)\n+ concat style (32)", PURPLE, fs=5.8, bold=True)
    arrow(ax, 12.1, 7.65, 13.1, 7.65)
    arrow(ax, 2.7, 5.4, 14.9, 6.7, color="#c0392b", ls="--", rad=-0.12)
    box(ax, 17.7, 6.7, 3.0, 1.9, "Decoder: 7 stages\n(dropout in first 3)\ntanh -> sketch y_hat", GREEN, fs=5.8)
    arrow(ax, 16.7, 7.65, 17.7, 7.65)
    ax.annotate("", xy=(19.2, 6.7), xytext=(10.0, 6.7),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#2c7a3f", ls="--", connectionstyle="arc3,rad=0.25"))
    ax.text(10.4, 5.0, "U-Net skip connections (encoder stage k -> decoder stage 7-k)", fontsize=6, color="#2c7a3f")
    ax.text(0.1, 4.0, "Discriminator D(x, y, s): PatchGAN", fontsize=8, fontweight="bold")
    box(ax, 0.3, 1.5, 5.2, 1.9, "concat [photo x, sketch y (real or\nG(x,s)), tiled style embedding]\n= 6 + 32 channels", GREY, fs=5.8)
    box(ax, 6.5, 1.5, 5.6, 1.9, "4x4 conv stages: 64, 128, 256 (stride 2),\n512 (stride 1), BatchNorm + LeakyReLU", GREEN, fs=5.8)
    box(ax, 13.1, 1.5, 3.8, 1.9, "4x4 conv\n-> 14x14 real/fake\nlogit map", ORANGE, fs=5.8, bold=True)
    arrow(ax, 5.5, 2.45, 6.5, 2.45)
    arrow(ax, 12.1, 2.45, 13.1, 2.45)
    box(ax, 17.9, 1.5, 3.8, 1.9, "BCE with logits:\nD: real->1, fake->0\nG: fool D + lambda_L1*L1", PURPLE, fs=5.8)
    arrow(ax, 16.9, 2.45, 17.9, 2.45)
    ax.text(0.3, 0.3, "The style embedding enters the generator (input and bottleneck) and the discriminator (its own learned table).",
            fontsize=6.2, style="italic")
    fig.savefig(OUT / "arch_gan.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_ae()
    fig_routing()
    fig_gan()
    print("saved", sorted(p.name for p in OUT.glob("arch_*.png")))
