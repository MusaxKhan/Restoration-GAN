"""Count learnable parameters of every selected model and write report/generated/params.tex.

    python scripts/count_params.py
"""
import json
from pathlib import Path

from restoration.models.classifier import CorruptionClassifier
from restoration.models.gan import PatchDiscriminator, UNetGenerator
from restoration.train_ae import build_ae

ROOT = Path(__file__).resolve().parent.parent


def n(m):
    return sum(p.numel() for p in m.parameters())


def load(p):
    return json.load(open(ROOT / p, encoding="utf-8"))


uni_cfg = load("results/task1/test_results.json")["cfg"]
spec_cfg = load("configs/ae_specialist_best.json")["params"]
cls_cfg = load("configs/cls_best.json")["params"]
gan_cfg = load("results/task4/test_results.json")["cfg"]

uni = n(build_ae(uni_cfg))
spec = n(build_ae(spec_cfg))
cls = n(CorruptionClassifier(cls_cfg["base_ch"], cls_cfg["depth"], cls_cfg["dropout"]))
gen = n(UNetGenerator(gan_cfg["base_ch"], gan_cfg["style_dim"], gan_cfg["dropout"]))
dis = n(PatchDiscriminator(gan_cfg["base_ch"], gan_cfg["style_dim"]))
moe = cls + 3 * spec


def fmt(x):
    return f"{x:,}".replace(",", "{,}")


rows = [
    ("Task 1 universal autoencoder", f"$c={uni_cfg['base_ch']}$, spatial latent $8{{\\times}}8{{\\times}}{uni_cfg['latent_ch']}$ (2{{,}}048 numbers)", uni),
    ("Task 2 specialist autoencoder (each of 3)", f"$c={spec_cfg['base_ch']}$, spatial latent $8{{\\times}}8{{\\times}}{spec_cfg['latent_ch']}$", spec),
    ("Task 2 classifier (= Task 3 gate)", f"{cls_cfg['depth']} conv blocks, base {cls_cfg['base_ch']} channels", cls),
    ("Task 3 soft MoE (gate + 3 experts)", "classifier + 3 specialists, $\\tau$ fixed", moe),
    ("Task 4 generator (U-Net)", f"base {gan_cfg['base_ch']} channels, style embedding {gan_cfg['style_dim']}", gen),
    ("Task 4 discriminator (PatchGAN)", f"base {gan_cfg['base_ch']} channels, style embedding {gan_cfg['style_dim']}", dis),
]
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Learnable parameters of the selected models (counted from the final configurations).}",
         r"\label{tab:params}", r"\footnotesize", r"\begin{tabular}{p{3.0cm}p{3.6cm}r}", r"\toprule",
         r"Model & Configuration & Parameters \\", r"\midrule"]
for a, b, c in rows:
    lines.append(f"{a} & {b} & {fmt(c)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(ROOT / "report" / "generated" / "params.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print({k: v for k, v in dict(universal=uni, specialist=spec, classifier=cls, moe=moe, generator=gen, discriminator=dis).items()})
