"""Export every inference model to ONNX and verify numerical consistency against PyTorch.

    python -m restoration.export_onnx --universal runs/task1/best.pt --cls runs/task2_cls/best.pt \
        --salt runs/task2_salt/best.pt --blur runs/task2_blur/best.pt --occ runs/task2_occ/best.pt \
        --moe runs/task3/best.pt --gan runs/task4/best.pt --out models

Outputs: universal_ae.onnx, classifier.onnx (softmax probabilities), specialist_{salt,blur,occ}.onnx,
soft_moe.onnx (restored image + 4 routing weights), sketch_generator.onnx, plus onnx_verification.json.
Any subset can be exported by passing only some of the checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from . import common as cm
from .data.fs2k import FS2KDataset
from .data.pets import PetsDataset
from .evaluate_task3 import load_moe
from .models.moe import SoftMoEExport
from .train_ae import build_ae
from .train_classifier import build_classifier
from .train_gan import build_G

OPSET = 17


class ClassifierProbs(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return torch.softmax(self.c(x), 1)


def _export(model, inputs: tuple, names_in, names_out, path, dynamic=True):
    model.eval()
    axes = {n: {0: "batch"} for n in list(names_in) + list(names_out)} if dynamic else None
    torch.onnx.export(model, inputs, str(path), input_names=list(names_in), output_names=list(names_out),
                      dynamic_axes=axes, opset_version=OPSET, dynamo=False)


def _verify(model, inputs: tuple, path, names_in, tol=1e-4):
    """Max abs difference between PyTorch and ONNX Runtime outputs on the same inputs."""
    model.eval()
    with torch.no_grad():
        ref = model(*inputs)
    ref = [r.numpy() for r in (ref if isinstance(ref, (tuple, list)) else [ref])]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {n: t.numpy() for n, t in zip(names_in, inputs)}
    got = sess.run(None, feed)
    diffs = [float(np.abs(a - b).max()) for a, b in zip(ref, got)]
    return {"max_abs_diff": max(diffs), "per_output": diffs, "passed": max(diffs) < tol,
            "size_mb": round(Path(path).stat().st_size / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("universal", "cls", "salt", "blur", "occ", "moe", "gan"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    pets = PetsDataset("test", subset=8)
    x = torch.stack([pets[i][0] for i in range(4)])  # real, deterministically corrupted test images
    report = {}

    def load(p):
        return torch.load(p, map_location=dev)

    def do(name, model, inputs, names_in, names_out):
        path = out / f"{name}.onnx"
        _export(model, inputs, names_in, names_out, path)
        report[name] = _verify(model, inputs, path, names_in)
        print(name, report[name])

    if a.universal:
        ck = load(a.universal); m = build_ae(ck["cfg"]); m.load_state_dict(ck["model"])
        do("universal_ae", m, (x,), ["image"], ["restored"])
    if a.cls:
        ck = load(a.cls); c = build_classifier(ck["cfg"]); c.load_state_dict(ck["model"])
        do("classifier", ClassifierProbs(c), (x,), ["image"], ["probabilities"])
    for k in ("salt", "blur", "occ"):
        if getattr(a, k):
            ck = load(getattr(a, k)); m = build_ae(ck["cfg"]); m.load_state_dict(ck["model"])
            do(f"specialist_{k}", m, (x,), ["image"], ["restored"])
    if a.moe:
        moe = load_moe(a.moe, dev)
        do("soft_moe", SoftMoEExport(moe), (x,), ["image"], ["restored", "weights"])
    if a.gan:
        ck = load(a.gan); G = build_G(ck["cfg"]); G.load_state_dict(ck["model"])
        fs = FS2KDataset("test", subset=4)
        photo = torch.stack([fs[i][0] for i in range(4)])
        style = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
        do("sketch_generator", G, (photo, style), ["photo", "style"], ["sketch"])
    cm.save_json(report, out / "onnx_verification.json")
    print("all passed:", all(v["passed"] for v in report.values()))


if __name__ == "__main__":
    main()
