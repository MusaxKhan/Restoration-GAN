"""Generate LaTeX tables for the report directly from the result JSON files (no manual transcription).

    python scripts/make_report_tables.py --results results --configs configs --models models --out report/generated
Missing inputs are skipped with a message, so it can be run at any stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROW = " \\\\\n"


def esc(s) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def load(p: Path):
    if not p.exists():
        print("skip (missing):", p)
        return None
    return json.loads(p.read_text())


def table(path: Path, caption: str, label: str, cols: str, header: list[str], rows: list[list], star=False, small=True):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\caption{" + caption + "}", f"\\label{{{label}}}"]
    if small:
        lines.append("\\footnotesize")
    lines += [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + ROW.rstrip("\n") + "\n\\midrule"]
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + ROW.rstrip("\n"))
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path)


def f(x, d=2):
    return f"{x:.{d}f}"


COND = ["clean", "salt_pepper/low", "salt_pepper/medium", "salt_pepper/high", "blur/low", "blur/medium", "blur/high",
        "occlusion/low", "occlusion/medium", "occlusion/high"]


def restoration_rows(res: dict):
    rows = []
    for c in COND:
        d = res["by_type"].get("clean") if c == "clean" else res["by_type_level"].get(c)
        if d:
            rows.append([esc(c.replace("/", " / ")), f(d["input_psnr"]), f(d["psnr"]), f(d["input_ssim"], 3), f(d["ssim"], 3), d["n"]])
    o = res["overall"]
    rows.append(["\\textbf{overall}", f(o["input_psnr"]), f(o["psnr"]), f(o["input_ssim"], 3), f(o["ssim"], 3), o["n"]])
    return rows


def optuna_table(out: Path, cfg_path: Path, name: str, title: str):
    c = load(cfg_path)
    if not c:
        return
    rows = [[esc(k), esc(v)] for k, v in c["search_space"].items()]
    table(out / f"optuna_{name}_space.tex", f"Optuna search space: {title}.", f"tab:optuna_{name}_space", "ll",
          ["Hyper-parameter", "Distribution"], rows)
    best = [[esc(k), f(v, 5) if isinstance(v, float) else esc(v)] for k, v in c["params"].items()]
    best += [["trials (complete / pruned / total)", f"{c['n_complete']} / {c['n_pruned']} / {c['n_trials']}"],
             ["best trial \\#, objective", f"{c['best_trial']}, {c['best_value']:.4f}"]]
    table(out / f"optuna_{name}_best.tex", f"Optuna result: {title}.", f"tab:optuna_{name}_best", "ll",
          ["Item", "Value"], best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="report/generated")
    a = ap.parse_args()
    R, Cf, out = Path(a.results), Path(a.configs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, title in [("ae_universal", "Task 1 universal autoencoder"), ("cls", "Task 2 corruption classifier"),
                        ("ae_specialist", "Task 2 specialist autoencoders (shared search)"),
                        ("moe", "Task 3 joint fine-tuning"), ("gan", "Task 4 face-to-sketch cGAN")]:
        optuna_table(out, Cf / f"{name}_best.json", name, title)

    # ---- Task 1
    t1 = load(R / "task1" / "test_results.json")
    if t1:
        table(out / "t1_test.tex", "Task 1 test results (PSNR in dB). Input = unrestored corrupted image.", "tab:t1_test",
              "lrrrrr", ["Condition", "PSNR in", "PSNR out", "SSIM in", "SSIM out", "n"], restoration_rows(t1["test"]))

    # ---- Task 2
    c2 = load(R / "task2" / "classifier_test.json")
    if c2:
        rows = [[esc(k), f(v["precision"], 3), f(v["recall"], 3), f(v["f1"], 3), v["support"]] for k, v in c2["per_class"].items()]
        rows.append(["\\textbf{macro avg}", f(c2["macro_precision"], 3), f(c2["macro_recall"], 3), f(c2["macro_f1"], 3), ""])
        rows.append(["\\textbf{accuracy}", "", "", f(c2["accuracy"], 3), ""])
        table(out / "t2_cls.tex", "Task 2 classifier, per-class test metrics.", "tab:t2_cls", "lrrrr",
              ["Class", "Precision", "Recall", "F1", "Support"], rows)
        rows = [[esc(k), f(v, 3)] for k, v in c2["accuracy_by_type_level"].items()]
        table(out / "t2_cls_levels.tex", "Classifier recall by corruption severity (test).", "tab:t2_cls_levels", "lr",
              ["Condition", "Accuracy"], rows)
    r2 = load(R / "task2" / "restoration_test.json")
    if r2:
        rows = []
        for c in COND:
            do = (r2["oracle"]["by_type"].get("clean") if c == "clean" else r2["oracle"]["by_type_level"].get(c))
            dp = (r2["predicted"]["by_type"].get("clean") if c == "clean" else r2["predicted"]["by_type_level"].get(c))
            if do and dp:
                rows.append([esc(c.replace("/", " / ")), f(do["psnr"]), f(do["ssim"], 3), f(dp["psnr"]), f(dp["ssim"], 3)])
        o, p = r2["oracle"]["overall"], r2["predicted"]["overall"]
        rows.append(["\\textbf{overall}", f(o["psnr"]), f(o["ssim"], 3), f(p["psnr"]), f(p["ssim"], 3)])
        table(out / "t2_restoration.tex", "Task 2 test restoration with oracle vs.\\ predicted routing.", "tab:t2_rest", "lrrrr",
              ["Condition", "PSNR oracle", "SSIM oracle", "PSNR pred.", "SSIM pred."], rows)

    # ---- Task 3
    t3 = load(R / "task3" / "test_results.json")
    if t3:
        table(out / "t3_test.tex", "Task 3 soft mixture-of-experts, test results (PSNR in dB).", "tab:t3_test",
              "lrrrrr", ["Condition", "PSNR in", "PSNR out", "SSIM in", "SSIM out", "n"], restoration_rows(t3["test"]))
    ra = load(R / "task3" / "routing_analysis.json")
    if ra:
        rows = [[esc(k)] + [f(x, 3) for x in v] for k, v in ra["weights_by_condition"].items()]
        table(out / "t3_weights.tex", "Mean gate weights per true condition (test).", "tab:t3_weights", "lrrrr",
              ["Condition", "identity", "salt", "blur", "occ."], rows)
        h = ra["gate_health"]
        rows = [["routing accuracy (argmax = true type)", f(h["routing_accuracy"], 3)],
                ["mean weight per branch", ", ".join(f(x, 3) for x in h["mean_weight_per_branch"])],
                ["inactive branches (mean weight $<0.05$)", esc(", ".join(h["inactive_branches"]) or "none")],
                ["mean gate entropy (nats)", f(h["mean_entropy_nats"], 3)]]
        table(out / "t3_health.tex", "Gate health on the test set.", "tab:t3_health", "ll", ["Statistic", "Value"], rows)

    # ---- Task 4
    t4 = load(R / "task4" / "test_results.json")
    if t4:
        rows = [["all", f(t4["test"]["overall"]["l1"], 4), f(t4["test"]["overall"]["psnr"]), f(t4["test"]["overall"]["ssim"], 3), t4["test"]["overall"]["n"]]]
        for k, v in t4["test"]["by_style"].items():
            rows.append([k, f(v["l1"], 4), f(v["psnr"]), f(v["ssim"], 3), v["n"]])
        table(out / "t4_test.tex", "Task 4 test results on the official FS2K test set (images in [0,1]).", "tab:t4_test", "lrrrr",
              ["Style", "L1", "PSNR (dB)", "SSIM", "n"], rows)

    # ---- ONNX
    ov = load(Path(a.models) / "onnx_verification.json")
    if ov:
        rows = [[esc(k), f"{v['max_abs_diff']:.2e}", "yes" if v["passed"] else "NO", v["size_mb"]] for k, v in ov.items()]
        table(out / "onnx.tex", "PyTorch vs.\\ ONNX Runtime consistency (max abs.\\ difference, real test images).", "tab:onnx", "lrcr",
              ["Model", "max |diff|", "passed ($<10^{-4}$)", "size (MB)"], rows)


if __name__ == "__main__":
    main()
