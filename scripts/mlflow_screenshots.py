"""Screenshots of the MLflow UI (experiment tracking evidence for the report).

    mlflow server --backend-store-uri sqlite:///experiments/mlflow.db --port 5000
    python scripts/mlflow_screenshots.py --out report/figures
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

# (screenshot name, URL fragment): run tables and metric charts of the final Task 3 and Task 4 runs
PAGES = [
    ("mlflow_runs_task3", "#/experiments/18/runs"),
    ("mlflow_metrics_task3", "#/experiments/18/runs/0ebee8dfd6454a51b24749f8f33904d6/model-metrics"),
    ("mlflow_metrics_task4", "#/experiments/13/runs/67660397ec2942d0a97e687513f70601/model-metrics"),
    ("mlflow_params_task4", "#/experiments/13/runs/67660397ec2942d0a97e687513f70601/overview"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    out = Path(a.out)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 950})
        for name, frag in PAGES:
            page.goto(f"{a.url}/{frag}")
            page.wait_for_timeout(6000)
            for label in ("Got it", "Dismiss"):  # close first-run hints
                loc = page.get_by_text(label, exact=True)
                if loc.count():
                    loc.first.click()
                    page.wait_for_timeout(300)
            page.screenshot(path=str(out / f"{name}.png"))
            print("saved", name)
        b.close()


if __name__ == "__main__":
    main()
