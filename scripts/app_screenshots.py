"""Capture screenshots of the four workspaces of the running application (for the report).

Needs the app running (frontend on :5173 or :3000, backend on :8000) and Google Chrome installed:
    python scripts/app_screenshots.py --url http://localhost:5173 --out report/figures
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def shot(page, path):
    page.wait_for_timeout(600)
    page.screenshot(path=str(path), full_page=True)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5173")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--sample", type=int, default=1, help="index of the sample thumbnail to use (0-based)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1000})

        def open_page(h):
            page.goto(f"{a.url}/#{h}")
            page.reload()
            page.wait_for_selector("text=Backend online", timeout=15000)
            page.wait_for_selector("img[alt^='pet_']", timeout=15000)

        def pick():
            page.locator("img[alt^='pet_']").nth(a.sample).click()
            page.wait_for_timeout(400)

        # --- Task 1: universal restoration, salt-and-pepper high
        open_page("universal"); pick()
        page.get_by_text("Salt-and-pepper", exact=True).click()
        page.get_by_text("High", exact=True).click()
        page.get_by_role("button", name="Restore image").click()
        page.wait_for_selector("text=Inference time", timeout=30000)
        shot(page, out / "app_universal.png")

        # --- Task 2: hard routing, occlusion medium
        open_page("hard"); pick()
        page.get_by_text("Rectangular occlusion", exact=True).click()
        page.get_by_text("Medium", exact=True).click()
        page.get_by_role("button", name="Restore image").click()
        page.wait_for_selector("text=Classifier probabilities", timeout=30000)
        shot(page, out / "app_hard_routed.png")

        # --- Task 3: soft MoE, blur high
        open_page("soft"); pick()
        page.get_by_text("Gaussian blur", exact=True).click()
        page.get_by_text("High", exact=True).click()
        page.get_by_role("button", name="Restore image").click()
        page.wait_for_selector("text=Routing weights", timeout=30000)
        shot(page, out / "app_soft_moe.png")

        # --- Task 4: sketch generator, style 2
        page.goto(f"{a.url}/#sketch"); page.reload()
        page.wait_for_selector("img[alt^='face_']", timeout=15000)
        page.locator("img[alt^='face_']").nth(a.sample + 2).click()
        page.get_by_text("Style 2", exact=True).click()
        page.get_by_role("button", name="Generate sketch").click()
        page.wait_for_selector("text=Download result", timeout=30000)
        shot(page, out / "app_face_to_sketch.png")
        b.close()


if __name__ == "__main__":
    main()
