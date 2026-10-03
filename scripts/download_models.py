"""Download the trained ONNX models into ./models.

Usage:  python scripts/download_models.py [--url ZIP_URL]

The location is read from models/SOURCE.txt (a direct link to models_final.zip, e.g. a GitHub release asset) unless
--url is given. Google Drive links work too when `gdown` is installed. No third-party package is needed for direct links.
If automatic download is impossible, download the zip manually and unzip it into ./models.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
EXPECTED = ["universal_ae.onnx", "classifier.onnx", "specialist_salt.onnx", "specialist_blur.onnx",
            "specialist_occ.onnx", "soft_moe.onnx", "sketch_generator.onnx"]


def fetch(url: str, dest: Path):
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            sys.exit("Google Drive link: install gdown first (pip install gdown) or use a direct link.")
        if "/folders/" in url:
            gdown.download_folder(url=url, output=str(MODELS), quiet=False, use_cookies=False)
            return None
        gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
        return dest
    print(f"downloading {url}")

    def hook(blocks, size, total):
        if total > 0 and blocks % 200 == 0:
            print(f"  {min(100, blocks * size * 100 // total)} %", end="\r")

    urllib.request.urlretrieve(url, dest, hook)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    src = MODELS / "SOURCE.txt"
    url = a.url or (src.read_text().strip() if src.exists() else None)
    if not url or url.startswith("TODO"):
        sys.exit("No download URL: pass --url or put the link into models/SOURCE.txt (see models/README.md).")
    out = fetch(url, MODELS / "models_final.zip")
    if out and zipfile.is_zipfile(out):
        with zipfile.ZipFile(out) as z:
            z.extractall(MODELS)
        out.unlink()
    for p in list(MODELS.rglob("*.onnx")) + list(MODELS.rglob("onnx_verification.json")):  # flatten nested folders
        if p.parent != MODELS:
            p.replace(MODELS / p.name)
    missing = [m for m in EXPECTED if not (MODELS / m).exists()]
    print("All models present." if not missing else f"Missing: {missing}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
