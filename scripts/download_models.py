"""Download the trained ONNX models into ./models.

Usage:  python scripts/download_models.py [--url FOLDER_OR_ZIP_URL]

The download location is read from models/SOURCE.txt (a Google Drive folder or direct zip link) unless --url is given.
Requires `pip install gdown` for Google Drive links. If automatic download is impossible, download the files manually
from the link in models/README.md and place them in ./models.
"""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
EXPECTED = ["universal_ae.onnx", "classifier.onnx", "specialist_salt.onnx", "specialist_blur.onnx",
            "specialist_occ.onnx", "soft_moe.onnx", "sketch_generator.onnx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    src = ROOT / "models" / "SOURCE.txt"
    url = a.url or (src.read_text().strip() if src.exists() else None)
    if not url:
        sys.exit("No download URL: pass --url or create models/SOURCE.txt (see models/README.md).")
    try:
        import gdown
    except ImportError:
        sys.exit("Install gdown first:  pip install gdown")
    if "/folders/" in url:
        gdown.download_folder(url=url, output=str(MODELS), quiet=False, use_cookies=False)
    else:
        out = MODELS / "models_download.zip"
        gdown.download(url=url, output=str(out), quiet=False, fuzzy=True)
        if zipfile.is_zipfile(out):
            with zipfile.ZipFile(out) as z:
                z.extractall(MODELS)
            out.unlink()
    # downloaded folders may nest the files one level down: flatten
    for p in list(MODELS.rglob("*.onnx")) + list(MODELS.rglob("onnx_verification.json")):
        if p.parent != MODELS:
            p.rename(MODELS / p.name)
    missing = [m for m in EXPECTED if not (MODELS / m).exists()]
    print("All models present." if not missing else f"Missing: {missing}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
