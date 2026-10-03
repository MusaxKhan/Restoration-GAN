"""Backend tests. Run from the repo root with MODEL_DIR pointing at a folder of ONNX models:
    MODEL_DIR=models python -m pytest backend/tests -q
(Model-dependent tests are skipped when the ONNX files are absent.)"""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_DIR", str(ROOT / "models"))

from app import corruptions as BC  # noqa: E402
from app.main import MODEL_DIR, MODEL_FILES, app  # noqa: E402

client = TestClient(app)
have = lambda *keys: all((MODEL_DIR / MODEL_FILES[k]).exists() for k in keys)


def png_bytes(size=(200, 150), color=(120, 80, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def post(path, data=None, content=None, name="a.png", ctype="image/png"):
    return client.post(path, files={"file": (name, png_bytes() if content is None else content, ctype)}, data=data or {})


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "models" in r.json()


def test_samples_listed():
    r = client.get("/api/samples")
    assert r.status_code == 200 and len(r.json()) >= 1 and r.json()[0]["image"].startswith("data:image/jpeg")


def test_rejects_non_image_and_empty_and_bad_params():
    assert post("/api/universal", content=b"not an image", name="x.png").status_code == 415
    assert post("/api/universal", content=b"", name="x.png").status_code == 400
    assert post("/api/universal", data={"corruption": "fire"}).status_code in (422, 503)


def test_rejects_oversized(monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "MAX_BYTES", 100)
    assert post("/api/universal").status_code == 413


def test_backend_corruptions_match_training_pipeline():
    torch = pytest.importorskip("torch")
    from restoration.data import corruptions as TC
    rng = np.random.default_rng(0)
    img = rng.random((128, 128, 3)).astype(np.float32)
    for ctype in (1, 2, 3):
        for level in range(3):
            spec = TC.spec_from_ui(ctype, level, seed=7)
            a = TC.apply_corruption(img, spec)
            b = BC.apply_corruption(img, spec)
            assert np.abs(a - b).max() < 1e-4, (ctype, level)


@pytest.mark.skipif(not have("universal"), reason="ONNX models not available")
def test_universal_with_synthetic_corruption():
    r = post("/api/universal", data={"mode": "corrupt", "corruption": "occlusion", "level": "high"})
    j = r.json()
    assert r.status_code == 200 and j["restored"].startswith("data:image/png") and j["inference_ms"] > 0
    assert j["metrics"]["psnr_restored"] is not None and j["settings"]["type"] == "occlusion"


@pytest.mark.skipif(not have("universal"), reason="ONNX models not available")
def test_universal_uploaded_image_has_no_reference():
    j = post("/api/universal", data={"mode": "uploaded"}).json()
    assert j["clean"] is None and j["metrics"] is None


@pytest.mark.skipif(not have("classifier", "salt", "blur", "occ"), reason="ONNX models not available")
def test_hard_route_returns_probabilities():
    j = post("/api/hard-route", data={"mode": "corrupt", "corruption": "blur", "level": "high"}).json()
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-3
    assert j["predicted"] in BC.CLASS_NAMES and j["selected_expert"]


@pytest.mark.skipif(not have("moe"), reason="ONNX models not available")
def test_soft_moe_weights_sum_to_one():
    j = post("/api/soft-moe", data={"mode": "corrupt", "corruption": "salt_pepper", "level": "low"}).json()
    assert abs(sum(j["weights"].values()) - 1) < 1e-3 and len(j["weights"]) == 4


@pytest.mark.skipif(not have("sketch"), reason="ONNX models not available")
def test_sketch_styles():
    outs = [post("/api/sketch", data={"style": s}).json()["sketch"] for s in (1, 2, 3)]
    assert all(o.startswith("data:image/png") for o in outs)
    assert post("/api/sketch", data={"style": 9}).status_code == 422
