"""FastAPI backend: validates uploads, preprocesses, runs the ONNX models, returns outputs + timings.

Endpoints
  GET  /api/health          model availability + runtime info
  GET  /api/samples         clean sample images that can be selected in the UI
  POST /api/universal       Task 1  universal restoration
  POST /api/hard-route      Task 2  classifier -> specialist (hard routing)
  POST /api/soft-moe        Task 3  soft mixture of experts
  POST /api/sketch          Task 4  face -> sketch generator
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from . import corruptions as C

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", Path(__file__).resolve().parent.parent / "samples"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
SIZE = 128

MODEL_FILES = {
    "universal": "universal_ae.onnx", "classifier": "classifier.onnx",
    "salt": "specialist_salt.onnx", "blur": "specialist_blur.onnx", "occ": "specialist_occ.onnx",
    "moe": "soft_moe.onnx", "sketch": "sketch_generator.onnx",
}
EXPERT_KEYS = {C.SALT: "salt", C.BLUR: "blur", C.OCC: "occ"}
EXPERT_NAMES = {C.SALT: "Salt-and-pepper specialist", C.BLUR: "Blur specialist", C.OCC: "Occlusion specialist"}

app = FastAPI(title="GenAI Assignment 1 - Restoration & Face-to-Sketch API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: dict[str, ort.InferenceSession] = {}


def session(key: str) -> ort.InferenceSession:
    """Load an ONNX model on first use (cached)."""
    if key not in _sessions:
        path = MODEL_DIR / MODEL_FILES[key]
        if not path.exists():
            raise HTTPException(503, f"Model file missing: {path.name}. See README: download the ONNX models "
                                     f"into the models/ folder.")
        _sessions[key] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions[key]


# ------------------------------------------------------------------ image helpers
async def read_image(file: UploadFile) -> Image.Image:
    """Validate (size, real image, allowed format) and return an RGB PIL image."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_BYTES // 1024 // 1024} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(415, "File is not a valid image.")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"Unsupported image format {fmt}; use JPEG, PNG, WEBP or BMP.")
    return img.convert("RGB")


def to_array(img: Image.Image) -> np.ndarray:
    """RGB PIL -> float32 (128, 128, 3) in [0, 1] (bicubic resize, as in training)."""
    return np.asarray(img.resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255.0


def to_chw(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def png_b64(a: np.ndarray) -> str:
    """float (H, W, 3) in [0, 1] -> base64 PNG data URL."""
    buf = io.BytesIO()
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(sess: ort.InferenceSession, feed: dict) -> tuple[list[np.ndarray], float]:
    t0 = time.perf_counter()
    out = sess.run(None, feed)
    return out, (time.perf_counter() - t0) * 1000


def psnr(a, b) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def prepare_input(img: Image.Image, mode: str, ctype: str, level: str, seed: int):
    """Return (clean_or_None, model_input, spec).

    mode='uploaded'  -> the image is used as-is (it may already be corrupted); no reference available.
    mode='corrupt'   -> the image is treated as a clean image and the chosen corruption is applied."""
    arr = to_array(img)
    if mode == "uploaded":
        return None, arr, {"type": C.CLEAN, "note": "image used as uploaded (no synthetic corruption applied)"}
    if ctype not in C.CLASS_NAMES or level not in C.LEVELS:
        raise HTTPException(422, f"corruption must be one of {C.CLASS_NAMES}, level one of {C.LEVELS}.")
    spec = C.spec_from_ui(C.CLASS_NAMES.index(ctype), C.LEVELS.index(level), seed)
    return arr, C.apply_corruption(arr, spec), spec


def settings_of(mode, spec, level, seed):
    d = C.describe(spec) if "note" not in spec else {"type": "as uploaded"}
    d.update({"mode": mode, "level": level if mode == "corrupt" else None, "seed": seed})
    return d


def error_map(clean, restored):
    return png_b64(np.clip(np.abs(restored - clean) * 4, 0, 1)) if clean is not None else None


def metrics(clean, restored, corrupted):
    if clean is None:
        return None
    return {"psnr_restored": round(psnr(restored, clean), 2), "psnr_input": round(psnr(corrupted, clean), 2)}


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    models = {k: (MODEL_DIR / f).exists() for k, f in MODEL_FILES.items()}
    return {"status": "ok", "models": models, "all_models_present": all(models.values()),
            "onnxruntime": ort.__version__, "providers": ort.get_available_providers()}


@app.get("/api/samples")
def samples():
    """Clean sample images (base64) that can be picked in the UI instead of uploading."""
    out = []
    for p in sorted(SAMPLE_DIR.glob("*.jpg")):
        out.append({"name": p.stem, "image": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()})
    return out


@app.post("/api/universal")
async def universal(file: UploadFile = File(...), mode: str = Form("corrupt"), corruption: str = Form("salt_pepper"),
                    level: str = Form("medium"), seed: int = Form(0)):
    img = await read_image(file)
    clean, x, spec = prepare_input(img, mode, corruption, level, seed)
    (restored,), ms = run(session("universal"), {"image": to_chw(x)})
    restored = restored[0].transpose(1, 2, 0)
    return {"task": "universal", "input": png_b64(x), "restored": png_b64(restored),
            "clean": png_b64(clean) if clean is not None else None,
            "error_map": error_map(clean, restored),
            "settings": settings_of(mode, spec, level, seed), "inference_ms": round(ms, 2),
            "metrics": metrics(clean, restored, x)}


@app.post("/api/hard-route")
async def hard_route(file: UploadFile = File(...), mode: str = Form("corrupt"), corruption: str = Form("salt_pepper"),
                     level: str = Form("medium"), seed: int = Form(0)):
    img = await read_image(file)
    clean, x, spec = prepare_input(img, mode, corruption, level, seed)
    inp = to_chw(x)
    (probs,), ms_cls = run(session("classifier"), {"image": inp})
    probs = probs[0]
    route = int(np.argmax(probs))
    if route == C.CLEAN:  # identity bypass: a clean image is not processed by any expert
        restored, ms_exp, expert = x, 0.0, "Identity bypass (no restoration)"
    else:
        (out,), ms_exp = run(session(EXPERT_KEYS[route]), {"image": inp})
        restored, expert = out[0].transpose(1, 2, 0), EXPERT_NAMES[route]
    return {"task": "hard_route", "input": png_b64(x), "restored": png_b64(restored),
            "clean": png_b64(clean) if clean is not None else None, "error_map": error_map(clean, restored),
            "probabilities": {n: round(float(p), 4) for n, p in zip(C.CLASS_NAMES, probs)},
            "predicted": C.CLASS_NAMES[route], "selected_expert": expert,
            "true_condition": C.CLASS_NAMES[spec["type"]] if mode == "corrupt" else None,
            "settings": settings_of(mode, spec, level, seed),
            "inference_ms": round(ms_cls + ms_exp, 2), "classifier_ms": round(ms_cls, 2), "expert_ms": round(ms_exp, 2),
            "metrics": metrics(clean, restored, x)}


@app.post("/api/soft-moe")
async def soft_moe(file: UploadFile = File(...), mode: str = Form("corrupt"), corruption: str = Form("salt_pepper"),
                   level: str = Form("medium"), seed: int = Form(0)):
    img = await read_image(file)
    clean, x, spec = prepare_input(img, mode, corruption, level, seed)
    (restored, w), ms = run(session("moe"), {"image": to_chw(x)})
    restored, w = restored[0].transpose(1, 2, 0), w[0]
    names = ["Identity (clean)", "Salt-and-pepper expert", "Blur expert", "Occlusion expert"]
    order = np.argsort(-w)
    return {"task": "soft_moe", "input": png_b64(x), "restored": png_b64(restored),
            "clean": png_b64(clean) if clean is not None else None, "error_map": error_map(clean, restored),
            "weights": {n: round(float(v), 4) for n, v in zip(names, w)},
            "dominant_branch": names[int(order[0])], "contributions_ranked": [names[i] for i in order if w[i] > 0.05],
            "settings": settings_of(mode, spec, level, seed), "inference_ms": round(ms, 2),
            "metrics": metrics(clean, restored, x)}


@app.post("/api/sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3.")
    img = await read_image(file)
    x = to_array(img)
    inp = (to_chw(x) * 2 - 1).astype(np.float32)
    (out,), ms = run(session("sketch"), {"photo": inp, "style": np.array([style - 1], dtype=np.int64)})
    sk = (out[0].transpose(1, 2, 0) + 1) / 2
    return {"task": "sketch", "input": png_b64(x), "sketch": png_b64(sk), "style": f"Style {style}",
            "inference_ms": round(ms, 2)}
