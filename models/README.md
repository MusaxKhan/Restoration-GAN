# Trained ONNX models

The seven inference models are too large for the repository (about 270 MB in total), so they are published as a
single archive. Download it with `python scripts/download_models.py` (reads `models/SOURCE.txt`) or manually from the
link below and unzip the files into this folder.

| File | Model | Size |
|---|---|---|
| `universal_ae.onnx` | Task 1 universal autoencoder | 9.6 MB |
| `classifier.onnx` | Task 2 corruption classifier (softmax probabilities) | 21 MB |
| `specialist_salt.onnx`, `specialist_blur.onnx`, `specialist_occ.onnx` | Task 2 specialists | 9.6 MB each |
| `soft_moe.onnx` | Task 3 complete soft mixture-of-experts pipeline (image -> restored image, 4 weights) | 50 MB |
| `sketch_generator.onnx` | Task 4 style-conditioned generator (photo, style id -> sketch) | 169 MB |

`onnx_verification.json` records the PyTorch-vs-ONNX Runtime comparison (max abs. difference < 4e-6 for every model).

Download link: see `SOURCE.txt` (GitHub release asset `models_final.zip`).
