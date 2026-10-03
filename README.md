# Restoration Studio - GenAI Assignment 1

Four generative-AI systems behind **one browser application**:

| Task | System | Workspace in the app |
|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder (L1 + SSIM) | **Universal Restoration** |
| 2 | Corruption classifier + three specialist autoencoders (hard routing) | **Hard-Routed Restoration** |
| 3 | Jointly trained soft mixture-of-experts (gate + identity + 3 experts) | **Soft Mixture-of-Experts** |
| 4 | Style-conditioned face-to-sketch cGAN (U-Net generator, PatchGAN) | **Face-to-Sketch Generator** |

Tasks 1-3 use the Oxford-IIIT Pet dataset (128x128, 80/20 split of trainval, seed 42, runtime corruptions,
deterministic validation/test manifests). Task 4 uses FS2K (official train/test, 15 % stratified validation, seed 42).
Everything is trained with PyTorch, tuned with Optuna, tracked with MLflow, exported to ONNX and served by a
FastAPI backend with a React + Tailwind frontend (designed first in Google Stitch, see `docs/stitch/`).

## Quick start (Docker) - evaluator instructions

```bash
git clone https://github.com/MusaxKhan/Restoration-GAN.git
cd Restoration-GAN
python scripts/download_models.py        # fetch the ONNX models into ./models  (see "Model files" below)
docker compose up --build                # starts backend (:8000) and frontend (:3000)
```

Open **http://localhost:3000** (API docs: http://localhost:8000/docs, health: http://localhost:8000/api/health).
The header badge turns green when the backend is up and all seven ONNX models are loaded.

### Model files
The trained ONNX models (`universal_ae`, `classifier`, `specialist_salt|blur|occ`, `soft_moe`, `sketch_generator`,
about 0.5 GB in total) are too large for the repository. They are published at the link in
[`models/README.md`](models/README.md); `scripts/download_models.py` downloads them into `models/`
(or copy the files there manually). `models/onnx_verification.json` records the PyTorch-vs-ONNX comparison.

## Repository layout

```
restoration/           PyTorch code: data/, models/, losses, training, Optuna, evaluation, ONNX export
  data/                corruptions.py, pets.py (split, manifests, datasets), fs2k.py
  train_ae.py          Task 1 universal AE and Task 2 specialists
  train_classifier.py  Task 2 classifier           routing.py  hard routing
  train_moe.py         Task 3 soft MoE             train_gan.py  Task 4 cGAN
  optuna_*.py          one Optuna study per task   evaluate_task{1..4}.py   final test evaluation
  export_onnx.py       ONNX export + consistency check
manifests/             fixed validation/test corruption manifests, split indices
configs/               best Optuna configurations (JSON, with full search space and trial tables)
notebooks/             colab_training.ipynb (the exact training run)
backend/               FastAPI app (+tests), Dockerfile       frontend/   React + Tailwind, nginx, Dockerfile
docs/stitch/           Google Stitch design evidence          results/    metrics, figures, MLflow export
report/                IEEE LaTeX technical report
docker-compose.yml     one-command deployment
```

## Reproducing the experiments

Data is **not** stored in the repository. Oxford-IIIT Pet downloads automatically (torchvision); FS2K must be placed in
`data/FS2K` (folders `photo/`, `sketch/` and `anno_train.json`, `anno_test.json`; set `DATA_ROOT` to change the root).

```bash
pip install -r requirements.txt
python -m restoration.data.make_manifests          # deterministic val/test corruption manifests (already committed)
python -m pytest tests -q                          # data / model / GAN unit tests
python -m restoration.optuna_ae --mode universal --trials 20            # Task 1 search -> configs/ae_universal_best.json
python -m restoration.train_ae --task universal --config configs/ae_universal_best.json --out runs/task1 --epochs 60
python -m restoration.evaluate_task1 --ckpt runs/task1/best.pt --out results/task1
```

Tasks 2-4 follow the same pattern (see `notebooks/colab_training.ipynb`, which contains every command that produced
the reported results, in order). Experiments are logged to MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # parameters, curves, checkpoints, sample images
```

## Tests

```bash
python -m pytest tests -q                              # data pipeline, models, GAN
MODEL_DIR=models python -m pytest backend/tests -q     # API (model tests skip when ONNX files are absent)
```

## AI-use statement
AI assistants (Claude) were used for code drafting, debugging, interface scaffolding and documentation; every component
was tested (unit tests, ONNX consistency checks, end-to-end runs). Details are in the technical report's AI-use appendix.
