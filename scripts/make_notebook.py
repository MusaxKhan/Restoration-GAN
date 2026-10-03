"""Generates notebooks/colab_training.ipynb (run once; the notebook itself is committed)."""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(True)})


md("""
# GenAI Assignment 1 - Colab training notebook
Run **Runtime > Change runtime type > T4 GPU** first. Run the cells top to bottom; every stage writes to Google Drive
(`MyDrive/genai_a1`) and is **resumable** (Optuna studies are SQLite-backed, training uses `--resume`), so if the free
session disconnects just re-run the setup cells and the stage you were in.
""")

md("## 0. Setup (run every time the session restarts)")
code("""
from google.colab import drive
drive.mount('/content/drive')
import os
BASE = '/content/drive/MyDrive/genai_a1'
os.makedirs(BASE, exist_ok=True)
os.environ['BASE'] = BASE
os.environ['DATA_ROOT'] = '/content/data'
os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{BASE}/mlflow.db'
os.environ['MLFLOW_ARTIFACT_ROOT'] = f'{BASE}/mlartifacts'
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
""")
code("""
%%bash
cd /content
if [ -d repo ]; then cd repo && git pull -q; else git clone -q https://github.com/MusaxKhan/Restoration-GAN.git repo; fi
cd /content/repo && git log --oneline | head -3
pip -q install optuna mlflow onnx onnxruntime scikit-image
nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code("""
%cd /content/repo
# Task 4 data: upload FS2K.zip (made on your PC) to MyDrive/genai_a1/FS2K.zip first
import os, subprocess
os.makedirs('/content/data', exist_ok=True)
if not os.path.exists('/content/data/FS2K'):
    subprocess.run(['unzip', '-q', f'{BASE}/FS2K.zip', '-d', '/content/data'], check=True)
print(os.listdir('/content/data/FS2K'))
""")
code("""
# Oxford-IIIT Pet is downloaded + cached to 128x128 automatically (takes ~2 min). Manifests come from the repo.
!python -m restoration.data.make_manifests
!python -m pytest tests/test_data.py -q
""")

md("""
## 1. Speed check (about 2 minutes) - please send me the printed epoch times
""")
code("""
!python -m restoration.train_ae --task universal --out $BASE/runs/speedcheck --epochs 2 --subset-train 1024 --subset-val 256
!python -m restoration.train_gan --out $BASE/runs/speedcheck_gan --epochs 2 --subset-train 256
""")

md("""
## 2. Task 1 - universal autoencoder
""")
code("""
!python -m restoration.optuna_ae --mode universal --trials 20 --epochs 6 --subset-train 1500 --out $BASE/optuna --config-out $BASE/configs/ae_universal_best.json
""")
code("""
!python -m restoration.train_ae --task universal --config $BASE/configs/ae_universal_best.json --out $BASE/runs/task1 --epochs 60 --mlflow-experiment task1_universal_ae --resume
""")
code("""
!python -m restoration.evaluate_task1 --ckpt $BASE/runs/task1/best.pt --out $BASE/results/task1
""")

md("""
## 3. Task 2 - classifier + specialists
""")
code("""
!python -m restoration.optuna_classifier --trials 20 --epochs 5 --subset-train 1500 --out $BASE/optuna --config-out $BASE/configs/cls_best.json
""")
code("""
!python -m restoration.train_classifier --config $BASE/configs/cls_best.json --out $BASE/runs/task2_cls --epochs 30 --mlflow-experiment task2_classifier --resume
""")
code("""
!python -m restoration.optuna_ae --mode specialist --trials 12 --epochs 6 --subset-train 1500 --out $BASE/optuna --config-out $BASE/configs/ae_specialist_best.json
""")
code("""
for t in ['salt', 'blur', 'occ']:
    !python -m restoration.train_ae --task {t} --config $BASE/configs/ae_specialist_best.json --out $BASE/runs/task2_{t} --epochs 60 --mlflow-experiment task2_specialist_{t} --resume
""")
code("""
!python -m restoration.evaluate_task2 --cls $BASE/runs/task2_cls/best.pt --salt $BASE/runs/task2_salt/best.pt --blur $BASE/runs/task2_blur/best.pt --occ $BASE/runs/task2_occ/best.pt --out $BASE/results/task2
""")

md("""
## 4. Task 3 - soft mixture of experts (needs Task 2 finished)
""")
code("""
!python -m restoration.optuna_moe --cls $BASE/runs/task2_cls/best.pt --salt $BASE/runs/task2_salt/best.pt --blur $BASE/runs/task2_blur/best.pt --occ $BASE/runs/task2_occ/best.pt --trials 15 --epochs 4 --warmup-epochs 1 --subset-train 1500 --out $BASE/optuna --config-out $BASE/configs/moe_best.json
""")
code("""
!python -m restoration.train_moe --cls $BASE/runs/task2_cls/best.pt --salt $BASE/runs/task2_salt/best.pt --blur $BASE/runs/task2_blur/best.pt --occ $BASE/runs/task2_occ/best.pt --config $BASE/configs/moe_best.json --out $BASE/runs/task3 --epochs 20 --mlflow-experiment task3_soft_moe
""")
code("""
!python -m restoration.evaluate_task3 --ckpt $BASE/runs/task3/best.pt --out $BASE/results/task3
""")

md("""
## 5. Task 4 - face-to-sketch cGAN
""")
code("""
!python -m restoration.optuna_gan --trials 12 --epochs 10 --out $BASE/optuna --config-out $BASE/configs/gan_best.json
""")
code("""
!python -m restoration.train_gan --config $BASE/configs/gan_best.json --out $BASE/runs/task4 --epochs 150 --mlflow-experiment task4_face_to_sketch --resume
""")
code("""
!python -m restoration.evaluate_task4 --ckpt $BASE/runs/task4/best.pt --out $BASE/results/task4
""")

md("""
## 6. ONNX export + PyTorch-vs-ONNX verification
""")
code("""
!python -m restoration.export_onnx --universal $BASE/runs/task1/best.pt --cls $BASE/runs/task2_cls/best.pt --salt $BASE/runs/task2_salt/best.pt --blur $BASE/runs/task2_blur/best.pt --occ $BASE/runs/task2_occ/best.pt --moe $BASE/runs/task3/best.pt --gan $BASE/runs/task4/best.pt --out $BASE/models
""")

md("""
## 7. Collect small result files (everything except checkpoints/ONNX) for the repo
""")
code("""
!cd $BASE && rm -f results_bundle.zip && zip -rq results_bundle.zip configs results optuna mlflow.db models/onnx_verification.json -x "*.pt"
!ls -la $BASE
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/colab_training.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/colab_training.ipynb with", len(cells), "cells")
