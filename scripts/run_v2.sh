#!/bin/sh
# Re-run Tasks 1-3 with the spatial-latent autoencoder option (Optuna chooses dense vs spatial).
# The Task 2 classifier is architecture-independent, so the one trained in the first run is re-used.
# Usage (Colab terminal):  sh scripts/run_v2.sh /content/drive/MyDrive/genai_a1
set -e
D=${1:-/content/drive/MyDrive/genai_a1}      # persistent directory with runs/task2_cls from the first run
O=$D/v2                                      # outputs of this run
CLS=$D/runs/task2_cls/best.pt
export DATA_ROOT=${DATA_ROOT:-/content/data}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////content/work/mlflow.db}
export MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-/content/work/mlartifacts}
export MLFLOW_DISABLE_AGENT_HINT=1
cd /content/repo

echo "== Task 1 =="
python -m restoration.optuna_ae --mode universal --trials 24 --epochs 8 --subset-train 1500 --out $O/optuna --config-out $O/configs/ae_universal_best.json
python -m restoration.train_ae --task universal --config $O/configs/ae_universal_best.json --out $O/runs/task1 --epochs 60 --mlflow-experiment v2_task1_universal_ae --resume
python -m restoration.evaluate_task1 --ckpt $O/runs/task1/best.pt --out $O/results/task1

echo "== Task 2 specialists =="
python -m restoration.optuna_ae --mode specialist --trials 14 --epochs 6 --subset-train 1500 --out $O/optuna --config-out $O/configs/ae_specialist_best.json
for t in salt blur occ; do
  python -m restoration.train_ae --task $t --config $O/configs/ae_specialist_best.json --out $O/runs/task2_$t --epochs 60 --mlflow-experiment v2_task2_specialist_$t --resume
done
python -m restoration.evaluate_task2 --cls $CLS --salt $O/runs/task2_salt/best.pt --blur $O/runs/task2_blur/best.pt --occ $O/runs/task2_occ/best.pt --out $O/results/task2

echo "== Task 3 =="
python -m restoration.optuna_moe --cls $CLS --salt $O/runs/task2_salt/best.pt --blur $O/runs/task2_blur/best.pt --occ $O/runs/task2_occ/best.pt --trials 15 --epochs 4 --warmup-epochs 1 --subset-train 1500 --out $O/optuna --config-out $O/configs/moe_best.json
python -m restoration.train_moe --cls $CLS --salt $O/runs/task2_salt/best.pt --blur $O/runs/task2_blur/best.pt --occ $O/runs/task2_occ/best.pt --config $O/configs/moe_best.json --out $O/runs/task3 --epochs 20 --mlflow-experiment v2_task3_soft_moe
python -m restoration.evaluate_task3 --ckpt $O/runs/task3/best.pt --out $O/results/task3
echo "== v2 done =="
