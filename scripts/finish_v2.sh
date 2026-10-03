#!/bin/sh
# Waits for run_v2.sh to finish, then exports the final ONNX models (v2 restoration models + Task 4 generator)
# and builds the download bundles on Drive.   Usage: sh scripts/finish_v2.sh /content/drive/MyDrive/genai_a1
D=${1:-/content/drive/MyDrive/genai_a1}
O=$D/v2
export DATA_ROOT=${DATA_ROOT:-/content/data}
cd /content/repo
while pgrep -f run_v2.sh > /dev/null; do sleep 20; done
if [ ! -f $O/runs/task3/best.pt ]; then echo "v2 did not finish (no task3 checkpoint)"; exit 1; fi
python -m restoration.export_onnx --universal $O/runs/task1/best.pt --cls $D/runs/task2_cls/best.pt \
  --salt $O/runs/task2_salt/best.pt --blur $O/runs/task2_blur/best.pt --occ $O/runs/task2_occ/best.pt \
  --moe $O/runs/task3/best.pt --gan $D/runs/task4/best.pt --out $D/models_final
cd $D/models_final && rm -f ../models_final.zip && zip -q ../models_final.zip *.onnx onnx_verification.json
cd $D && rm -f final_results_bundle.zip
zip -rq final_results_bundle.zip results/task4 configs v2/results v2/configs v2/optuna v2/runs runs/task4/val_results.json runs/task4/samples \
  models_final/onnx_verification.json -x '*.pt'
zip -jq final_results_bundle.zip /content/work/mlflow.db
ls -la $D/models_final.zip $D/final_results_bundle.zip
echo "== finish_v2 done =="
