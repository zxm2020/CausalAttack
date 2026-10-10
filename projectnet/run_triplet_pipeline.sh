#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ----------------------------
# Load key from key.txt if env not set
# ----------------------------
if [[ -z "${MODELSCOPE_SDK_TOKEN:-}" ]]; then
  if [[ -f "key.txt" ]]; then
    export MODELSCOPE_SDK_TOKEN="$(tr -d '\r\n' < key.txt)"
  else
    echo "ERROR: key.txt not found and MODELSCOPE_SDK_TOKEN not set"
    exit 1
  fi
fi

echo "PyTorch / CUDA check"
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ----------------------------
# Generate A+B=C triplet dataset
# ----------------------------
python generate_data_modelscope.py \
    --source modelscope \
    --model Qwen/Qwen-Image \
    --target-type truck \
    --scenes 3 \
    --pairs-per-scene 2 \
    --size 256 \
    --out data/truck_triplets_3x2 \
    --max-wait 1800 \
    --poll-interval 10

# ----------------------------
# Train ProjectNet using existing train.py
# ----------------------------
python train.py \
    --manifest data/truck_triplets_3x2/manifest.csv \
    --out runs/truck_triplets_3x2 \
    --allow-synthetic \
    --epochs 3 \
    --batch 2 \
    --lr 2e-4 \
    --save-fisher

echo "Pipeline finished."
echo "Weights: runs/truck_triplets_3x2/projectnet.pt"
