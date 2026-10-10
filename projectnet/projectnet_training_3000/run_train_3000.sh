#!/usr/bin/env bash
set -euo pipefail
cd /data0/zxm/PR/projectnet_modelscope_complete
# Existing projectnet conda env must be active; no API token is necessary for training.
python train_projectnet_3000.py \
  --data-root /data0/zxm/PR/projectnet_modelscope_complete/data/mixed_triplets_3000 \
  --out /data0/zxm/PR/projectnet_modelscope_complete/outputs/projectnet_mixed_3000 \
  --expected-pairs 3000 \
  --allow-synthetic \
  --size 256 \
  --epochs 20 \
  --batch 4 \
  --cache-batch 2 \
  --lr 0.0002 \
  --fisher-batches 8 \
  "$@"
