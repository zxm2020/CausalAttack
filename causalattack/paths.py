"""Repository-local paths used by the command-line entry points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTNET_ROOT = ROOT / 'projectnet'
YOLOV5_ROOT = ROOT / 'third_party' / 'yolov5'
PROJECTNET_CHECKPOINT = ROOT / 'checkpoints' / 'projectnet' / 'projectnet_best.pt'
VAE_ROOT = ROOT / 'checkpoints' / 'sd-vae-ft-mse'
YOLOV5_WEIGHTS = ROOT / 'checkpoints' / 'yolov5s.pt'
PROJECTNET_DATA = ROOT / 'data' / 'projectnet' / 'mixed_triplets_3000'
