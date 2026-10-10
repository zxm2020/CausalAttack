"""Evaluate saved offline outputs without inventing attack-success labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from causalattack.backends import FixedSmokeFeatures, YOLOv5Features
from causalattack.core import biased_mmd_squared, rbf, shared_indices, spatial_rows
from causalattack.paths import YOLOV5_ROOT, YOLOV5_WEIGHTS


HERE = Path(__file__).resolve().parent


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        values = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'demo')
    parser.add_argument('--capture-target', type=Path,
                        help='optional measured/synthetic capture ground truth for image metrics')
    parser.add_argument('--samples', type=int, default=1024)
    args = parser.parse_args()
    metadata = json.loads((args.output / 'metrics.json').read_text(encoding='utf-8'))
    reference = image_tensor(args.output / 'reference.png')
    predicted = image_tensor(args.output / 'predicted_capture.png')
    if reference.shape != predicted.shape:
        raise ValueError('Reference and prediction dimensions differ')
    if metadata['feature_backend'] == 'smoke':
        model = FixedSmokeFeatures().eval()
    else:
        model = YOLOv5Features(YOLOV5_ROOT, YOLOV5_WEIGHTS,
                               torch.device('cpu'), size=reference.shape[-1])
    with torch.no_grad():
        ref_map = model(reference)
        attacked_map = model(predicted)
        indices = shared_indices(ref_map, args.samples, 2026)
        a = spatial_rows(ref_map)[indices]
        b = spatial_rows(attacked_map)[indices]
        mmd = float(biased_mmd_squared(a, b, 1.0, rbf(a, a, 1.0).mean()))
    result = {'feature_backend': metadata['feature_backend'],
              'reference_guided_mmd_squared': mmd,
              'attack_success_rate': None,
              'asr_reason': 'No clean-qualified labeled detection population was supplied'}
    if args.capture_target:
        target = image_tensor(args.capture_target)
        if target.shape != predicted.shape:
            raise ValueError('Capture target and prediction dimensions differ')
        error = float((predicted - target).square().mean())
        result['capture_mse'] = error
        result['capture_psnr_db'] = float(-10 * np.log10(error)) if error > 0 else 'infinite'
        from skimage.metrics import structural_similarity
        result['capture_ssim'] = float(structural_similarity(
            predicted[0].permute(1, 2, 0).numpy(),
            target[0].permute(1, 2, 0).numpy(),
            data_range=1.0, channel_axis=2))
    print(json.dumps(result, indent=2))
    (args.output / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
