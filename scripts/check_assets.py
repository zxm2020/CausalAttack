"""Check the local dataset and model assets expected by the public entry points."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalattack.paths import (PROJECTNET_CHECKPOINT, PROJECTNET_DATA, VAE_ROOT,
                                YOLOV5_WEIGHTS)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--training', action='store_true',
                    help='also require the training manifest and EWC state')
args = parser.parse_args()

assets = {
    'ProjectNet checkpoint': PROJECTNET_CHECKPOINT,
    'frozen VAE config': VAE_ROOT / 'config.json',
    'YOLOv5s checkpoint': YOLOV5_WEIGHTS,
}
if args.training:
    assets.update({
        'ProjectNet EWC state': PROJECTNET_CHECKPOINT.with_name('ewc.pt'),
        'ProjectNet dataset manifest': PROJECTNET_DATA / 'manifest.csv',
    })
status = {name: {'path': str(path.relative_to(ROOT)), 'present': path.is_file()}
          for name, path in assets.items()}
print(json.dumps(status, indent=2))
missing = [name for name, item in status.items() if not item['present']]
if missing:
    print('\nMissing assets: ' + ', '.join(missing), file=sys.stderr)
    print('Download and arrange the release archive as documented in README.md.', file=sys.stderr)
    raise SystemExit(1)
print('\nAll required release assets are present.')
