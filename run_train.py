"""Validated entry point to the existing ProjectNet training implementations.

Training never overwrites the supplied ProjectNet or YOLOv5 repositories.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from causalattack.paths import PROJECTNET_CHECKPOINT, PROJECTNET_DATA, PROJECTNET_ROOT


HERE = Path(__file__).resolve().parent
SOURCE = PROJECTNET_ROOT


def count_manifest(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Training manifest is empty')
    counts = {}
    for row in rows:
        key = row.get('split', 'unspecified')
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['inspect', 'base', 'real'], default='inspect')
    p.add_argument('--data-root', type=Path,
                   default=PROJECTNET_DATA)
    p.add_argument('--real-manifest', type=Path)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--base-checkpoint', type=Path,
                   default=PROJECTNET_CHECKPOINT)
    p.add_argument('--ewc', type=Path,
                   default=PROJECTNET_CHECKPOINT.with_name('ewc.pt'))
    args = p.parse_args()
    if args.stage == 'inspect':
        manifest = args.data_root / 'manifest.csv'
        counts = count_manifest(manifest)
        state = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
        print(json.dumps({'manifest': str(manifest), 'split_counts': counts,
                          'checkpoint': str(args.base_checkpoint),
                          'checkpoint_epoch': state.get('epoch'),
                          'checkpoint_label_type': state.get('label_type'),
                          'cuda_available': torch.cuda.is_available()}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The existing ProjectNet training scripts require a CUDA GPU. '
                           'This machine currently has a CPU-only PyTorch environment; '
                           'base and real training were not executed.')
    destination = HERE / 'outputs' / ('projectnet_base' if args.stage == 'base' else 'projectnet_real')
    destination.mkdir(parents=True, exist_ok=True)
    if args.stage == 'base':
        script = SOURCE / 'projectnet_training_3000' / 'train_projectnet_3000.py'
        command = [sys.executable, str(script), '--data-root', str(args.data_root.resolve()),
                   '--out', str(destination), '--allow-synthetic']
        if args.epochs is not None:
            command += ['--epochs', str(args.epochs)]
        if args.batch is not None:
            command += ['--batch', str(args.batch)]
    else:
        if args.real_manifest is None:
            p.error('--real-manifest is required for real-domain adaptation')
        script = SOURCE / 'train.py'
        command = [sys.executable, str(script), '--manifest', str(args.real_manifest.resolve()),
                   '--out', str(destination), '--resume', str(args.base_checkpoint.resolve()),
                   '--ewc', str(args.ewc.resolve())]
        if args.epochs is not None:
            command += ['--epochs', str(args.epochs)]
        if args.batch is not None:
            command += ['--batch', str(args.batch)]
    subprocess.run(command, cwd=script.parent, check=True)


if __name__ == '__main__':
    main()
