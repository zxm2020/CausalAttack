"""One-command: ModelScope generated base scenes -> toy-labeled triples -> local CUDA LDM weights.

Do not paste a token into a chat or put it in a source file. Read only environment variable.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(script, *args):
    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    print('Executing', script, '(API token never appears in the command)', flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen-Image')
    p.add_argument('--source', choices=['modelscope', 'procedural'], default='modelscope')
    p.add_argument('--scenes', type=int, default=12)
    p.add_argument('--pairs-per-scene', type=int, default=16)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--data', default='data/synthetic')
    p.add_argument('--out', default='outputs/synthetic')
    p.add_argument('--generate-only', action='store_true')
    p.add_argument('--train-only', action='store_true')
    a = p.parse_args()
    if not a.generate_only:
        import torch
        if not torch.cuda.is_available():
            p.error('CUDA GPU not available; check driver, CUDA-enabled PyTorch and CUDA_VISIBLE_DEVICES before generating billable API data')
    if not a.train_only:
        if a.source == 'modelscope' and not os.getenv('MODELSCOPE_SDK_TOKEN'):
            p.error('Please set MODELSCOPE_SDK_TOKEN in your OWN runtime; NEVER send your token into chat.')
        call('generate_data_modelscope.py', '--source', a.source, '--model', a.model,
             '--scenes', a.scenes, '--pairs-per-scene', a.pairs_per_scene, '--size', a.size, '--out', a.data)
    if not a.generate_only:
        call('train.py', '--manifest', str(Path(a.data) / 'manifest.csv'),
             '--out', a.out, '--size', a.size, '--epochs', a.epochs,
             '--batch', a.batch, '--allow-synthetic')
    print('Finished requested stages. Reminder: SYNTHETIC weights do not validate physical P&C.', flush=True)


if __name__ == '__main__':
    main()
