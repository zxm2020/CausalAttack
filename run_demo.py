"""Offline single-image attack update. --smoke uses explicitly synthetic fixtures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from causalattack.backends import (FixedSmokeFeatures, ProjectNetOnline,
                                   SyntheticPC, YOLOv5Features)
from causalattack.core import AttackSettings, display_command, optimize
from causalattack.paths import (PROJECTNET_CHECKPOINT, PROJECTNET_ROOT, VAE_ROOT,
                                YOLOV5_ROOT, YOLOV5_WEIGHTS)

HERE = Path(__file__).resolve().parent


def save_rgb(tensor: torch.Tensor, path: Path):
    x = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(x*255).astype(np.uint8), 'RGB').save(path)


def save_texture_preview(tensor: torch.Tensor, path: Path):
    """Contrast-stretched diagnostic only; never used as a projector command."""
    x = tensor.detach().cpu()[0].permute(1, 2, 0).numpy()
    low, high = float(x.min()), float(x.max())
    preview = np.zeros_like(x) if high <= low else (x-low)/(high-low)
    Image.fromarray(np.rint(preview*255).astype(np.uint8), 'RGB').save(path)


def load_rgb(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB').resize((size,size), Image.Resampling.LANCZOS),
                           dtype=np.float32).copy()/255.0
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(device)


def synthetic_pair(size: int, device: torch.device):
    reference = Image.new('RGB', (size,size), (80,104,124))
    draw = ImageDraw.Draw(reference)
    draw.rectangle((0,int(size*.72),size,int(size*.88)), fill=(62,74,72))
    scene = reference.copy()
    draw = ImageDraw.Draw(scene)
    draw.rectangle((int(size*.38),int(size*.25),int(size*.63),int(size*.72)),
                   fill=(190,72,38))
    draw.ellipse((int(size*.42),int(size*.14),int(size*.59),int(size*.31)),
                 fill=(210,154,116))
    def tensor(image):
        return torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1).unsqueeze(0).to(device)
    return tensor(scene),tensor(reference)


def synthetic_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = Image.new('L', (size,size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((int(size*.38),int(size*.25),int(size*.63),int(size*.72)), fill=255)
    draw.ellipse((int(size*.42),int(size*.14),int(size*.59),int(size*.31)), fill=255)
    return torch.from_numpy(np.asarray(mask,dtype=np.float32).copy()/255).unsqueeze(0).unsqueeze(0).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='synthetic P&C and fixed non-victim features')
    p.add_argument('--fixture', action='store_true',
                   help='deterministic software inputs with the trained ProjectNet backend')
    p.add_argument('--features', choices=['smoke','yolo'], default=None)
    p.add_argument('--scene', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--mask', type=Path, help='target-local grayscale mask (required in ProjectNet mode)')
    p.add_argument('--projectnet-checkpoint', type=Path,
                   default=PROJECTNET_CHECKPOINT)
    p.add_argument('--vae', type=Path,
                   default=VAE_ROOT,
                   help='local stabilityai/sd-vae-ft-mse directory')
    p.add_argument('--yolo-weights', type=Path, default=YOLOV5_WEIGHTS)
    p.add_argument('--output', type=Path, default=HERE/'outputs'/'demo')
    p.add_argument('--size', type=int)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--samples', type=int, default=1024)
    p.add_argument('--smoke-learning-rate', type=float, default=12.0,
                   help='smoke-only visibility override; paper configuration remains 0.01')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = p.parse_args()
    if args.smoke and args.fixture:
        p.error('--smoke and --fixture are mutually exclusive')
    if args.steps < 1 or args.samples < 1:
        p.error('steps and samples must be positive')
    if not args.smoke and not args.fixture and (args.scene is None or args.reference is None or args.mask is None):
        p.error('ProjectNet mode requires scene, matching target-free reference, and target mask')
    size = args.size or (64 if args.smoke else 256)
    if not args.smoke and size != 256:
        p.error('the local ProjectNet checkpoint uses 256x256 inputs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    torch.manual_seed(2026)
    if args.scene and args.reference:
        scene = load_rgb(args.scene,size,device)
        reference = load_rgb(args.reference,size,device)
        reference_provenance = 'user_supplied_matched_target_free_reference'
    elif args.smoke or args.fixture:
        scene,reference = synthetic_pair(size,device)
        reference_provenance = ('deterministic_synthetic_smoke_fixture' if args.smoke else
                                'deterministic_synthetic_ProjectNet_integration_fixture')
    else:
        p.error('scene/reference inputs missing')
    if args.mask is not None:
        with Image.open(args.mask) as source_mask:
            mask_values = np.asarray(source_mask.convert('L').resize(
                (size,size), Image.Resampling.NEAREST), dtype=np.float32).copy()/255.0
        target_mask = torch.from_numpy(mask_values).unsqueeze(0).unsqueeze(0).to(device)
    elif args.smoke or args.fixture:
        target_mask = synthetic_mask(size,device)
    else:
        p.error('target mask missing')
    projector = SyntheticPC() if args.smoke else ProjectNetOnline(
        PROJECTNET_ROOT,args.projectnet_checkpoint,device,
        vae_source=args.vae)
    feature_type = args.features or ('smoke' if args.smoke else 'yolo')
    features = FixedSmokeFeatures().to(device).eval() if feature_type == 'smoke' else YOLOv5Features(
        YOLOV5_ROOT,args.yolo_weights,device,size=size)
    if isinstance(projector,ProjectNetOnline):
        projector.prepare_scene(scene)
    settings = AttackSettings(outer_steps=args.steps,samples=args.samples,
                              learning_rate=args.smoke_learning_rate if args.smoke else 0.01,
                              alpha=1.0/(size*size) if args.smoke else 1.0)
    initial = torch.zeros_like(scene)
    texture,history,measurements = optimize(scene,reference,projector,features,settings,initial,
                                            mask=target_mask)
    predicted = projector(scene,texture)
    command = display_command(texture,canvas_hw=None)
    args.output.mkdir(parents=True,exist_ok=True)
    for filename,image in [('scene.png',scene),('reference.png',reference),
                           ('initial_texture.png',initial),('optimized_texture.png',texture),
                           ('predicted_capture.png',predicted)]:
        save_rgb(image,args.output/filename)
    save_texture_preview(texture,args.output/'optimized_texture_preview.png')
    Image.fromarray(np.rint(target_mask[0,0].cpu().numpy()*255).astype(np.uint8),'L').save(
        args.output/'target_mask.png')
    Image.fromarray(command[0].permute(1,2,0).cpu().numpy(),'RGB').save(args.output/'display_texture.png')
    torch.save(texture.cpu(),args.output/'continuous_texture.pt')
    with (args.output/'loss_curve.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    metrics = dict(mode='synthetic_smoke_not_physical' if args.smoke else 'local_trained_synthetic_ProjectNet',
                   reference_provenance=reference_provenance,
                   feature_backend=feature_type,
                   projectnet_provenance=getattr(projector,'provenance',None),
                   scene_shape=list(scene.shape),prediction_shape=list(predicted.shape),
                   feature_shape=list(features(reference).shape),
                   display_shape=list(command.shape),
                   display_mapping='identity_uncalibrated_offline',
                   display_min=int(command.min()),display_max=int(command.max()),
                   texture_changed=bool((texture-initial).abs().max()>0),
                   learning_rate=settings.learning_rate, alpha=settings.alpha, **measurements)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
    print('Outputs:',args.output.resolve())


if __name__ == '__main__':
    main()
