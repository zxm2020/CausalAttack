"""Offline P&C and source-feature adapters.

The smoke adapters are deterministic software fixtures, not trained physical models.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from causalattack.paths import ROOT


class SyntheticPC:
    """Explicitly synthetic additive appearance mapping for the smoke test."""
    def __call__(self, scene: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
        return (scene + 0.25 * texture).clamp(0, 1)


class FixedSmokeFeatures(torch.nn.Module):
    """Fixed color/edge features; not a learned victim detector."""
    def __init__(self):
        super().__init__()
        self.register_buffer('edge_x', torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3))
        self.register_buffer('edge_y', torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(1, keepdim=True)
        edges = torch.cat((F.conv2d(gray, self.edge_x, padding=1),
                           F.conv2d(gray, self.edge_y, padding=1)), 1)
        return F.avg_pool2d(torch.cat((image, edges), 1), 8)


class YOLOv5Features:
    """P3/stride-8 neck output from a verified local YOLOv5s checkpoint."""
    def __init__(self, repository: Path, weights: Path, device: torch.device, size: int = 256):
        repository, weights = repository.resolve(), weights.resolve()
        if not weights.is_file() or not weights.is_relative_to(ROOT.resolve()):
            raise FileNotFoundError('YOLOv5 checkpoint must be inside the repository')
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
        os.environ.setdefault('YOLOV5_CONFIG_DIR', str(ROOT / '.yolov5'))
        sys.path.insert(0, str(repository))
        import models.experimental as experimental
        original_load = torch.load
        def trusted_local_load(*args, **kwargs):
            kwargs['weights_only'] = False  # local, user-provided YOLOv5 pickle only
            return original_load(*args, **kwargs)
        torch.load = trusted_local_load
        try:
            self.model = experimental.attempt_load(str(weights), device=device,
                                                   inplace=False, fuse=False).eval()
        finally:
            torch.load = original_load
        self.model.requires_grad_(False)
        self.size = size
        self.feature = None
        if len(self.model.model) <= 17:
            raise RuntimeError('Loaded model has no layer 17; cannot assume P3 feature')
        self.layer_name = type(self.model.model[17]).__name__
        self.hook = self.model.model[17].register_forward_hook(self._save_feature)
        with torch.no_grad():
            test = torch.zeros((1,3,size,size), device=device)
            self(test)
            if self.feature.shape[-2:] != (size//8,size//8):
                raise RuntimeError(f'Layer 17 is not stride-8 P3: {tuple(self.feature.shape)}')
        self.verified_shape = tuple(self.feature.shape)

    def _save_feature(self, module, inputs, output):
        self.feature = output

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.size, self.size):
            image = F.interpolate(image, (self.size,self.size), mode='bilinear', align_corners=False)
        self.model(image)
        if not isinstance(self.feature, torch.Tensor) or self.feature.ndim != 4:
            raise RuntimeError('YOLOv5 layer 17 did not emit a spatial tensor')
        return self.feature


class ProjectNetOnline:
    """Single-step τ=100 inference through the locally trained conditional U-Net."""
    def __init__(self, repository: Path, checkpoint: Path, device: torch.device,
                 seed: int = 2026, vae_source: str | Path | None = None):
        repository, checkpoint = repository.resolve(), checkpoint.resolve()
        if not checkpoint.is_file() or not checkpoint.is_relative_to(ROOT.resolve()):
            raise FileNotFoundError('ProjectNet checkpoint must be inside the repository')
        sys.path.insert(0, str(repository / 'projectnet_training_3000'))
        from network import make_unet, make_noise_schedule
        from diffusers import AutoencoderKL
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('size') != 256:
            raise RuntimeError(f"Expected 256x256 ProjectNet; checkpoint states {state.get('size')}")
        self.provenance = {k: state.get(k) for k in ('training_source','label_type','epoch','vae_id')}
        self.unet = make_unet(256).to(device).eval()
        self.unet.load_state_dict(state['unet'])
        self.unet.requires_grad_(False)
        source = str(Path(vae_source).resolve()) if vae_source is not None else state['vae_id']
        self.vae = AutoencoderKL.from_pretrained(source).to(device).eval()
        self.vae.requires_grad_(False)
        self.schedule = make_noise_schedule()
        self.alpha = self.schedule.alphas_cumprod[100].to(device)
        self.device, self.seed = device, seed
        self.scene_latent = self.initial = None

    def prepare_scene(self, scene: torch.Tensor, seed: int | None = None):
        with torch.no_grad():
            normalized = scene.mul(2).sub(1)
            self.scene_latent = self.vae.encode(normalized).latent_dist.mode() * self.vae.config.scaling_factor
            generator = torch.Generator(device=self.device).manual_seed(
                self.seed if seed is None else seed)
            noise = torch.randn(self.scene_latent.shape, generator=generator, device=self.device)
            self.initial = self.alpha.sqrt()*self.scene_latent + (1-self.alpha).sqrt()*noise

    def __call__(self, scene: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
        if self.scene_latent is None:
            self.prepare_scene(scene)
        # Do not use no_grad: the frozen VAE and U-Net must transmit texture gradients.
        z_texture = self.vae.encode(texture.mul(2).sub(1)).latent_dist.mode() * self.vae.config.scaling_factor
        eps_hat = self.unet(torch.cat((self.initial, self.scene_latent, z_texture), dim=1), 100).sample
        predicted = (self.initial - (1-self.alpha).sqrt()*eps_hat)/self.alpha.sqrt()
        return self.vae.decode(predicted/self.vae.config.scaling_factor).sample.add(1).div(2).clamp(0,1)
