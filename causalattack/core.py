"""Differentiable objective and completed-update optimization.

The display conversion is deliberately outside the SGD/SAM loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def spatial_rows(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"Expected [1,C,H,W] feature, got {tuple(feature.shape)}")
    return feature.flatten(2).transpose(1, 2)[0]


def shared_indices(feature: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    positions = feature.shape[-2] * feature.shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(positions, (count,), generator=generator).to(feature.device)


def rbf(a: torch.Tensor, b: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("MMD inputs must be matrices with matching feature dimension")
    distances = torch.cdist(a, b).square()
    return torch.exp(-distances / (2.0 * bandwidth * bandwidth))


def biased_mmd_squared(a: torch.Tensor, b: torch.Tensor, bandwidth: float = 1.0,
                       cached_aa: torch.Tensor | None = None) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Paired MMD samples differ: {tuple(a.shape)} != {tuple(b.shape)}")
    aa = rbf(a, a, bandwidth).mean() if cached_aa is None else cached_aa
    return aa + rbf(b, b, bandwidth).mean() - 2.0 * rbf(a, b, bandwidth).mean()


def rgb_to_lab(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable CIELAB conversion; RGB values are continuous, not hard-clipped."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB texture must be [B,3,H,W]")
    linear = torch.where(rgb <= 0.04045, rgb / 12.92,
                         ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4))
    r, g, b = linear[:, 0], linear[:, 1], linear[:, 2]
    x = (0.4124564*r + 0.3575761*g + 0.1804375*b) / 0.95047
    y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    z = (0.0193339*r + 0.1191920*g + 0.9503041*b) / 1.08883
    def f(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value > 0.008856, value.clamp_min(0).pow(1/3),
                           7.787037 * value + 16/116)
    fx, fy, fz = f(x), f(y), f(z)
    return 116*fy-16, 500*(fx-fy), 200*(fy-fz)


def lab_ab_loss(texture: torch.Tensor) -> torch.Tensor:
    _, a, b = rgb_to_lab(texture)
    return (a.square() + b.square()).sum()


def display_command(texture: torch.Tensor, canvas_hw: tuple[int, int] | None = None
                    ) -> torch.Tensor:
    """Final clip/quantize; no geometry is claimed when canvas_hw is omitted."""
    mapped = texture if canvas_hw is None else F.interpolate(texture, size=canvas_hw,
                                                               mode="bilinear", align_corners=False)
    return (mapped.clamp(0, 1) * 255).round().to(torch.uint8)


@dataclass
class AttackSettings:
    outer_steps: int = 3
    learning_rate: float = 0.01
    sam_radius: float = 0.05
    samples: int = 1024
    bandwidth: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    l2_weight: float = 1e-6
    seed: int = 2026


def optimize(scene: torch.Tensor, reference: torch.Tensor, projector,
             features, settings: AttackSettings, initial: torch.Tensor | None = None,
             mask: torch.Tensor | None = None):
    """One completed update: E SAM inner/outer evaluations, then return continuous texture."""
    if scene.shape != reference.shape or scene.ndim != 4 or scene.shape[0] != 1:
        raise ValueError("Scene/reference must be matching [1,3,H,W] images")
    if mask is None:
        mask = torch.ones_like(scene[:, :1])
    if mask.shape != scene[:, :1].shape or not torch.isfinite(mask).all():
        raise ValueError("Target mask must be a finite [1,1,H,W] tensor")
    mask = mask.clamp(0, 1)
    texture = torch.nn.Parameter(torch.zeros_like(scene) if initial is None else initial.detach().clone())
    optimizer = torch.optim.SGD([texture], lr=settings.learning_rate)
    with torch.no_grad():
        ref_map = features(reference)
        indices = shared_indices(ref_map, settings.samples, settings.seed)
        ref_rows = spatial_rows(ref_map)[indices].detach()
        cached_aa = rbf(ref_rows, ref_rows, settings.bandwidth).mean().detach()

    def mmd(candidate):
        prediction = projector(scene, candidate * mask)
        attack_map = features(prediction)
        if attack_map.shape[-2:] != ref_map.shape[-2:]:
            raise ValueError("Reference and attacked feature grids differ")
        attacked = spatial_rows(attack_map)[indices]
        return biased_mmd_squared(ref_rows, attacked, settings.bandwidth, cached_aa)

    history = []
    start = time.perf_counter()
    first = float(mmd(texture).detach())
    for step in range(settings.outer_steps):
        optimizer.zero_grad(set_to_none=True)
        inner = mmd(texture)
        gradient = torch.autograd.grad(inner, texture)[0]
        norm = float(gradient.norm().detach())
        if not torch.isfinite(gradient).all() or norm <= 0:
            raise RuntimeError("MMD gradient to continuous texture is invalid or zero")
        displacement = settings.sam_radius * gradient / (gradient.norm() + 1e-12)
        outer = mmd(texture + displacement.detach())
        chroma = lab_ab_loss(texture * mask)
        total = settings.alpha*chroma + settings.beta*outer + settings.l2_weight*(texture*mask).square().sum()
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite attack loss")
        total.backward()
        optimizer.step()
        history.append(dict(step=step+1, inner_mmd=float(inner.detach()),
                            sam_mmd=float(outer.detach()), lab_ab=float(chroma.detach()),
                            total_loss=float(total.detach()), gradient_norm=norm))
    elapsed = time.perf_counter() - start
    final_mmd = float(mmd(texture).detach())
    return (texture*mask).detach(), history, dict(initial_mmd=first, final_mmd=final_mmd,
                                          elapsed_seconds=elapsed,
                                          final_texture_norm=float(texture.detach().norm()))
