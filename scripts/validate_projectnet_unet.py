"""Validate the local ProjectNet U-Net checkpoint without downloading the VAE."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from causalattack.paths import PROJECTNET_CHECKPOINT, PROJECTNET_ROOT  # noqa: E402
PROJECTNET = PROJECTNET_ROOT
sys.path.insert(0, str(PROJECTNET / 'projectnet_training_3000'))
from network import make_unet  # noqa: E402


checkpoint = PROJECTNET_CHECKPOINT
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = make_unet(256).eval().requires_grad_(False)
model.load_state_dict(state['unet'])
latent_input = torch.randn(1, 12, 32, 32, requires_grad=True)
prediction = model(latent_input, 100).sample
prediction.square().mean().backward()
result = {
    'checkpoint': str(checkpoint),
    'epoch': state.get('epoch'),
    'label_type': state.get('label_type'),
    'input_shape': list(latent_input.shape),
    'output_shape': list(prediction.shape),
    'finite_output': bool(torch.isfinite(prediction).all()),
    'input_gradient_norm': float(latent_input.grad.norm()),
    'parameter_count': sum(value.numel() for value in model.parameters()),
}
print(json.dumps(result, indent=2))
(ROOT / 'outputs' / 'projectnet_unet_validation.json').write_text(
    json.dumps(result, indent=2), encoding='utf-8')
