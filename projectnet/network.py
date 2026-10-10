"""Independent, *not author-verified*, conditioned latent diffusion P&C approximation.

Two conditioning images are VAE-encoded independently and concatenated with the noisy
captured-image latent. A frozen VAE is used for both encoding and decoding. The model
is a stochastic *synthetic or measured image predictor*, NOT an exact physical simulator.
"""
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler, DDIMScheduler

VAE_ID = 'stabilityai/sd-vae-ft-mse'


def transform(path, resolution):
    image = Image.open(path).convert('RGB').resize((resolution, resolution), Image.Resampling.LANCZOS)
    array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.
    return torch.from_numpy(array).permute(2, 0, 1)


class PairDataset(Dataset):
    def __init__(self, manifest, split, size, allow_synth=False):
        self.root = Path(manifest).resolve().parent
        self.size = size
        with open(manifest, encoding='utf-8', newline='') as f:
            all_rows = list(csv.DictReader(f))
        self.rows = [r for r in all_rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No records for split={split}. Generate >= 3 complete scenes first.')
        if not allow_synth and any(r['source'] == 'synthetic_renderer' for r in self.rows):
            raise ValueError('Synthetic examples require --allow-synthetic, and must not be called camera captures.')
        for r in self.rows:
            for k in ('scene', 'texture', 'captured'):
                if not self.path(r[k]).is_file():
                    raise FileNotFoundError(self.path(r[k]))

    def path(self, name):
        p = Path(name)
        return p if p.is_absolute() else self.root / p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return tuple(transform(self.path(row[k]), self.size) for k in ('scene', 'texture', 'captured'))


def make_unet(size):
    if size % 64:
        raise ValueError('size must be divisible by 64')
    return UNet2DModel(sample_size=size // 8, in_channels=12, out_channels=4,
                       layers_per_block=1, block_out_channels=(64, 128, 128),
                       down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'),
                       up_block_types=('AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
                       norm_num_groups=16)


def make_noise_schedule():
    return DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                         prediction_type='epsilon', clip_sample=False)


def vae_features(vae, scene, texture, captured=None):
    # Frozen VAE; no activations stored for backprop through VAE.
    with torch.no_grad():
        scale = vae.config.scaling_factor
        a = vae.encode(scene).latent_dist.mode() * scale
        b = vae.encode(texture).latent_dist.mode() * scale
        c = vae.encode(captured).latent_dist.mode() * scale if captured is not None else None
    return a, b, c


def denoise_loss(unet, vae, noise_schedule, batch, device):
    clean, texture, captured = [x.to(device, non_blocking=True) for x in batch]
    n, d, y = vae_features(vae, clean, texture, captured)
    eps = torch.randn_like(y)
    t = torch.randint(0, noise_schedule.config.num_train_timesteps, (y.shape[0],), device=device)
    noisy = noise_schedule.add_noise(y, eps, t)
    predicted = unet(torch.cat((noisy, n, d), dim=1), t).sample
    return torch.nn.functional.mse_loss(predicted.float(), eps.float())


@torch.no_grad()
def sample_capture(unet, vae, clean, texture, steps=30, seed=2026):
    if clean.shape != texture.shape or clean.shape[0] != 1:
        raise ValueError('Expect two images of matching shape [1,3,H,W]')
    device = clean.device
    n, d, _ = vae_features(vae, clean, texture)
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n.shape, generator=generator, device=device, dtype=n.dtype)
    schedule = DDIMScheduler.from_config(make_noise_schedule().config)
    schedule.set_timesteps(steps, device=device)
    for t in schedule.timesteps:
        eps = unet(torch.cat((z, n, d), dim=1), t).sample
        z = schedule.step(eps, t, z, eta=0.).prev_sample
    return (vae.decode(z / vae.config.scaling_factor).sample / 2 + .5).clamp(0, 1)


def load_predictor(checkpoint, device):
    # Only load a checkpoint that you made yourself or otherwise trust.
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    unet = make_unet(ck['size']).to(device).eval()
    unet.load_state_dict(ck['unet'])
    vae = AutoencoderKL.from_pretrained(ck['vae_id']).to(device).eval()
    vae.requires_grad_(False)
    return ck, unet, vae
