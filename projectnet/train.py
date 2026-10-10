"""Train independent ProjectNet-like conditional LDM and optional EWC on a real new session.

The API supplies images, NOT arbitrary custom model training. This script produces local
exportable weights. Requires a CUDA GPU and network access for the pretrained VAE.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from network import (VAE_ID, PairDataset, make_unet, make_noise_schedule,
                     denoise_loss, load_predictor)


def fisher_proxy(unet, vae, sched, loader, device, batches):
    # Diagonal importance estimated from squared minibatch gradients.
    unet.eval()
    state = {n: torch.zeros_like(p, dtype=torch.float32, device='cpu')
             for n, p in unet.named_parameters() if p.requires_grad}
    count = 0
    for batch in loader:
        if count >= batches:
            break
        unet.zero_grad(set_to_none=True)
        loss = denoise_loss(unet, vae, sched, batch, device)
        loss.backward()
        for n, p in unet.named_parameters():
            if n in state and p.grad is not None:
                state[n].add_(p.grad.detach().cpu().float().square())
        count += 1
    if count == 0:
        raise ValueError('Need batches for Fisher proxy')
    unet.zero_grad(set_to_none=True)
    return {'fisher': {n: f / count for n, f in state.items()},
            'reference': {n: p.detach().cpu().float().clone() for n, p in unet.named_parameters() if n in state}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', default='outputs/synthetic')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--vae-id', default=VAE_ID)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--allow-synthetic', action='store_true')
    p.add_argument('--resume', help='old projectnet.pt checkpoint for incremental stage')
    p.add_argument('--ewc', help='old ewc.pt checkpoint for incremental stage')
    p.add_argument('--ewc-weight', type=float, default=1.)
    p.add_argument('--fisher-batches', type=int, default=8)
    args = p.parse_args()
    if args.ewc and not args.resume:
        p.error('--ewc requires --resume')
    if not torch.cuda.is_available():
        raise RuntimeError('This training example needs an available CUDA GPU; inference API credits do not pay for GPU training.')
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train = PairDataset(args.manifest, 'train', args.size, args.allow_synthetic)
    val = PairDataset(args.manifest, 'val', args.size, args.allow_synthetic)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0)
    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device).eval()
    vae.requires_grad_(False)
    model = make_unet(args.size).to(device)
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        if previous['size'] != args.size or previous['vae_id'] != args.vae_id:
            raise ValueError('Resume checkpoint size/VAE mismatch')
        model.load_state_dict(previous['unet'])
    ewc = None
    if args.ewc:
        state = torch.load(args.ewc, map_location='cpu', weights_only=True)
        ewc = {group: {n: x.to(device) for n, x in params.items()} for group, params in state.items()}
        if set(ewc['fisher']) != {n for n, _ in model.named_parameters()}:
            raise ValueError('EWC keys do not match the current model')
    schedule = make_noise_schedule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    best_val = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = denoise_loss(model, vae, schedule, batch, device)
                if ewc is not None:
                    # Calculate regularizer in FP32; current theta remains differentiable.
                    penalty = sum((ewc['fisher'][n] * (v.float() - ewc['reference'][n]).square()).sum()
                                  for n, v in model.named_parameters())
                    loss = loss + args.ewc_weight * .5 * penalty
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_losses = [float(denoise_loss(model, vae, schedule, batch, device)) for batch in val_loader]
        row = {'epoch': epoch, 'train_loss': sum(losses) / len(losses),
               'validation_denoising_loss': sum(val_losses) / len(val_losses)}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {'unet': {n: x.detach().cpu() for n, x in model.state_dict().items()},
                      'size': args.size, 'vae_id': args.vae_id, 'epoch': epoch,
                      'training_source': sorted({r['source'] for r in train.rows}),
                      'architecture': 'independent_concat_conditioned_latent_diffusion_unet'}
        torch.save(checkpoint, output / 'projectnet_last.pt')
        if row['validation_denoising_loss'] < best_val:
            best_val = row['validation_denoising_loss']
            torch.save(checkpoint, output / 'projectnet_best.pt')
        (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    torch.save(fisher_proxy(model, vae, schedule, train_loader, device, args.fisher_batches), output / 'ewc.pt')
    print('Saved', output / 'projectnet_best.pt', 'and', output / 'ewc.pt', flush=True)


if __name__ == '__main__':
    main()
