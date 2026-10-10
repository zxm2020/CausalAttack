"""Evaluate image reconstruction on scene/session-disjoint held-out real or synthetic triples."""
import argparse
import math
import torch
from network import PairDataset, load_predictor, sample_capture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='outputs/synthetic/projectnet_best.pt')
    p.add_argument('--manifest', required=True)
    p.add_argument('--allow-synthetic', action='store_true')
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--max-items', type=int, default=20)
    p.add_argument('--steps', type=int, default=30)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck, net, vae = load_predictor(args.weights, device)
    data = PairDataset(args.manifest, args.split, ck['size'], args.allow_synthetic)
    errors = []
    for i in range(min(args.max_items, len(data))):
        clean, texture, target = data[i]
        y = sample_capture(net, vae, clean.unsqueeze(0).to(device), texture.unsqueeze(0).to(device),
                           steps=args.steps, seed=2026 + i)[0]
        mse = ((y - ((target.to(device) + 1) / 2)).square().mean()).item()
        errors.append(mse)
    mean = sum(errors) / len(errors)
    print(f'{args.split}: N={len(errors)}, source={data.rows[0]["source"]}, '
          f'MSE_RGB_[0,1]={mean:.7f}, PSNR_dB={-10 * math.log10(max(mean, 1e-12)):.3f}')


if __name__ == '__main__':
    main()
