"""Predict an image from an already trained P&C surrogate."""
import argparse
import numpy as np
import torch
from PIL import Image
from network import transform, load_predictor, sample_capture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='outputs/synthetic/projectnet_best.pt')
    p.add_argument('--scene', required=True)
    p.add_argument('--texture', required=True)
    p.add_argument('--out', default='predicted.png')
    p.add_argument('--steps', type=int, default=30)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state, model, vae = load_predictor(args.weights, device)
    n = transform(args.scene, state['size']).unsqueeze(0).to(device)
    d = transform(args.texture, state['size']).unsqueeze(0).to(device)
    image = sample_capture(model, vae, n, d, args.steps)[0]
    Image.fromarray((255 * image.permute(1, 2, 0).cpu().numpy()).round().astype(np.uint8)).save(args.out)
    print('Prediction saved to', args.out)


if __name__ == '__main__':
    main()
