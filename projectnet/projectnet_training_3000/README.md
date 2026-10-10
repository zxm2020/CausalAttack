# ProjectNet Stage-I training

This directory contains the conditional latent-diffusion training implementation used
for ProjectNet Stage I. Dataset preparation, released weights, environment setup,
training commands, validation, and Stage-II measured-domain adaptation are documented
in the repository's main [`README.md`](../../README.md#projectnet-training).

The default paper configuration uses 3,000 triplets, a scene-disjoint
2,400/300/300 split, 20 epochs, batch size 4, learning rate `2e-4`, a frozen
`stabilityai/sd-vae-ft-mse` VAE, and eight minibatches for diagonal importance
estimation. Run the public wrapper from the repository root:

```bash
python run_train.py --stage base --epochs 20 --batch 4
```
