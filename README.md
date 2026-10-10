# CausalAttack

Official code and reproducibility package for:

> **CausalAttack: A Real-time Dynamic Light-based Physical Adversarial Attack Framework for Moving Targets**

CausalAttack generates target-local adversarial projection textures for dynamic
camera--projector environments. The released implementation includes ProjectNet
training and inference, reference-guided feature-distribution alignment, differentiable
LAB projector compensation, sharpness-aware texture optimization, post-update display
conversion, offline evaluation, and asynchronous live camera/projector execution.

## Contents

- [Installation](#installation)
- [Data and model weights](#data-and-model-weights)
- [Quick verification](#quick-verification)
- [Offline attack](#offline-attack)
- [Live camera and projector output](#live-camera-and-projector-output)
- [ProjectNet training](#projectnet-training)
- [Evaluation](#evaluation)
- [Configurations and implementation details](#configurations-and-implementation-details)
- [Repository structure](#repository-structure)

## Installation

Python 3.10--3.12 is recommended. Install a CUDA-enabled PyTorch build compatible
with the local NVIDIA driver before installing the remaining packages. CPU execution
is supported for verification and inference, but ProjectNet training requires a CUDA
GPU and full texture updates are substantially slower on CPU.

### Windows PowerShell

```powershell
git clone https://github.com/zxm2020/CausalAttack.git
cd CausalAttack
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch for the local CUDA version first; see https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

### Linux

```bash
git clone https://github.com/zxm2020/CausalAttack.git
cd CausalAttack
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch for the local CUDA version first.
python -m pip install -r requirements.txt
```

The code has been exercised with Python 3.12, PyTorch 2.14, Diffusers 0.39,
Transformers 4.57, and OpenCV 4.10. Exact package bounds are provided in
`requirements.txt`.

## Data and model weights

ProjectNet training data and released ProjectNet weights are available at:

- **Baidu Netdisk:** <https://pan.baidu.com/s/1GuhQHUnqOLi4aUuhCKuMxg?pwd=e62n>
- **Extraction code:** `e62n`

Large datasets and model binaries are intentionally not stored in Git. After download,
arrange the assets as follows:

```text
CausalAttack/
├── checkpoints/
│   ├── projectnet/
│   │   ├── projectnet_best.pt
│   │   └── ewc.pt
│   ├── sd-vae-ft-mse/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors (or .bin)
│   └── yolov5s.pt
└── data/
    └── projectnet/
        └── mixed_triplets_3000/
            ├── manifest.csv
            └── ... scene/texture/captured images ...
```

The frozen VAE is `stabilityai/sd-vae-ft-mse` in Diffusers format. It may be
downloaded from Hugging Face or placed at `checkpoints/sd-vae-ft-mse`. The detector
checkpoint is the YOLOv5s checkpoint used by the feature adapter and should be placed
at `checkpoints/yolov5s.pt`.

Verify the asset layout before running an experiment:

```bash
python scripts/check_assets.py
```

The script reports every required path and exits with a nonzero status if an asset is
missing. Use `python scripts/check_assets.py --training` to additionally validate the
Stage-I dataset manifest and EWC state. Do not commit downloaded checkpoints, raw datasets, captured imagery, API
tokens, or local `.env` files; these paths are covered by `.gitignore`.

## Quick verification

Run the unit tests for MMD, shared sampling indices, LAB compensation, gradients,
SAM/SGD updates, and final display conversion:

```bash
python -m unittest discover -s tests -v
```

Validate the ProjectNet conditional U-Net checkpoint and its 12-channel input /
4-channel noise-prediction output:

```bash
python scripts/validate_projectnet_unet.py
```

For a lightweight installation check that does not require released weights, the
following command uses the deterministic diagnostic backend:

```bash
python run_demo.py --smoke --output outputs/diagnostic
```

This command verifies the differentiable optimization and file-output path; it is not
used for the paper's physical measurements.

## Offline attack

The main single-observation entry point requires three aligned inputs:

1. `scene.png`: current camera observation containing the target;
2. `reference.png`: scene-matched target-free reference;
3. `target_mask.png`: target-local projection support as a grayscale mask.

```bash
python run_attack.py \
  --scene data/example/scene.png \
  --reference data/example/reference.png \
  --mask data/example/target_mask.png \
  --projectnet-checkpoint checkpoints/projectnet/projectnet_best.pt \
  --vae checkpoints/sd-vae-ft-mse \
  --yolo-weights checkpoints/yolov5s.pt \
  --features yolo \
  --steps 3 \
  --samples 1024 \
  --output outputs/example_attack
```

The output directory contains:

- `optimized_texture.png`: continuous optimized texture clipped only for image export;
- `continuous_texture.pt`: continuous tensor used for optimization/warm starts;
- `display_texture.png`: final clipped and 8-bit-quantized projector command;
- `predicted_capture.png`: ProjectNet prediction;
- `loss_curve.csv`: per-step MMD, LAB, total loss, and gradient norm;
- `metrics.json`: shapes, timing, feature backend, and optimization measurements.

The default display transform is an identity canvas mapping. A physical deployment
must replace it with the projector/camera geometry measured for that setup.

## Live camera and projector output

`run_live.py` separates continuous camera acquisition and display refresh from the
slower texture optimization. The projector window continuously shows the latest
completed texture while a background worker computes the next update. The subsequent
update warm-starts from the previous continuous texture, not from the clipped and
quantized display command.

```bash
python run_live.py \
  --source 0 \
  --backend projectnet \
  --features yolo \
  --projectnet-checkpoint checkpoints/projectnet/projectnet_best.pt \
  --vae checkpoints/sd-vae-ft-mse \
  --yolo-weights checkpoints/yolov5s.pt \
  --display-mode normalized \
  --output outputs/live_session
```

Interactive controls:

| Key | Action |
|---|---|
| `R` | Capture the current target-free reference frame |
| `I` | Select the target-local ROI |
| `S` | Start or pause asynchronous texture updates |
| `V` | Cycle normalized, amplified, and exact-command display modes |
| `C` | Clear the current texture and display command |
| `Q` / `Esc` | Exit |

`normalized` makes low-amplitude texture structure visible for inspection.
`amplified` applies the explicit `--display-gain`. `command` shows the exact
post-update clipped/quantized device command. Every completed update saves the exact
command independently of the selected visualization mode. Use `--display-x`,
`--display-y`, or `--fullscreen` to place the output window on a connected projector.

Camera acquisition rate and texture-update rate are distinct. A new pattern is
published only after all configured SAM/SGD steps finish; the previous completed
pattern remains displayed in the meantime.

## ProjectNet training

ProjectNet consists of a frozen `stabilityai/sd-vae-ft-mse` VAE and a trainable
conditional diffusion U-Net. RGB inputs are resized to 256x256 and encoded as
4x32x32 latents. Noisy capture, scene, and projection-texture latents form the
12-channel U-Net input; the network predicts four-channel diffusion noise. The
scheduler uses 1,000 training steps and the `squaredcos_cap_v2` beta schedule.

### Stage I: synthetic base training

The released dataset contains 3,000 triplets from 150 base scenes, with scene-disjoint
2,400/300/300 train/validation/test splits. Confirm that the downloaded manifest is at
`data/projectnet/mixed_triplets_3000/manifest.csv`, then run:

```bash
python run_train.py --stage base --epochs 20 --batch 4
```

The default optimizer is AdamW with learning rate `2e-4`. The VAE remains frozen.
Training checkpoints and audit metadata are written below `outputs/projectnet_base`.

The dataset generator is also included. Procedural generation does not require a
credential:

```bash
python projectnet/generate_data_modelscope.py \
  --source procedural --scenes 150 --pairs-per-scene 20 --size 256 \
  --out data/projectnet/mixed_triplets_3000
```

ModelScope-based scene generation is optional. Set `MODELSCOPE_SDK_TOKEN` in the
local environment; never place a token in source code or commit `key.txt`:

```bash
python projectnet/generate_data_modelscope.py \
  --source modelscope --scenes 150 --pairs-per-scene 20 --size 256 \
  --out data/projectnet/mixed_triplets_3000
```

### Stage II: measured-domain adaptation

Organize measured triplets by acquisition session:

```text
raw_real/<session>/<sample>/scene.png
raw_real/<session>/<sample>/texture.png
raw_real/<session>/<sample>/captured.png
```

Create the session-disjoint manifest and run EWC-regularized adaptation:

```bash
python projectnet/ingest_real.py \
  --input raw_real --out data/projectnet/real_triplets

python run_train.py --stage real \
  --real-manifest data/projectnet/real_triplets/manifest.csv \
  --base-checkpoint checkpoints/projectnet/projectnet_best.pt \
  --ewc checkpoints/projectnet/ewc.pt \
  --epochs 6 --batch 2
```

The measured set uses session-disjoint 60/20/20 train/validation/test splits. The
conditional noise-prediction objective is retained and EWC is added only during this
adaptation stage.

## Evaluation

Evaluate ProjectNet image fidelity and the saved reference-guided feature distance:

```bash
python run_evaluate.py \
  --output outputs/example_attack \
  --capture-target data/example/measured_capture.png \
  --samples 1024
```

When a capture target is supplied, the evaluator reports MSE, PSNR, and SSIM. MMD is
computed from the same fixed shared spatial sample indices used by the attack. Attack
success rate requires clean-qualified task labels and valid detection/classification/
tracking matches; the evaluator intentionally does not infer ASR from loss reduction.

## Configurations and implementation details

The complete paper settings are recorded in:

- `configs/paper.yaml`: attack optimization, MMD, ProjectNet inference, and display;
- `configs/projectnet.yaml`: architecture and two-stage training settings.

Core implementation details:

- ProjectNet online inference uses fixed `tau=100` and keeps VAE/U-Net parameters
  frozen while retaining gradients to the candidate texture.
- YOLOv5 features are taken from the runtime-verified stride-8 P3 neck output.
- MMD uses 1,024 spatial samples drawn uniformly with replacement, fixed shared
  indices, a Gaussian RBF kernel with bandwidth 1.0, and the biased squared estimator.
- The target-free reference features and reference-reference kernel are cached within
  a completed update.
- LAB compensation regularizes the A/B chromatic channels and is a soft loss.
- SAM uses one L2 inner ascent (`rho=0.05`) for each of three outer SGD steps.
- RGB clipping and 8-bit conversion occur only after a complete texture update.

## Repository structure

```text
causalattack/                 Core differentiable attack and model adapters
configs/                      Paper and ProjectNet configurations
projectnet/                   ProjectNet architecture, data, training, and evaluation
third_party/yolov5/           YOLOv5 model/feature code and upstream license
scripts/                      Asset validation and checkpoint validation
tests/                        Unit tests for objectives and optimization
run_attack.py                 Offline attack entry point
run_demo.py                   Single-observation implementation
run_live.py                   Asynchronous camera/projector execution
run_train.py                  ProjectNet Stage-I/Stage-II training wrapper
run_evaluate.py               Image/feature evaluation
```

## Responsible use

This code is intended for authorized robustness evaluation of camera-based perception
systems. Physical tests should be conducted on controlled equipment and scenes, with
appropriate authorization, protection of captured imagery, and no interference with
operational safety systems.
