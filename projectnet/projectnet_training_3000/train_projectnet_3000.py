#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProjectNet synthetic pretraining on 3000 scene/texture/rendered-capture triplets.

Drop this file next to the EXISTING network.py in projectnet_modelscope_complete.
The ModelScope API is NOT called by this program. GPU trains and exports a local UNet.
This implements the conditional latent diffusion ProjectNet training stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from diffusers import AutoencoderKL
from network import VAE_ID, make_unet, make_noise_schedule

REQUIRED = ("scene", "texture", "captured", "split", "source", "scene_id")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="e.g. /data0/zxm/PR/projectnet_modelscope_complete/data/mixed_triplets_3000")
    ap.add_argument("--out", type=Path, default=Path("outputs/projectnet_mixed_3000"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cache-batch", type=int, default=2, help="number of triplets per VAE encoding batch")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--vae-id", default=VAE_ID,
                    help="model name OR absolute path to a locally cached AutoencoderKL")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--expected-pairs", type=int, default=3000,
                    help="required manifest rows; use 0 only if intentionally training on incomplete data")
    ap.add_argument("--allow-synthetic", action="store_true", required=True,
                    help="explicitly acknowledge that 'captured' labels are SYNTHETIC, not real camera images")
    ap.add_argument("--resume", action="store_true", help="resume from OUT/train_state.pt")
    ap.add_argument("--fisher-batches", type=int, default=8,
                    help="0 disables optional EWC proxy; >0 computes for later real-data adaptation")
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.cache_batch < 1 or args.size % 64:
        ap.error("epochs, batch, cache-batch must be positive; size must be divisible by 64")
    if args.lr <= 0 or args.expected_pairs < 0 or args.fisher_batches < 0 or args.save_every < 1:
        ap.error("invalid lr / expected-pairs / fisher-batches / save-every")
    return args


def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def validate_manifest(root: Path, manifest: Path, expected_pairs: int, size: int):
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; generation must finish and write manifest.csv first")
    raw = manifest.read_bytes()
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in REQUIRED):
            raise ValueError(f"manifest columns must contain {REQUIRED}; got {reader.fieldnames}")
        rows = list(reader)
    if expected_pairs and len(rows) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} triplets but manifest contains {len(rows)}. "
                         "Do not start '3000 sample' training on an incomplete dataset.")
    if not rows:
        raise ValueError("Empty dataset")
    allowed = {"train", "val", "test"}
    groups = defaultdict(set)
    split_counts = Counter()
    sources = Counter()
    duplicates = set()
    seen = set()
    for i, row in enumerate(rows):
        split = row["split"]
        if split not in allowed:
            raise ValueError(f"Row {i}: invalid split {split!r}")
        if not row["scene_id"]:
            raise ValueError(f"Row {i}: missing scene_id")
        if not row["source"]:
            raise ValueError(f"Row {i}: missing source provenance")
        triple = tuple(row[k] for k in ("scene", "texture", "captured"))
        if triple in seen:
            duplicates.add(triple)
        seen.add(triple)
        groups[row["scene_id"]].add(split)
        split_counts[split] += 1
        sources[row["source"]] += 1
        for key in ("scene", "texture", "captured"):
            fpath = Path(row[key])
            if not fpath.is_absolute():
                fpath = root / fpath
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing image: row={i}, field={key}, path={fpath}")
            # Verify actual decodability/dimensions at latent encoding; here use headers for quick audit.
            with Image.open(fpath) as img:
                if img.size != (size, size):
                    raise ValueError(f"Image size mismatch at {fpath}: {img.size} != {(size, size)}")
    if duplicates:
        raise ValueError(f"Found {len(duplicates)} duplicated triplet paths")
    leaked = {scene: splits for scene, splits in groups.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"Scene split leakage: {str(leaked)[:250]}")
    if not split_counts["train"] or not split_counts["val"] or not split_counts["test"]:
        raise ValueError(f"Expected nonempty train/val/test; got {dict(split_counts)}")
    # Explicit synthetic provenance: this script only handles stage-1 synthetic pretraining.
    if not all(("renderer" in source or "synthetic" in source or "procedural" in source)
               for source in sources):
        raise ValueError(f"Non-synthetic sources found {dict(sources)}. Use a separate real-data stage.")
    stats = {"triplets": len(rows), "scenes": len(groups), "split_counts": dict(split_counts),
             "source_counts": dict(sources), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
             "label_type": "synthetic_renderer_not_real_camera_capture"}
    print("[DATA AUDIT]", json.dumps(stats, ensure_ascii=False), flush=True)
    return rows, stats


class ImageTripleDataset(Dataset):
    def __init__(self, root, rows, size):
        self.root, self.rows, self.size = root, rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        items = []
        for name in ("scene", "texture", "captured"):
            path = Path(r[name])
            if not path.is_absolute():
                path = self.root / path
            with Image.open(path) as im:
                x = np.asarray(im.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS),
                               dtype=np.float32).copy() / 127.5 - 1.0
            items.append(torch.from_numpy(x).permute(2, 0, 1))
        return tuple(items)


def get_or_build_cache(root, rows, stats, args, vae, device, out):
    cache_dir = out / "latent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({"manifest": stats["manifest_sha256"],
                                     "root": str(root.resolve()), "size": args.size,
                                     "vae": args.vae_id}, sort_keys=True).encode()).hexdigest()
    cached = {}
    for split in ("train", "val"):
        cache_file = cache_dir / f"{split}_{key[:16]}.pt"
        if cache_file.is_file():
            saved = torch.load(cache_file, map_location="cpu", weights_only=True)
            if saved["key"] != key or len(saved["z_scene"]) != sum(r["split"] == split for r in rows):
                raise ValueError(f"Latent cache invalid: {cache_file}")
            print(f"[CACHE] Reusing {split}: {cache_file}", flush=True)
            cached[split] = saved
            continue
        subrows = [r for r in rows if r["split"] == split]
        loader = DataLoader(ImageTripleDataset(root, subrows, args.size),
                            batch_size=args.cache_batch, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        buckets = [[], [], []]
        vae.eval()
        for batch_idx, batch in enumerate(loader):
            with torch.inference_mode():
                # Encode all 3 images in one VAE pass; mode() gives deterministic targets.
                input_images = torch.cat([x.to(device, non_blocking=True) for x in batch], dim=0)
                z = vae.encode(input_images).latent_dist.mode() * vae.config.scaling_factor
                z_scene, z_texture, z_captured = z.chunk(3, dim=0)
                for dst, src in zip(buckets, (z_scene, z_texture, z_captured)):
                    dst.append(src.cpu().to(dtype=torch.float16))
            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(loader):
                print(f"[CACHE] {split}: {batch_idx + 1}/{len(loader)} batches encoded", flush=True)
        saved = {"key": key, "z_scene": torch.cat(buckets[0]),
                 "z_texture": torch.cat(buckets[1]), "z_captured": torch.cat(buckets[2])}
        atomic_save(saved, cache_file)
        print(f"[CACHE] Saved {split}: {cache_file}", flush=True)
        cached[split] = saved
    return cached


def diffusion_loss(net, schedule, latent_batch, device, generator=None):
    z_scene, z_texture, z_target = [x.to(device, non_blocking=True).float() for x in latent_batch]
    epsilon = torch.randn(z_target.shape, device=device, dtype=z_target.dtype, generator=generator)
    steps = torch.randint(0, schedule.config.num_train_timesteps, (len(z_target),),
                          device=device, generator=generator)
    z_noisy = schedule.add_noise(z_target, epsilon, steps)
    epsilon_hat = net(torch.cat([z_noisy, z_scene, z_texture], dim=1), steps).sample
    return torch.nn.functional.mse_loss(epsilon_hat.float(), epsilon.float())


def create_checkpoint(model, args, epoch, stats):
    # network.py load_predictor() reads exactly these keys.
    return {"unet": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "size": args.size, "vae_id": args.vae_id, "epoch": epoch,
            "training_source": sorted(stats["source_counts"]),
            "architecture": "concat_conditioned_latent_diffusion_unet",
            "manifest_sha256": stats["manifest_sha256"], "label_type": stats["label_type"]}


def compute_fisher(model, train_loader, schedule, device, batches):
    """Estimate diagonal importance from eight squared minibatch gradients."""
    model.eval()
    fisher = {name: torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32)
              for name, p in model.named_parameters()}
    done = 0
    for batch in train_loader:
        if done >= batches:
            break
        model.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, schedule, batch, device)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().float().cpu().square()
        done += 1
    model.zero_grad(set_to_none=True)
    if not done:
        raise RuntimeError("Cannot estimate Fisher without batches")
    return {"fisher": {k: v / done for k, v in fisher.items()},
            "reference": {k: p.detach().float().cpu().clone() for k, p in model.named_parameters()}}


def main():
    args = parse_args()
    root = args.data_root.resolve()
    manifest = root / "manifest.csv"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, stats = validate_manifest(root, manifest, args.expected_pairs, args.size)
    atomic_json(stats, out / "dataset_audit.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found; verify projectnet conda environment")
    device = torch.device("cuda:0")
    print(f"[GPU] {torch.cuda.get_device_name(device)}, torch={torch.__version__}", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device).eval()
    vae.requires_grad_(False)
    latent = get_or_build_cache(root, rows, stats, args, vae, device, out)
    del vae
    torch.cuda.empty_cache()
    datasets = {split: TensorDataset(item["z_scene"], item["z_texture"], item["z_captured"])
                for split, item in latent.items()}
    train_loader = DataLoader(datasets["train"], batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = make_unet(args.size).to(device)
    schedule = make_noise_schedule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    history, best_val, start_epoch = [], math.inf, 1
    state_file = out / "train_state.pt"
    if args.resume:
        if not state_file.is_file():
            raise FileNotFoundError(f"--resume requested, but missing {state_file}")
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != stats["manifest_sha256"] or state["size"] != args.size or state["vae_id"] != args.vae_id:
            raise ValueError("Cannot resume: dataset / resolution / VAE differs from checkpoint")
        model.load_state_dict(state["unet"])
        optimizer.load_state_dict(state["optimizer"])
        if not use_bf16 and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best_val, history = state["best_val"], state["history"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"], device=device)
        print(f"[RESUME] epoch {state['epoch']} -> {start_epoch}; best_val={best_val:.6f}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = diffusion_loss(model, schedule, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss epoch={epoch} batch={batch_idx}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            steps += 1
            if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(train_loader):
                print(f"[EPOCH {epoch}/{args.epochs}] batch {batch_idx + 1}/{len(train_loader)} loss={float(loss):.6f}", flush=True)
        # Fixed validation noise/timesteps per batch across epochs, for meaningful selection.
        model.eval()
        val_total, val_steps = 0.0, 0
        with torch.inference_mode():
            for idx, batch in enumerate(val_loader):
                gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + idx)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    v = diffusion_loss(model, schedule, batch, device, generator=gen)
                val_total += float(v)
                val_steps += 1
        val_mean = val_total / val_steps
        row = {"epoch": epoch, "train_denoising_mse": total / steps,
               "val_denoising_mse": val_mean, "train_pairs": len(datasets["train"]),
               "val_pairs": len(datasets["val"])}
        history.append(row)
        print("[EPOCH RESULT]", json.dumps(row), flush=True)
        checkpoint = create_checkpoint(model, args, epoch, stats)
        if val_mean < best_val:
            best_val = val_mean
            atomic_save(checkpoint, out / "projectnet_best.pt")
            print(f"[BEST] val={best_val:.6f} -> {out / 'projectnet_best.pt'}", flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            atomic_save(checkpoint, out / "projectnet_last.pt")
            # Persist optimizer/RNG to allow exact restart at the next epoch boundary.
            state = {**checkpoint, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "history": history, "best_val": best_val,
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_state": torch.cuda.get_rng_state(device)}
            atomic_save(state, state_file)
        atomic_json(history, out / "history.json")
    # EWC is computed at the BEST stage-1 checkpoint. No Fisher from test/val data.
    if args.fisher_batches and not (args.resume and start_epoch > args.epochs and (out / "ewc.pt").exists()):
        best_file = out / "projectnet_best.pt"
        if not best_file.is_file():
            raise FileNotFoundError(f"Cannot compute Fisher: missing best checkpoint {best_file}")
        best_ckpt = torch.load(best_file, map_location="cpu", weights_only=True)
        model.load_state_dict(best_ckpt["unet"])
        print("[EWC] Estimating diagonal gradient-squared proxy on training data", flush=True)
        state = compute_fisher(model, train_loader, schedule, device, args.fisher_batches)
        state["manifest_sha256"] = stats["manifest_sha256"]
        state["reference_checkpoint"] = "projectnet_best.pt"
        state["type"] = "minibatch_squared_gradient_proxy_not_exact_Fisher"
        atomic_save(state, out / "ewc.pt")
    print("[DONE]", out, flush=True)
    print("[OUTPUT]", out / "projectnet_best.pt", flush=True)
    print("[OUTPUT]", out / "projectnet_last.pt", flush=True)
    print("[OUTPUT]", out / "history.json", flush=True)
    if args.fisher_batches:
        print("[OUTPUT]", out / "ewc.pt", flush=True)
    print("[NOTE] This is synthetic P&C pretraining; validate on actual measured projector-camera triples before claiming real-world reproduction.")


if __name__ == "__main__":
    main()
