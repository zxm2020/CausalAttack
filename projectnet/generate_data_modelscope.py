#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate ProjectNet triplet data with ModelScope API.

Triplet meaning:
    texture  = A = projection image (delta_d)
    scene    = B = physical scene (n)
    captured = C = synthesized projected capture (I)

This keeps manifest compatible with existing train.py:
    scene, texture, captured, split, source, scene_id, session

Main features:
- Uses ModelScope only to generate the physical scene B.
- Generates diverse projection textures A locally.
- Synthesizes captured image C locally (A + B -> C).
- Supports mixed-object scene generation.
- Supports resume / timeout recovery by caching task_id per scene.
- Saves triplet visualization images.

Author-facing note:
- captured.png is synthetic, NOT a real camera-captured projector image.
- Good for synthetic pretraining of ProjectNet, not a substitute for real P&C data.
"""

import argparse
import csv
import io
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter


BASE_URL = "https://api-inference.modelscope.cn"
CONFIG_NAME = "dataset_config.json"
SCENE_META_NAME = "scene_metadata.json"

DEFAULT_OBJECT_POOL = [
    "toy truck",
    "toy car",
    "mug",
    "water bottle",
    "backpack",
    "book",
    "shoe",
    "potted plant",
    "teddy bear",
    "toolbox",
    "helmet",
    "speaker",
    "desk lamp",
    "small chair",
    "small suitcase",
    "keyboard",
]


# ============================================================
# Basic utilities
# ============================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pil_to_np01(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def np01_to_pil(x: np.ndarray) -> Image.Image:
    x = np.clip(x, 0.0, 1.0)
    return Image.fromarray((x * 255).round().astype(np.uint8), "RGB")


def get_token() -> str:
    token = os.getenv("MODELSCOPE_SDK_TOKEN", "").strip()
    if token:
        if token.startswith("Bearer "):
            token = token[len("Bearer "):].strip()
        return token

    key_file = Path("key.txt")
    if key_file.exists():
        token = key_file.read_text(encoding="utf-8").strip()
        if token.startswith("Bearer "):
            token = token[len("Bearer "):].strip()
        if token:
            return token

    raise RuntimeError(
        "No ModelScope token found.\n"
        "Please either:\n"
        "  1) export MODELSCOPE_SDK_TOKEN=...\n"
        "  2) create key.txt in the project directory"
    )


def checked_json(response: requests.Response):
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"ModelScope HTTP {response.status_code}: {response.text[:800]}"
        ) from exc
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse JSON response: {response.text[:800]}"
        ) from exc


# ============================================================
# Prompting
# ============================================================

def build_scene_prompt(scene_idx: int, target_object: str) -> str:
    """
    Ask ModelScope to generate the physical scene only, before projection.
    """
    return (
        f"A realistic indoor experimental scene with one main target object: {target_object}. "
        f"The object is clearly visible near the center or lower-center area of the image. "
        f"The background is simple and clean, such as a desk, wall, shelf, curtain, corridor, "
        f"office, room, or laboratory-like indoor environment. "
        f"No projected light pattern, no colorful illumination, no text, no watermark. "
        f"This is the raw physical scene before projection. "
        f"Scene variation {scene_idx}."
    )


# ============================================================
# ModelScope API
# ============================================================

def extract_output_images(data: dict):
    urls = data.get("output_images")
    if urls:
        return urls

    outputs = data.get("outputs")
    if isinstance(outputs, dict):
        urls = outputs.get("output_images")
        if urls:
            return urls
        urls = outputs.get("images")
        if urls:
            return urls

    return []


def submit_generation_task(token: str, model: str, prompt: str) -> str:
    response = requests.post(
        f"{BASE_URL}/v1/images/generations",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        },
        json={
            "model": model,
            "prompt": prompt,
        },
        timeout=60,
    )
    result = checked_json(response)
    task_id = result.get("task_id")
    if not task_id:
        raise RuntimeError(f"Task creation failed, response={result}")
    return task_id


def query_task(token: str, task_id: str) -> dict:
    response = requests.get(
        f"{BASE_URL}/v1/tasks/{task_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "X-ModelScope-Task-Type": "image_generation",
        },
        timeout=60,
    )
    return checked_json(response)


def download_image(url: str, size: int) -> Image.Image:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC)


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def procedural_scene(size: int, rng: random.Random, target_object: str) -> Image.Image:
    """
    Offline fallback scene generator if source=procedural.
    This is only a smoke-test fallback, not the main intended path.
    """
    img = Image.new("RGB", (size, size), (rng.randint(180, 235), rng.randint(180, 235), rng.randint(180, 235)))
    draw = ImageDraw.Draw(img)

    # background floor / wall split
    wall_y = int(size * rng.uniform(0.40, 0.60))
    draw.rectangle((0, 0, size, wall_y), fill=(rng.randint(180, 240), rng.randint(180, 240), rng.randint(180, 240)))
    draw.rectangle((0, wall_y, size, size), fill=(rng.randint(120, 180), rng.randint(120, 180), rng.randint(120, 180)))

    # simple main object blob/shape
    cx = size // 2 + rng.randint(-20, 20)
    cy = int(size * 0.62) + rng.randint(-15, 15)
    ow = rng.randint(size // 5, size // 3)
    oh = rng.randint(size // 6, size // 2)

    if any(k in target_object.lower() for k in ["bottle", "lamp", "plant"]):
        draw.rounded_rectangle((cx - ow // 3, cy - oh // 2, cx + ow // 3, cy + oh // 2), radius=10, fill=(220, 220, 220))
    elif any(k in target_object.lower() for k in ["book", "keyboard", "toolbox", "suitcase"]):
        draw.rounded_rectangle((cx - ow // 2, cy - oh // 4, cx + ow // 2, cy + oh // 4), radius=8, fill=(210, 210, 210))
    else:
        draw.ellipse((cx - ow // 2, cy - oh // 2, cx + ow // 2, cy + oh // 2), fill=(220, 220, 220))

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def generate_or_resume_scene(
    source: str,
    token: str,
    model: str,
    prompt: str,
    size: int,
    scene_dir: Path,
    target_object: str,
    max_wait: int,
    poll_interval: int,
    scene_idx: int,
    rng: random.Random,
):
    """
    Resume-capable scene generation.
    scene_dir contains:
        scene.png
        meta.json
    """
    ensure_dir(scene_dir)
    scene_path = scene_dir / "scene.png"
    meta_path = scene_dir / "meta.json"

    if scene_path.exists():
        scene = Image.open(scene_path).convert("RGB")
        meta = read_json(meta_path) if meta_path.exists() else {}
        return scene, meta

    # Procedural fallback
    if source == "procedural":
        scene = procedural_scene(size, rng, target_object)
        meta = {
            "created_at": now_str(),
            "source": "procedural",
            "model": "procedural",
            "task_id": None,
            "target_object": target_object,
            "scene_prompt": prompt,
            "status": "SUCCEED",
            "raw_output_images": [],
        }
        scene.save(scene_path)
        write_json(meta_path, meta)
        return scene, meta

    # ModelScope main path
    meta = {}
    task_id = None

    if meta_path.exists():
        try:
            meta = read_json(meta_path)
            task_id = meta.get("task_id")
        except Exception:
            meta = {}
            task_id = None

    if task_id is None:
        print(f"Requesting ModelScope scene {scene_idx + 1}", flush=True)
        task_id = submit_generation_task(token, model, prompt)
        meta = {
            "created_at": now_str(),
            "source": "modelscope",
            "model": model,
            "task_id": task_id,
            "target_object": target_object,
            "scene_prompt": prompt,
            "status": "SUBMITTED",
            "raw_output_images": [],
        }
        write_json(meta_path, meta)
        print(f"Submitted task_id: {task_id}", flush=True)
    else:
        print(f"Resuming existing task_id for scene {scene_idx + 1}: {task_id}", flush=True)

    start = time.time()
    last_data = None

    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            meta["last_checked_at"] = now_str()
            meta["status"] = "TIMEOUT"
            if last_data is not None:
                meta["last_raw_status"] = last_data.get("task_status")
            write_json(meta_path, meta)
            raise TimeoutError(
                f"Task {task_id} timed out after {max_wait} seconds.\n"
                f"Task id has been saved in {meta_path}. Re-run the SAME command later to continue querying it."
            )

        data = query_task(token, task_id)
        last_data = data
        status = data.get("task_status")
        print(f"Task status: {status}", flush=True)

        meta["last_checked_at"] = now_str()
        meta["status"] = status

        if status == "SUCCEED":
            urls = extract_output_images(data)
            if not urls:
                write_json(meta_path, meta)
                raise RuntimeError(f"Task succeeded but no image URL found. Raw data={data}")
            scene = download_image(urls[0], size)
            scene.save(scene_path)
            meta["raw_output_images"] = urls
            meta["status"] = "SUCCEED"
            write_json(meta_path, meta)
            return scene, meta

        if status == "FAILED":
            meta["status"] = "FAILED"
            write_json(meta_path, meta)
            raise RuntimeError(f"ModelScope task failed: {data}")

        write_json(meta_path, meta)
        time.sleep(poll_interval)


# ============================================================
# Texture generation
# ============================================================

def texture_blocks(size: int, rng: random.Random) -> Image.Image:
    grid = rng.choice([8, 10, 12, 14])
    cell = max(1, size // grid)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(grid):
        for j in range(grid):
            color = [rng.randint(0, 255) for _ in range(3)]
            y1, y2 = i * cell, min((i + 1) * cell, size)
            x1, x2 = j * cell, min((j + 1) * cell, size)
            canvas[y1:y2, x1:x2] = color
    img = Image.fromarray(canvas, "RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))


def texture_stripes(size: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(img)
    direction = rng.choice(["h", "v", "d1", "d2"])
    num = rng.randint(10, 22)

    for i in range(num):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if direction == "h":
            y1 = int(i * size / num)
            y2 = int((i + 1) * size / num)
            draw.rectangle((0, y1, size, y2), fill=color)
        elif direction == "v":
            x1 = int(i * size / num)
            x2 = int((i + 1) * size / num)
            draw.rectangle((x1, 0, x2, size), fill=color)
        elif direction == "d1":
            offset = int(i * size / num)
            draw.line((offset, 0, 0, offset), fill=color, width=rng.randint(8, 20))
        else:
            offset = int(i * size / num)
            draw.line((size - offset, 0, size, offset), fill=color, width=rng.randint(8, 20))
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def texture_circles(size: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), tuple(rng.randint(0, 40) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(25, 60)):
        r = rng.randint(size // 30, size // 8)
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def make_projection_texture(size: int, rng: random.Random) -> Image.Image:
    mode = rng.choice(["blocks", "stripes", "circles"])
    if mode == "blocks":
        return texture_blocks(size, rng)
    elif mode == "stripes":
        return texture_stripes(size, rng)
    else:
        return texture_circles(size, rng)


# ============================================================
# Synthetic renderer: A + B -> C
# ============================================================

def smooth_box_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int, blur: int = 21):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x1, y1, x2, y2), fill=255)
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return arr[..., None]


def make_target_mask(size: int, target_object: str):
    """
    Generic projection area for mixed-object scenes.
    We do not assume exact segmentation.
    """
    h = size
    w = size
    obj = target_object.lower()

    # tall objects
    if any(k in obj for k in ["person", "lamp", "plant", "chair"]):
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        y1 = int(h * 0.15)
        y2 = int(h * 0.90)

    # wide / flat tabletop objects
    elif any(k in obj for k in ["keyboard", "book", "toolbox", "suitcase"]):
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        y1 = int(h * 0.35)
        y2 = int(h * 0.82)

    # small / medium tabletop objects
    else:
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        y1 = int(h * 0.28)
        y2 = int(h * 0.82)

    blur = max(6, size // 20)
    return smooth_box_mask(h, w, x1, y1, x2, y2, blur=blur)


def synthesize_capture(scene: Image.Image,
                       texture: Image.Image,
                       target_object: str,
                       rng: random.Random) -> Image.Image:
    """
    Lightweight synthetic projector-camera renderer.
    """
    scene = scene.convert("RGB")
    texture = texture.convert("RGB").resize(scene.size, Image.Resampling.BICUBIC)

    s = pil_to_np01(scene)
    p = pil_to_np01(texture)

    h, w, _ = s.shape
    mask = make_target_mask(w, target_object)

    gamma = rng.uniform(1.8, 2.3)
    strength = rng.uniform(0.42, 0.72)
    ambient_scale = rng.uniform(0.96, 1.04)

    # linearize approx
    s_lin = np.clip(s, 0.0, 1.0) ** gamma
    p_lin = np.clip(p, 0.0, 1.0) ** gamma

    # slight blur on projected pattern
    p_img = np01_to_pil(p_lin).filter(ImageFilter.GaussianBlur(radius=max(1, w // 128)))
    p_lin = pil_to_np01(p_img)

    out_lin = np.clip(ambient_scale * s_lin + strength * mask * p_lin, 0.0, 1.0)

    # mild camera noise
    noise_std = rng.uniform(0.003, 0.012)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, noise_std, out_lin.shape)
    out_lin = np.clip(out_lin + noise, 0.0, 1.0)

    out = np.clip(out_lin, 0.0, 1.0) ** (1.0 / gamma)
    return np01_to_pil(out)


# ============================================================
# Visualization
# ============================================================

def save_triplet_visualization(texture: Image.Image,
                               scene: Image.Image,
                               captured: Image.Image,
                               out_path: Path):
    thumb_w = 220
    thumb_h = 220
    margin = 12
    label_h = 28
    total_w = thumb_w * 3 + margin * 4
    total_h = thumb_h + label_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    entries = [
        ("Projection image δ_d", texture),
        ("Physical scene n", scene),
        ("Synthesized captured I", captured),
    ]

    for idx, (label, img) in enumerate(entries):
        x = margin + idx * (thumb_w + margin)
        y = margin + label_h
        img = img.resize((thumb_w, thumb_h), Image.Resampling.BICUBIC)
        canvas.paste(img, (x, y))
        draw.text((x, margin), label, fill="black")

    canvas.save(out_path)


# ============================================================
# Dataset config / split
# ============================================================

def load_existing_config(root: Path):
    cfg_path = root / CONFIG_NAME
    if cfg_path.exists():
        return read_json(cfg_path)
    return None


def validate_or_write_config(root: Path, args, object_pool):
    current = {
        "source": args.source,
        "model": args.model,
        "size": args.size,
        "seed": args.seed,
        "scenes": args.scenes,
        "pairs_per_scene": args.pairs_per_scene,
        "target_mode": args.target_mode,
        "target_object": args.target_object,
        "object_pool": object_pool,
        "renderer": "triplet_renderer_mixed_objects_v1",
        "label": "synthetic_triplets_not_real_camera_capture",
    }

    cfg_path = root / CONFIG_NAME
    old = load_existing_config(root)

    if old is None:
        write_json(cfg_path, current)
        return

    if old != current:
        raise ValueError(
            "Output dataset already exists with different model/settings; "
            "choose a NEW --out directory"
        )


def scene_split_map(num_scenes: int, seed: int):
    if num_scenes < 3:
        raise ValueError("Need at least 3 scenes to create train/val/test split")

    idxs = list(range(num_scenes))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    train_n = min(num_scenes - 2, max(1, int(0.8 * num_scenes)))
    val_n = max(1, int(0.1 * num_scenes))

    if train_n + val_n >= num_scenes:
        val_n = 1

    test_n = num_scenes - train_n - val_n
    if test_n < 1:
        test_n = 1
        if train_n > 1:
            train_n -= 1
        else:
            val_n -= 1

    split = {}
    for i, idx in enumerate(idxs):
        if i < train_n:
            split[idx] = "train"
        elif i < train_n + val_n:
            split[idx] = "val"
        else:
            split[idx] = "test"
    return split


def choose_target_object(scene_idx: int, args, object_pool):
    if args.target_mode == "fixed":
        return args.target_object.strip()

    rng = random.Random(args.seed + scene_idx * 10007)
    return rng.choice(object_pool)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["modelscope", "procedural"], default="modelscope")
    parser.add_argument("--model", default="Qwen/Qwen-Image")
    parser.add_argument("--target-mode", choices=["fixed", "mixed"], default="mixed")
    parser.add_argument("--target-object", default="toy truck")
    parser.add_argument(
        "--object-pool",
        default="toy truck,toy car,mug,water bottle,backpack,book,shoe,potted plant,teddy bear,toolbox,helmet,speaker,desk lamp,small chair,small suitcase,keyboard"
    )
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--pairs-per-scene", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="data/mixed_triplets")
    parser.add_argument("--max-wait", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=10)
    args = parser.parse_args()

    if args.size % 8 != 0:
        raise ValueError("Image size must be divisible by 8 for latent diffusion training")

    if args.scenes < 3:
        raise ValueError("--scenes must be >= 3")

    if args.pairs_per_scene < 1:
        raise ValueError("--pairs-per-scene must be >= 1")

    object_pool = [x.strip() for x in args.object_pool.split(",") if x.strip()]
    if not object_pool:
        object_pool = DEFAULT_OBJECT_POOL[:]

    root = Path(args.out)
    ensure_dir(root)

    validate_or_write_config(root, args, object_pool)

    token = None
    if args.source == "modelscope":
        token = get_token()

    rng = random.Random(args.seed)
    split_map = scene_split_map(args.scenes, args.seed)

    rows = []
    scene_meta_summary = {}

    for scene_idx in range(args.scenes):
        scene_id = f"scene_{scene_idx:04d}"
        scene_dir = root / scene_id
        ensure_dir(scene_dir)

        target_object = choose_target_object(scene_idx, args, object_pool)
        prompt = build_scene_prompt(scene_idx, target_object)

        # ----------------------------------------------------
        # Generate / resume scene
        # ----------------------------------------------------
        scene, meta = generate_or_resume_scene(
            source=args.source,
            token=token,
            model=args.model,
            prompt=prompt,
            size=args.size,
            scene_dir=scene_dir,
            target_object=target_object,
            max_wait=args.max_wait,
            poll_interval=args.poll_interval,
            scene_idx=scene_idx,
            rng=rng,
        )

        scene_path = scene_dir / "scene.png"
        scene_meta_summary[scene_id] = {
            "scene_path": str(scene_path.relative_to(root)),
            "split": split_map[scene_idx],
            "target_object": target_object,
            "task_id": meta.get("task_id"),
            "status": meta.get("status"),
        }

        # ----------------------------------------------------
        # Multiple textures / captures for each scene
        # ----------------------------------------------------
        for pair_idx in range(args.pairs_per_scene):
            tex_path = scene_dir / f"texture_{pair_idx:04d}.png"
            cap_path = scene_dir / f"captured_{pair_idx:04d}.png"
            vis_path = scene_dir / f"triplet_vis_{pair_idx:04d}.png"

            if tex_path.exists() and cap_path.exists():
                print(f"Reusing sample: {scene_id}, pair {pair_idx}", flush=True)
            else:
                texture = make_projection_texture(args.size, rng)
                captured = synthesize_capture(scene, texture, target_object, rng)

                texture.save(tex_path)
                captured.save(cap_path)
                save_triplet_visualization(texture, scene, captured, vis_path)

            rows.append({
                "scene": str(scene_path.relative_to(root)),
                "texture": str(tex_path.relative_to(root)),
                "captured": str(cap_path.relative_to(root)),
                "split": split_map[scene_idx],
                "source": "modelscope_scene_plus_local_triplet_renderer" if args.source == "modelscope" else "procedural_scene_plus_local_triplet_renderer",
                "scene_id": scene_id,
                "session": scene_id,
                "target_object": target_object,
            })

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "texture", "captured", "split", "source", "scene_id", "session", "target_object"]
        )
        writer.writeheader()
        writer.writerows(rows)

    write_json(root / SCENE_META_NAME, scene_meta_summary)

    print("\nDone.", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Total triplets: {len(rows)}", flush=True)
    print(f"Output root: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()