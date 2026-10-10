"""Live camera capture with asynchronous CausalAttack texture updates.

The camera and projector windows keep refreshing while a worker optimizes the next
continuous texture.  Only completed updates are clipped and quantized for display.
No physical geometry calibration is implied by the default identity canvas mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from causalattack.backends import (FixedSmokeFeatures, ProjectNetOnline,
                                   SyntheticPC, YOLOv5Features)
from causalattack.core import AttackSettings, display_command, optimize
from causalattack.paths import (PROJECTNET_CHECKPOINT, PROJECTNET_ROOT, VAE_ROOT,
                                YOLOV5_ROOT, YOLOV5_WEIGHTS)

HERE = Path(__file__).resolve().parent


def frame_tensor(frame_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32).div_(255)


def load_rgb(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        values = np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS),
                            dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device)


def load_mask(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        values = np.asarray(image.convert('L').resize((size, size), Image.Resampling.NEAREST),
                            dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)


def roi_mask(roi: tuple[int, int, int, int], frame_shape: tuple[int, ...], size: int,
             device: torch.device) -> torch.Tensor:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError('ROI width and height must be positive')
    fh, fw = frame_shape[:2]
    x0, x1 = max(0, x), min(fw, x + w)
    y0, y1 = max(0, y), min(fh, y + h)
    raw = np.zeros((fh, fw), dtype=np.uint8)
    raw[y0:y1, x0:x1] = 255
    resized = cv2.resize(raw, (size, size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized.astype(np.float32) / 255).unsqueeze(0).unsqueeze(0).to(device)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(piece.strip()) for piece in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('ROI must be x,y,w,h') from exc
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError('ROI must be x,y,w,h with positive width/height')
    return parts


def parse_source(value: str):
    return int(value) if value.isdecimal() else value


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.reference: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.continuous: torch.Tensor | None = None
        self.display_rgb: np.ndarray | None = None
        self.running = False
        self.optimizing = False
        self.update_id = 0
        self.last_metrics: dict | None = None
        self.last_error: str | None = None


def rendered_texture(state: SharedState, mode: str, gain: float, size: int) -> np.ndarray:
    """Return the selected live visualization without changing the saved true command."""
    with state.lock:
        command = state.display_rgb.copy()
        continuous = None if state.continuous is None else state.continuous.detach().cpu().clone()
        has_reference = state.reference is not None
        has_mask = state.mask is not None
        running = state.running
    if continuous is None:
        panel = np.full((size, size, 3), 32, dtype=np.uint8)
        lines = ['NO COMPLETED TEXTURE',
                 f"reference: {'OK' if has_reference else 'press R'}",
                 f"target ROI: {'OK' if has_mask else 'press I'}",
                 f"optimizer: {'RUNNING' if running else 'press S'}"]
        for row, line in enumerate(lines):
            cv2.putText(panel, line, (10, 35 + row * 34), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (230, 230, 230), 1, cv2.LINE_AA)
        return panel
    if mode == 'command':
        return command
    if mode == 'amplified':
        return np.rint(np.clip(command.astype(np.float32) * gain, 0, 255)).astype(np.uint8)
    values = continuous[0].permute(1, 2, 0).numpy()
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return np.rint((values - low) / (high - low) * 255).astype(np.uint8)


def save_completed_update(output: Path, update_id: int, scene: torch.Tensor,
                          texture: torch.Tensor, command: np.ndarray,
                          history: list[dict], metrics: dict) -> None:
    folder = output / f'update_{update_id:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    scene_rgb = np.rint(scene[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    texture_rgb = texture[0].detach().cpu().permute(1, 2, 0).numpy()
    low, high = float(texture_rgb.min()), float(texture_rgb.max())
    preview = np.zeros_like(texture_rgb) if high <= low else (texture_rgb - low) / (high - low)
    Image.fromarray(scene_rgb, 'RGB').save(folder / 'scene.png')
    Image.fromarray(command, 'RGB').save(folder / 'display_texture.png')
    Image.fromarray(np.rint(preview * 255).astype(np.uint8), 'RGB').save(
        folder / 'optimized_texture_preview.png')
    torch.save(texture.detach().cpu(), folder / 'continuous_texture.pt')
    with (folder / 'loss_curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def optimization_worker(state: SharedState, stop: threading.Event, projector, features,
                        settings: AttackSettings, size: int, device: torch.device,
                        output: Path) -> None:
    while not stop.is_set():
        with state.lock:
            ready = state.running and state.latest_frame is not None and \
                state.reference is not None and state.mask is not None
            if ready:
                frame = state.latest_frame.copy()
                reference = state.reference.detach().clone()
                mask = state.mask.detach().clone()
                initial = None if state.continuous is None else state.continuous.detach().clone()
                update_id = state.update_id + 1
                state.optimizing = True
                state.last_error = None
        if not ready:
            stop.wait(0.03)
            continue
        try:
            scene = frame_tensor(frame, size, device)
            if isinstance(projector, ProjectNetOnline):
                projector.prepare_scene(scene, seed=settings.seed + update_id)
            texture, history, measurements = optimize(
                scene, reference, projector, features, settings, initial=initial, mask=mask)
            command_tensor = display_command(texture)
            command = command_tensor[0].permute(1, 2, 0).cpu().numpy()
            metrics = dict(update_id=update_id, completed_at=time.time(),
                           display_mapping='identity_uncalibrated', **measurements)
            save_completed_update(output, update_id, scene, texture, command, history, metrics)
            with state.lock:
                state.continuous = texture
                state.display_rgb = command
                state.update_id = update_id
                state.last_metrics = metrics
                state.optimizing = False
        except Exception as exc:
            if isinstance(exc, RuntimeError) and 'gradient to continuous texture' in str(exc):
                # A stationary candidate is not a display failure. Keep the last completed
                # command visible and wait for a newer camera observation.
                with state.lock:
                    state.last_metrics = dict(update_id=state.update_id,
                                              status='stationary_texture_waiting_for_new_scene')
                    state.optimizing = False
                stop.wait(0.25)
                continue
            with state.lock:
                state.last_error = f'{type(exc).__name__}: {exc}'
                state.optimizing = False
                state.running = False


def open_capture(source):
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(str(source))
    return capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='camera index or video path')
    parser.add_argument('--backend', choices=['projectnet', 'smoke'], default='projectnet')
    parser.add_argument('--features', choices=['yolo', 'smoke'], default='yolo')
    parser.add_argument('--reference', type=Path, help='scene-matched target-free reference')
    parser.add_argument('--mask', type=Path, help='target-local grayscale mask')
    parser.add_argument('--roi', type=parse_roi, help='initial camera ROI as x,y,w,h')
    parser.add_argument('--auto-start', action='store_true')
    parser.add_argument('--headless', action='store_true', help='no windows; intended for automated tests')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--record', type=Path, help='optional MP4 recording of displayed commands')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--display-x', type=int)
    parser.add_argument('--display-y', type=int)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--display-mode', choices=['command', 'amplified', 'normalized'],
                        default='normalized',
                        help='live projector-window rendering; true command is always saved separately')
    parser.add_argument('--display-gain', type=float, default=16.0,
                        help='gain used only by amplified display mode')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'live')
    parser.add_argument('--projectnet-checkpoint', type=Path, default=
                        PROJECTNET_CHECKPOINT)
    parser.add_argument('--vae', type=Path, default=VAE_ROOT)
    parser.add_argument('--yolo-weights', type=Path,
                        default=YOLOV5_WEIGHTS)
    args = parser.parse_args()
    if args.steps < 1 or args.samples < 1 or args.fps <= 0 or args.display_gain <= 0:
        parser.error('steps, samples, fps, and display gain must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    size = 256 if args.backend == 'projectnet' or args.features == 'yolo' else 64
    args.output.mkdir(parents=True, exist_ok=True)

    print('Loading attack backends...')
    projector = (ProjectNetOnline(PROJECTNET_ROOT,
                                  args.projectnet_checkpoint, device, vae_source=args.vae)
                 if args.backend == 'projectnet' else SyntheticPC())
    features = (YOLOv5Features(YOLOV5_ROOT, args.yolo_weights, device, size=size)
                if args.features == 'yolo' else FixedSmokeFeatures().to(device).eval())
    learning_rate = 0.01 if args.backend == 'projectnet' else 12.0
    alpha = 1.0 if args.backend == 'projectnet' else 1.0 / (size * size)
    settings = AttackSettings(outer_steps=args.steps, samples=args.samples,
                              learning_rate=learning_rate, alpha=alpha)

    capture = open_capture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open camera/video source: {args.source}')
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f'Camera/video source produced no frame: {args.source}')

    state = SharedState()
    state.latest_frame = first_frame.copy()
    if args.reference:
        state.reference = load_rgb(args.reference, size, device)
    if args.mask:
        state.mask = load_mask(args.mask, size, device)
    elif args.roi:
        state.mask = roi_mask(args.roi, first_frame.shape, size, device)
    state.running = bool(args.auto_start and state.reference is not None and state.mask is not None)
    state.display_rgb = np.zeros((size, size, 3), dtype=np.uint8)

    stop = threading.Event()
    worker = threading.Thread(target=optimization_worker,
                              args=(state, stop, projector, features, settings, size,
                                    device, args.output), daemon=True)
    worker.start()

    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (size, size))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create video recording: {args.record}')

    if not args.headless:
        cv2.namedWindow('CausalAttack camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('CausalAttack projector output', cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty('CausalAttack projector output', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        if args.display_x is not None and args.display_y is not None:
            cv2.moveWindow('CausalAttack projector output', args.display_x, args.display_y)
        print('Keys: r=capture target-free reference, i=select ROI, s=start/pause, '
              'v=cycle display mode, c=clear texture, q/Esc=quit')
        print(f'Live display mode: {args.display_mode}. The exact clipped/quantized command '
              'is saved in each update folder.')

    frame_count = 0
    frame_period = 1.0 / args.fps
    try:
        frame = first_frame
        while True:
            tick = time.perf_counter()
            if frame_count:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
            with state.lock:
                state.latest_frame = frame.copy()
                running, optimizing = state.running, state.optimizing
                update_id, error = state.update_id, state.last_error
                has_reference, has_mask = state.reference is not None, state.mask is not None
            shown_rgb = rendered_texture(state, args.display_mode, args.display_gain, size)
            command_bgr = cv2.cvtColor(shown_rgb, cv2.COLOR_RGB2BGR)
            if writer:
                writer.write(command_bgr)
            if not args.headless:
                overlay = frame.copy()
                status = (f"{'RUN' if running else 'PAUSE'} | "
                          f"{'optimizing' if optimizing else 'idle'} | update {update_id}")
                cv2.putText(overlay, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (0, 255, 0) if running else (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, f'reference={has_reference} mask={has_mask}', (12, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                if error:
                    cv2.putText(overlay, error[:90], (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('CausalAttack camera', overlay)
                cv2.imshow('CausalAttack projector output', command_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord('r'):
                    with state.lock:
                        state.reference = frame_tensor(frame, size, device)
                    print('Captured target-free reference from the current camera frame.')
                elif key == ord('i'):
                    selected = tuple(int(v) for v in cv2.selectROI(
                        'CausalAttack camera', frame, showCrosshair=True, fromCenter=False))
                    if selected[2] > 0 and selected[3] > 0:
                        with state.lock:
                            state.mask = roi_mask(selected, frame.shape, size, device)
                        print('Updated target-local ROI:', selected)
                elif key == ord('s'):
                    with state.lock:
                        if state.reference is None or state.mask is None:
                            print('Capture/load a reference and select/load an ROI mask first.')
                        else:
                            state.running = not state.running
                            print('Optimization', 'started' if state.running else 'paused')
                elif key == ord('v'):
                    modes = ['normalized', 'amplified', 'command']
                    args.display_mode = modes[(modes.index(args.display_mode) + 1) % len(modes)]
                    print('Live display mode:', args.display_mode)
                elif key == ord('c'):
                    with state.lock:
                        state.continuous = None
                        state.display_rgb = np.zeros((size, size, 3), dtype=np.uint8)
                    print('Cleared the continuous texture and display command.')
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
            delay = frame_period - (time.perf_counter() - tick)
            if delay > 0:
                time.sleep(delay)
    finally:
        stop.set()
        capture.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        worker.join(timeout=2.0)
        summary = dict(frames=frame_count, completed_updates=state.update_id,
                       stopped_during_update=state.optimizing, last_error=state.last_error,
                       backend=args.backend, features=args.features, source=str(args.source),
                       display_mapping='identity_uncalibrated',
                       live_display_mode=args.display_mode,
                       true_command_saved_per_completed_update=True)
        (args.output / 'live_session.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
