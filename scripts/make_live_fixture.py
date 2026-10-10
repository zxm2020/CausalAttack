"""Create a short deterministic camera-video fixture plus reference and ROI mask."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'live_fixture'
OUT.mkdir(parents=True, exist_ok=True)
size = 256
reference = np.full((size, size, 3), (124, 104, 80), dtype=np.uint8)
cv2.rectangle(reference, (0, 184), (255, 225), (72, 74, 62), -1)
mask = np.zeros((size, size), dtype=np.uint8)
cv2.rectangle(mask, (92, 52), (168, 184), 255, -1)
Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)).save(OUT / 'reference.png')
Image.fromarray(mask).save(OUT / 'mask.png')
writer = cv2.VideoWriter(str(OUT / 'camera_fixture.avi'),
                         cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (size, size))
if not writer.isOpened():
    raise RuntimeError('Could not create live camera fixture')
for index in range(90):
    frame = reference.copy()
    shift = int(8 * np.sin(index / 12))
    cv2.rectangle(frame, (100 + shift, 72), (160 + shift, 184), (38, 72, 190), -1)
    cv2.circle(frame, (130 + shift, 55), 22, (116, 154, 210), -1)
    writer.write(frame)
writer.release()
print(OUT)
