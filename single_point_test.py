#!/usr/bin/env python3
"""
single_point_test.py
--------------------
Minimal end‑to‑end test:
* 64×64 white video, 60 frames @ 30 fps
* Embed one bit in a 2×2 block at (32,32)
* Verify the same bit is decoded
Videos are written to ./test_videos/orig.mp4 and ./test_videos/water.mp4
"""
import os
import random
import cv2
import numpy as np
from pathlib import Path

# ───────────────────────────────────────── config ─────────────────────────────────────────
W, H = 64, 64           # frame size
FPS = 30
FRAMES = 60             # 2 seconds
BLOCK = 2               # watermark block size
X, Y = 32, 32           # top‑left corner of 2×2 block
DELTA0 = 15             # bit 0 → −15 (odd)
DELTA1 = 60             # bit 1 → −60 (even)
THRESH = 30             # magnitude > THRESH  ⇒ bit 1
OUT_DIR = Path("test_videos")
OUT_DIR.mkdir(exist_ok=True)

orig_path = OUT_DIR / "orig.mp4"
water_path = OUT_DIR / "water.mp4"

# ───────────────────────────────────── generate original ──────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
orig_writer = cv2.VideoWriter(str(orig_path), fourcc, FPS, (W, H))
white = np.full((H, W, 3), 255, dtype=np.uint8)
for _ in range(FRAMES):
    orig_writer.write(white)
orig_writer.release()
print(f"Wrote reference video to {orig_path}")

# ───────────────────────────────────── embed watermark ────────────────────────────────────
bit = random.randint(0, 1)
delta = -(DELTA0 if bit == 0 else DELTA1)
wm_writer = cv2.VideoWriter(str(water_path), fourcc, FPS, (W, H))
for _ in range(FRAMES):
    frame = white.copy()
    roi = frame[Y:Y+BLOCK, X:X+BLOCK].astype(int) + delta
    frame[Y:Y+BLOCK, X:X+BLOCK] = np.clip(roi, 0, 255).astype(np.uint8)
    wm_writer.write(frame)
wm_writer.release()
print(f"Embedded bit {bit} (delta={delta}) → {water_path}")

# ───────────────────────────────────── decode watermark ───────────────────────────────────
cap_o = cv2.VideoCapture(str(orig_path))
cap_w = cv2.VideoCapture(str(water_path))
ret_o, f0 = cap_o.read()
ret_w, f1 = cap_w.read()
cap_o.release(), cap_w.release()
assert ret_o and ret_w, "Could not read frames for decoding"

diff_block = f1.astype(int)[Y:Y+BLOCK, X:X+BLOCK] - f0.astype(int)[Y:Y+BLOCK, X:X+BLOCK]
mag = np.abs(diff_block).mean()
decoded_bit = 1 if mag > THRESH else 0

# ────────────────────────────────────────── result ───────────────────────────────────────
print(f"Measured |ΔY| ≈ {mag:.1f}  → decoded bit {decoded_bit}")
if decoded_bit == bit:
    print("✔ SUCCESS – bit recovered correctly.")
else:
    print("✖ FAILURE – mismatch!")