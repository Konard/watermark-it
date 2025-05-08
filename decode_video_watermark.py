#!/usr/bin/env python3
"""
Recover the 64‑bit watermark by diffing a watermarked video against
its pristine original.

Strategy
--------
For each of the 64 blocks we compute the mean absolute luminance delta
over every frame, then decide “bit 0 or bit 1” by seeing whether that
delta is closer to `delta0` (odd, e.g. 3) or `delta1` (even, e.g. 4).

Example
-------
    python decode_video_watermark.py \
        -o original.mp4 -w watermarked.mp4 --delta0 3 --delta1 4
"""
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def compute_positions(w: int, h: int, block: int, pad: int) -> List[Tuple[int, int]]:
    usable_w = w - 2 * pad - block
    usable_h = h - 2 * pad - block
    step_x = usable_w / 15
    step_y = usable_h / 15
    pos = []
    for i in range(16):
        pos.append((int(round(pad + i * step_x)), pad))                         # top
    for i in range(16):
        pos.append((w - pad - block, int(round(pad + i * step_y))))             # right
    for i in range(16):
        pos.append((int(round(w - pad - block - i * step_x)), h - pad - block)) # bottom
    for i in range(16):
        pos.append((pad, int(round(h - pad - block - i * step_y))))             # left
    return pos


def bits_to_int(bits: List[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def decode(orig: str,
           water: str,
           block: int,
           pad: int,
           delta0: int,
           delta1: int,
           debug: bool = False) -> int:
    co = cv2.VideoCapture(orig)
    cw = cv2.VideoCapture(water)
    w = int(co.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(co.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pos = compute_positions(w, h, block, pad)

    sums = np.zeros(len(pos), dtype=np.float64)
    frames = 0
    while True:
        ok_o, f0 = co.read()
        ok_w, f1 = cw.read()
        if not (ok_o and ok_w):
            break
        diff = f1.astype(np.int16) - f0.astype(np.int16)
        for idx, (x, y) in enumerate(pos):
            roi = diff[y:y + block, x:x + block]
            sums[idx] += np.abs(roi).mean()
        frames += 1

    co.release()
    cw.release()

    abs_means = sums / max(1, frames)
    bits = []
    for idx, m in enumerate(abs_means):
        bit = 0 if abs(m - delta0) < abs(m - delta1) else 1
        bits.append(bit)
        if debug:
            print(f"{idx:02d}: meanΔ={m:.2f}  → bit {bit}")

    return bits_to_int(bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--original", required=True)
    ap.add_argument("-w", "--watermarked", required=True)
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--pad", type=int, default=42)
    ap.add_argument("--delta0", type=int, default=3,
                    help="Expected magnitude for bit 0 (must match embed)")
    ap.add_argument("--delta1", type=int, default=4,
                    help="Expected magnitude for bit 1 (must match embed)")
    ap.add_argument("--debug", action="store_true",
                    help="Print per‑spot diagnostics")
    args = ap.parse_args()

    msg = decode(args.original, args.watermarked,
                 args.block, args.pad, args.delta0, args.delta1, args.debug)
    print(msg)


if __name__ == "__main__":
    main()