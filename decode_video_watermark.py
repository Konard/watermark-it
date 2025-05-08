#!/usr/bin/env python3
"""
Recover the 64‑bit watermark by diffing a watermarked video against
its pristine original.

Example:
    python decode_video_watermark.py -o original.mp4 -w watermarked.mp4
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
        pos.append((int(round(pad + i * step_x)), pad))
    for i in range(16):
        pos.append((w - pad - block, int(round(pad + i * step_y))))
    for i in range(16):
        pos.append((int(round(w - pad - block - i * step_x)), h - pad - block))
    for i in range(16):
        pos.append((pad, int(round(h - pad - block - i * step_y))))
    return pos


def bits_to_int(bits: List[int]) -> int:
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def decode(orig: str, water: str, block: int, pad: int) -> int:
    co = cv2.VideoCapture(orig)
    cw = cv2.VideoCapture(water)
    w = int(co.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(co.get(cv2.CAP_PROP_FRAME_HEIGHT))
    positions = compute_positions(w, h, block, pad)
    sums = np.zeros(len(positions), dtype=np.float64)
    frames = 0

    while True:
        ok_o, fo = co.read()
        ok_w, fw = cw.read()
        if not (ok_o and ok_w):
            break

        diff = fw.astype(np.int16) - fo.astype(np.int16)
        for idx, (x, y) in enumerate(positions):
            roi = diff[y:y+block, x:x+block]
            sums[idx] += roi.mean()
        frames += 1

    co.release()
    cw.release()

    bits = []
    for mean_delta in sums / max(1, frames):
        mag = abs(int(round(mean_delta)))
        bits.append(0 if mag % 2 else 1)
    return bits_to_int(bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--original", required=True)
    ap.add_argument("-w", "--watermarked", required=True)
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--pad", type=int, default=42)
    args = ap.parse_args()
    msg = decode(args.original, args.watermarked, args.block, args.pad)
    print(msg)


if __name__ == "__main__":
    main()
