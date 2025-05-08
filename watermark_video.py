#!/usr/bin/env python3
"""
Embed a 64‑bit message in a video by nudging 2×2 pixel blocks around the
frame perimeter.  Odd magnitude deltas → bit 0, even → bit 1.
The same watermark is applied to *every* frame.

Example:
    python watermark_video.py -i original.mp4 -o watermarked.mp4 -m 123456789
"""
import argparse
import random
from typing import List, Tuple

import cv2
import numpy as np


def int_to_bits(value: int, bit_count: int = 64) -> List[int]:
    return [(value >> (bit_count - 1 - i)) & 1 for i in range(bit_count)]


def compute_positions(w: int, h: int, block: int, pad: int) -> List[Tuple[int, int]]:
    """
    Return 64 (x,y) positions starting top‑left, clockwise, 16 per edge.
    All positions are top‑left corners of the block.
    """
    usable_w = w - 2 * pad - block
    usable_h = h - 2 * pad - block
    step_x = usable_w / 15
    step_y = usable_h / 15

    pos = []
    # top edge
    for i in range(16):
        x = int(round(pad + i * step_x))
        y = pad
        pos.append((x, y))
    # right edge
    for i in range(16):
        x = w - pad - block
        y = int(round(pad + i * step_y))
        pos.append((x, y))
    # bottom edge
    for i in range(16):
        x = int(round(w - pad - block - i * step_x))
        y = h - pad - block
        pos.append((x, y))
    # left edge
    for i in range(16):
        x = pad
        y = int(round(h - pad - block - i * step_y))
        pos.append((x, y))
    assert len(pos) == 64
    return pos


def embed(infile: str,
          outfile: str,
          message: int,
          block: int = 2,
          pad: int = 42,
          delta_min: int = 1,
          delta_max: int = 3,
          seed: int = 0) -> None:
    rng = random.Random(seed)
    bits = int_to_bits(message)
    cap = cv2.VideoCapture(infile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(outfile, fourcc, fps, (w, h))

    positions = compute_positions(w, h, block, pad)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for (x, y), bit in zip(positions, bits):
            mag = rng.randint(delta_min, delta_max)  # pick 1–3
            if mag % 2 != bit ^ 1:                   # force odd→0, even→1
                mag += 1
            if np.any(frame[y:y+block, x:x+block] + mag > 255):
                mag = -mag                          # flip sign if overflow
            frame[y:y+block, x:x+block] = np.clip(
                frame[y:y+block, x:x+block].astype(int) + mag, 0, 255
            ).astype(np.uint8)
        vw.write(frame)
        frame_idx += 1

    cap.release()
    vw.release()
    print(f"Watermarked {frame_idx} frames → {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--message", required=True,
                    help="64‑bit integer (decimal or 0x‑prefixed hex)")
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--pad", type=int, default=42)
    ap.add_argument("--delta-min", type=int, default=1)
    ap.add_argument("--delta-max", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed so embed/detect use same deltas")
    args = ap.parse_args()

    msg_int = int(args.message, 0) & ((1 << 64) - 1)
    embed(args.input, args.output, msg_int,
          args.block, args.pad, args.delta_min, args.delta_max, args.seed)


if __name__ == "__main__":
    main()
