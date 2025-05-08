#!/usr/bin/env python3
"""
Embed a 64‑bit message in a video by nudging 2×2 pixel blocks around the
frame perimeter.  Magnitude 3 → bit 0, magnitude 4 → bit 1.
The same watermark is applied to *every* frame.

Example:
    python watermark_video.py -i original.mp4 -o watermarked.mp4 -m 123456789
"""
import argparse
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
    for i in range(16):
        pos.append((int(round(pad + i * step_x)), pad))                     # top
    for i in range(16):
        pos.append((w - pad - block, int(round(pad + i * step_y))))         # right
    for i in range(16):
        pos.append((int(round(w - pad - block - i * step_x)), h - pad - block))  # bottom
    for i in range(16):
        pos.append((pad, int(round(h - pad - block - i * step_y))))         # left
    return pos


def embed(infile: str,
          outfile: str,
          message: int,
          block: int = 2,
          pad: int = 42,
          delta0: int = 3,
          delta1: int = 4) -> None:
    """
    delta0 is the absolute luminance change for bit 0 (must be odd),
    delta1 for bit 1 (must be even).  Defaults 3 / 4 – small but survives
    H.264 compression.
    """
    assert delta0 % 2 == 1, "delta0 must be odd"
    assert delta1 % 2 == 0, "delta1 must be even"

    bits = int_to_bits(message)
    cap = cv2.VideoCapture(infile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(outfile, fourcc, fps, (w, h))

    positions = compute_positions(w, h, block, pad)

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for (x, y), bit in zip(positions, bits):
            mag = delta0 if bit == 0 else delta1      # 3 → 0, 4 → 1
            delta = -mag                              # always darken = safe for white
            roi = frame[y:y + block, x:x + block].astype(int) + delta
            frame[y:y + block, x:x + block] = np.clip(roi, 0, 255).astype(np.uint8)

        vw.write(frame)
        written += 1

    cap.release()
    vw.release()
    print(f"Watermarked {written} frames → {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--message", required=True,
                    help="64‑bit integer (decimal or 0x‑prefixed hex)")
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--pad", type=int, default=42)
    ap.add_argument("--delta0", type=int, default=3,
                    help="Odd  luminance step for bit 0 (default 3)")
    ap.add_argument("--delta1", type=int, default=4,
                    help="Even luminance step for bit 1 (default 4)")
    args = ap.parse_args()

    msg_int = int(args.message, 0) & ((1 << 64) - 1)
    embed(args.input, args.output, msg_int,
          args.block, args.pad, args.delta0, args.delta1)


if __name__ == "__main__":
    main()