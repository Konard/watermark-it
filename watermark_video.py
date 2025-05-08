#!/usr/bin/env python3
"""
Embed a 64‑bit message in a video by nudging small blocks around the
frame perimeter.

*  odd  ∆Y  → bit 0   (default ‑15)
*  even ∆Y  → bit 1   (default ‑30)

A 15 → 30 step survives x264’s heavy quantisation yet is still
visually invisible on a white frame.

Example
-------
python watermark_video.py -i original.mp4 -o water.mp4 -m 123456789
"""
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def int_to_bits(value: int, bit_count: int = 64) -> List[int]:
    return [(value >> (bit_count - 1 - i)) & 1 for i in range(bit_count)]


def compute_positions(w: int, h: int, block: int, pad: int) -> List[Tuple[int, int]]:
    usable_w, usable_h = w - 2 * pad - block, h - 2 * pad - block
    step_x, step_y = usable_w / 15, usable_h / 15
    pos = []
    pos += [(int(round(pad + i * step_x)), pad) for i in range(16)]
    pos += [(w - pad - block, int(round(pad + i * step_y))) for i in range(16)]
    pos += [(int(round(w - pad - block - i * step_x)), h - pad - block) for i in range(16)]
    pos += [(pad, int(round(h - pad - block - i * step_y))) for i in range(16)]
    return pos


def embed(infile: str,
          outfile: str,
          message: int,
          block: int = 2,
          pad: int = 42,
          delta0: int = 15,   # odd  → 0
          delta1: int = 30) -> None:  # even → 1
    assert delta0 % 2 == 1 and delta1 % 2 == 0, "delta0 must be odd, delta1 even"

    bits = int_to_bits(message)
    cap = cv2.VideoCapture(infile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(outfile, fourcc, fps, (w, h))
    pos = compute_positions(w, h, block, pad)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for (x, y), b in zip(pos, bits):
            delta = -(delta0 if b == 0 else delta1)  # always darken
            roi = frame[y:y + block, x:x + block].astype(int) + delta
            frame[y:y + block, x:x + block] = np.clip(roi, 0, 255).astype(np.uint8)
        vw.write(frame)
        frames += 1

    cap.release()
    vw.release()
    print(f"Watermarked {frames} frames → {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--message", required=True,
                    help="64‑bit integer (decimal or 0x‑prefixed hex)")
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--pad", type=int, default=42)
    ap.add_argument("--delta0", type=int, default=15)
    ap.add_argument("--delta1", type=int, default=30)
    args = ap.parse_args()

    msg_int = int(args.message, 0) & ((1 << 64) - 1)
    embed(args.input, args.output, msg_int,
          args.block, args.pad, args.delta0, args.delta1)


if __name__ == "__main__":
    main()