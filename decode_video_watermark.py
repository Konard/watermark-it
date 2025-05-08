#!/usr/bin/env python3
"""
Recover the 64‑bit watermark by comparing a watermarked video with its
pristine original.

The algorithm is codec‑agnostic:

1.  Measure |ΔY| at each of the 64 blocks across *all* frames.
2.  Use a simple two‑cluster split: threshold = (min + max) / 2.
3.  Smaller cluster → bit 0, larger → bit 1.

Debug lines go to **stderr**, so normal callers (like test.py) still get
only the integer on stdout.

Example
-------
python decode_video_watermark.py -o orig.mp4 -w water.mp4 --debug
"""
import argparse
import sys
from typing import List, Tuple

import cv2
import numpy as np


def positions(w: int, h: int, block: int, pad: int) -> List[Tuple[int, int]]:
    ux, uy = (w - 2 * pad - block) / 15, (h - 2 * pad - block) / 15
    pts = []
    pts += [(int(round(pad + i * ux)), pad) for i in range(16)]
    pts += [(w - pad - block, int(round(pad + i * uy))) for i in range(16)]
    pts += [(int(round(w - pad - block - i * ux)), h - pad - block) for i in range(16)]
    pts += [(pad, int(round(h - pad - block - i * uy))) for i in range(16)]
    return pts


def bits_to_int(bits: List[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def decode(orig: str, water: str, block: int, pad: int, debug: bool) -> int:
    co, cw = cv2.VideoCapture(orig), cv2.VideoCapture(water)
    w, h = (int(co.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(co.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pts = positions(w, h, block, pad)

    sums, frames = np.zeros(64), 0
    while True:
        ok0, f0 = co.read()
        ok1, f1 = cw.read()
        if not (ok0 and ok1):
            break
        diff = f1.astype(np.int16) - f0.astype(np.int16)
        for i, (x, y) in enumerate(pts):
            sums[i] += np.abs(diff[y:y + block, x:x + block]).mean()
        frames += 1
    co.release(), cw.release()

    means = sums / max(1, frames)
    t = (means.min() + means.max()) / 2.0  # midpoint threshold

    bits = []
    for i, m in enumerate(means):
        bit = 0 if m < t else 1
        bits.append(bit)
        if debug:
            print(f"{i:02d}: |Δ|={m:.3f}  threshold={t:.3f} → bit {bit}",
                  file=sys.stderr)

    return bits_to_int(bits)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--original", required=True)
    p.add_argument("-w", "--watermarked", required=True)
    p.add_argument("--block", type=int, default=2)
    p.add_argument("--pad", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()

    print(decode(a.original, a.watermarked, a.block, a.pad, a.debug))


if __name__ == "__main__":
    main()