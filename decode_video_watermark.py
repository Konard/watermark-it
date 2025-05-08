#!/usr/bin/env python3
"""
Decode the 64‑bit watermark by clustering each block’s |ΔY| into two
groups (low & high) via a lightweight k‑means‑2.

Advantages
----------
*  Automatically adapts to any codec scaling or per‑block drift.
*  No hand‑tuned deltas or thresholds.
*  Debug lines go to stderr so test.py sees only the final integer.
"""
import argparse
import sys
from typing import List, Tuple

import cv2
import numpy as np


# ───────────────────────── geometry helpers ────────────────────────────
def positions(w: int, h: int, block: int, pad: int) -> List[Tuple[int, int]]:
    ux, uy = (w - 2 * pad - block) / 15, (h - 2 * pad - block) / 15
    pts = []
    pts += [(int(round(pad + i * ux)), pad) for i in range(16)]                      # top
    pts += [(w - pad - block, int(round(pad + i * uy))) for i in range(16)]          # right
    pts += [(int(round(w - pad - block - i * ux)), h - pad - block) for i in range(16)]  # bottom
    pts += [(pad, int(round(h - pad - block - i * uy))) for i in range(16)]          # left
    return pts


def bits_to_int(bits: List[int]) -> int:
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


# ───────────────────────── simple k‑means‑2 ────────────────────────────
def kmeans2(values: np.ndarray, iters: int = 20) -> Tuple[float, float, np.ndarray]:
    """
    Very small k=2 1‑D k‑means.  Returns (c_low, c_high, labels[0..n-1])
    where c_low < c_high.
    """
    c_low, c_high = values.min(), values.max()
    labels = np.zeros_like(values, dtype=np.uint8)

    for _ in range(iters):
        labels = (np.abs(values - c_high) < np.abs(values - c_low)).astype(np.uint8)
        if labels.all() or (~labels.astype(bool)).all():
            break  # degenerate; all points one side
        c_low = values[labels == 0].mean()
        c_high = values[labels == 1].mean()

    if c_low > c_high:         # ensure ordering
        c_low, c_high = c_high, c_low
        labels ^= 1
    return c_low, c_high, labels


# ───────────────────────── main decode routine ─────────────────────────
def decode(orig: str, water: str, block: int, pad: int, debug: bool) -> int:
    cap0, cap1 = cv2.VideoCapture(orig), cv2.VideoCapture(water)
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pts = positions(w, h, block, pad)

    sums, frames = np.zeros(64), 0
    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            break
        diff = f1.astype(np.int16) - f0.astype(np.int16)
        for i, (x, y) in enumerate(pts):
            sums[i] += np.abs(diff[y:y + block, x:x + block]).mean()
        frames += 1

    cap0.release(), cap1.release()
    means = sums / max(frames, 1)

    c_low, c_high, labels = kmeans2(means)
    # labels: 0 -> low cluster (bit 0), 1 -> high cluster (bit 1)

    if debug:
        for i, (m, b) in enumerate(zip(means, labels)):
            print(f"{i:02d}: |Δ|={m:.3f}  → bit {b}", file=sys.stderr)
        print(f"Cluster centroids: low≈{c_low:.2f}, high≈{c_high:.2f}",
              file=sys.stderr)

    return bits_to_int(labels.tolist())


# ───────────────────────── argument glue ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--original", required=True)
    ap.add_argument("-w", "--watermarked", required=True)
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--pad", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(decode(args.original, args.watermarked,
                 args.block, args.pad, args.debug))


if __name__ == "__main__":
    main()