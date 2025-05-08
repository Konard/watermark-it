#!/usr/bin/env python3
"""
Generate a solid‑colour test video (default 512 × 512, 10 s, pure white).

Example:
    python generate_video.py -o original.mp4
"""
import argparse
import cv2
import numpy as np


def generate_video(outfile: str,
                   width: int = 512,
                   height: int = 512,
                   duration: float = 10.0,
                   fps: int = 30,
                   colour: int = 255) -> None:
    frame_cnt = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    frame = np.full((height, width, 3), colour, dtype=np.uint8)
    for _ in range(frame_cnt):
        vw.write(frame)
    vw.release()
    print(f"Wrote {frame_cnt} frames to {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="plain.mp4")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--colour", type=int, default=255,
                    help="Base grey level 0–255")
    args = ap.parse_args()
    generate_video(args.output, args.width, args.height,
                   args.duration, args.fps, args.colour)


if __name__ == "__main__":
    main()
