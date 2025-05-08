#!/usr/bin/env python3
"""
End‑to‑end sanity check:
  1. generate_video.py → pristine.mp4
  2. watermark_video.py embeds random 64‑bit message
  3. decode_video_watermark.py extracts it back

Uses the current Python interpreter (sys.executable) instead of the
string 'python' so it works on systems where only `python3` exists.
"""
import os
import random
import subprocess
import sys


def run(cmd):
    """Run a subprocess, echoing the command line."""
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    local_folder = "./test_videos"
    os.makedirs(local_folder, exist_ok=True)

    orig = os.path.join(local_folder, "orig.mp4")
    water = os.path.join(local_folder, "water.mp4")

    message = random.getrandbits(64)
    print(f"Watermark message: {message}")

    exe = sys.executable  # path to the current Python interpreter

    run([exe, "generate_video.py", "-o", orig])
    run([
        exe,
        "watermark_video.py",
        "-i", orig,
        "-o", water,
        "-m", str(message),
    ])
    decoded = subprocess.check_output([
        exe,
        "decode_video_watermark.py",
        "-o", orig,
        "-w", water,
        "--debug"
    ]).decode().strip()

    print(f"Decoded message : {decoded}")
    if int(decoded) == message:
        print("\n✔ SUCCESS – message recovered correctly.")
        sys.exit(0)
    else:
        print("\n✖ FAILURE – mismatch!")
        sys.exit(1)


if __name__ == "__main__":
    main()