#!/usr/bin/env python3
"""
End‑to‑end sanity check:
  1. generate_video.py → pristine.mp4
  2. watermark_video.py embeds random 64‑bit message
  3. decode_video_watermark.py extracts it back
"""
import os
import random
import subprocess
import sys
import tempfile


def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orig = os.path.join(tmp, "orig.mp4")
        water = os.path.join(tmp, "water.mp4")

        message = random.getrandbits(64)
        print(f"Watermark message: {message}")

        run(["python", "generate_video.py", "-o", orig])
        run(["python", "watermark_video.py", "-i", orig, "-o", water,
             "-m", str(message)])
        decoded = subprocess.check_output(
            ["python", "decode_video_watermark.py",
             "-o", orig, "-w", water]).decode().strip()
        print(f"Decoded message : {decoded}")
        if int(decoded) == message:
            print("\n✔ SUCCESS – message recovered correctly.")
            sys.exit(0)
        else:
            print("\n✖ FAILURE – mismatch!")
            sys.exit(1)


if __name__ == "__main__":
    main()
