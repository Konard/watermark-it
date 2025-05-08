#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# init_watermark_suite.sh
# Bootstrap the watermark demo into an empty repo
# -----------------------------------------------------------------------------
set -euo pipefail

# --- helper ---------------------------------------------------------------
add_file () {
  local name="$1"; shift
  printf "  • %s\n" "$name"
  cat > "$name" <<'PY'
$@
PY
}

echo "🔧 Creating watermark demo files …"

# ---------------------------------------------------------------- generate_video.py
cat > generate_video.py <<'PY'
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
PY

# ---------------------------------------------------------------- watermark_video.py
cat > watermark_video.py <<'PY'
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
PY

# ---------------------------------------------------------------- decode_video_watermark.py
cat > decode_video_watermark.py <<'PY'
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
PY

# ---------------------------------------------------------------- test.py
cat > test.py <<'PY'
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
PY

# ---------------------------------------------------------------- requirements.txt
cat > requirements.txt <<'REQ'
opencv-python
numpy
REQ

# ---------------------------------------------------------------- chmod & git
chmod +x *.py

if [ ! -d ".git" ]; then
  echo "🔧 Initializing git repository"
  git init -q
fi

git add generate_video.py watermark_video.py decode_video_watermark.py test.py requirements.txt
git commit -q -m "initial watermark suite"

echo "✅ Watermark demo files created and committed."
echo "Next steps:"
echo "  pip install -r requirements.txt"
echo "  python test.py"