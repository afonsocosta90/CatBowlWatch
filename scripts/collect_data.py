"""Frame sampler for CatBowlWatch data collection.

Samples frames from a video file or webcam at a fixed time interval and saves
them as numbered JPEGs. Use this to build the training/val image dataset before
labeling in Roboflow.

Usage:
    python scripts/collect_data.py --source data/videos/sample_video.mp4 --interval 1.0
    python scripts/collect_data.py --source 0 --interval 0.5  # webcam index 0
"""

import argparse
import os
import sys
import time

import cv2


def _next_index(out_dir: str, prefix: str) -> int:
    """Return the next available frame index to avoid overwriting existing files."""
    existing = [
        f for f in os.listdir(out_dir)
        if f.startswith(prefix + "_") and f.endswith(".jpg")
    ]
    if not existing:
        return 0
    indices = []
    for name in existing:
        stem = name[len(prefix) + 1:-4]
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices) + 1 if indices else 0


def collect(source, interval: float, out_dir: str, prefix: str, max_frames: int | None):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source '{source}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(fps * interval))

    idx = _next_index(out_dir, prefix)
    saved = 0
    frame_pos = 0

    print(f"Source FPS: {fps:.1f}  |  saving every {frame_skip} frames  |  output: {out_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_pos % frame_skip == 0:
            filename = os.path.join(out_dir, f"{prefix}_{idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            idx += 1
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_pos += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample frames from video or webcam.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video file path or webcam index (default: 0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between saved frames (default: 1.0)",
    )
    parser.add_argument(
        "--out",
        default="data/images",
        help="Output directory (default: data/images)",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Filename prefix (default: frame)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after saving this many frames",
    )
    args = parser.parse_args()

    # Treat numeric strings as webcam indices
    source = int(args.source) if args.source.isdigit() else args.source

    collect(
        source=source,
        interval=args.interval,
        out_dir=args.out,
        prefix=args.prefix,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
