"""Generate a tiny synthetic labelled dataset for pipeline dry-runs.

Drops a flat layout into <out>/ (default: data/raw/labelled/) so we can exercise
the full `make data` chain (organise -> validate -> split) without real images.

Output covers:
  - normal 2-bowl frames (one empty, one not_empty, x-ordered)
  - a single-bowl frame
  - a frame whose label file is empty (validator accepts this)
  - a meta sibling (classes.txt) that organise_raw must ignore

This script is a dev aid for Phase 1b. Not used at runtime. The output dir is
gitignored via .gitignore's `data/raw/` rule.

Usage:
    poetry run python scripts/_generate_synthetic.py
    poetry run python scripts/_generate_synthetic.py --count 20 --out data/raw/labelled
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_BOWL_EMPTY = 0
CLASS_BOWL_NOT_EMPTY = 1
IMG_SIZE = 640


def _render_frame(rng: random.Random, kind: str) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Return (BGR image, list of YOLO labels) for a frame of the given kind."""
    bg_gray = rng.randint(40, 200)
    img = np.full((IMG_SIZE, IMG_SIZE, 3), bg_gray, dtype=np.uint8)
    img += rng.randint(0, 15) * np.random.randint(0, 2, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    labels: list[tuple[int, float, float, float, float]] = []

    def draw_bowl(cx_n: float, cy_n: float, class_id: int) -> None:
        radius = rng.randint(70, 95)
        cx, cy = int(cx_n * IMG_SIZE), int(cy_n * IMG_SIZE)
        rim_color = (180, 180, 180)
        cv2.circle(img, (cx, cy), radius, rim_color, thickness=-1)
        inner_radius = radius - 12
        if class_id == CLASS_BOWL_EMPTY:
            cv2.circle(img, (cx, cy), inner_radius, (60, 60, 60), thickness=-1)
        else:
            cv2.circle(img, (cx, cy), inner_radius, (50, 90, 140), thickness=-1)
            for _ in range(8):
                jx = cx + rng.randint(-inner_radius // 2, inner_radius // 2)
                jy = cy + rng.randint(-inner_radius // 2, inner_radius // 2)
                cv2.circle(img, (jx, jy), rng.randint(4, 9), (40, 70, 110), thickness=-1)
        w_n = (2 * radius) / IMG_SIZE
        h_n = (2 * radius) / IMG_SIZE
        labels.append((class_id, cx_n, cy_n, w_n, h_n))

    if kind == "two_bowls":
        left_class = rng.choice([CLASS_BOWL_EMPTY, CLASS_BOWL_NOT_EMPTY])
        right_class = rng.choice([CLASS_BOWL_EMPTY, CLASS_BOWL_NOT_EMPTY])
        draw_bowl(0.30, 0.50, left_class)
        draw_bowl(0.70, 0.50, right_class)
    elif kind == "one_bowl":
        draw_bowl(0.50, 0.50, rng.choice([CLASS_BOWL_EMPTY, CLASS_BOWL_NOT_EMPTY]))
    elif kind == "no_bowls":
        pass
    else:
        raise ValueError(f"unknown kind: {kind}")

    return img, labels


def _write_label(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic labelled dataset.")
    parser.add_argument("--out", default="data/raw/labelled", help="Output dir (flat layout)")
    parser.add_argument("--count", type=int, default=12, help="Number of frames (default: 12)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan: list[str] = []
    for i in range(args.count):
        if i == args.count - 2:
            plan.append("one_bowl")
        elif i == args.count - 1:
            plan.append("no_bowls")
        else:
            plan.append("two_bowls")

    for i, kind in enumerate(plan):
        stem = f"synth_{i:03d}"
        img, labels = _render_frame(rng, kind)
        cv2.imwrite(str(out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _write_label(out / f"{stem}.txt", labels)

    (out / "classes.txt").write_text("bowl_empty\nbowl_not_empty\n", encoding="utf-8")

    print(f"Wrote {args.count} synthetic image/label pairs to {out}")
    print(f"  two_bowls: {plan.count('two_bowls')}  one_bowl: {plan.count('one_bowl')}  no_bowls: {plan.count('no_bowls')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
