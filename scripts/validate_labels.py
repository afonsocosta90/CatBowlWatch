"""Validate YOLO-format labels for CatBowlWatch.

Checks:
  - every image has a matching .txt label file (empty file = no objects, allowed)
  - every non-empty line is `class_id cx cy w h`
  - class_id is in the allowed set (default {0, 1} for bowl_empty / bowl_not_empty)
  - all four coords are floats in [0, 1]
  - bbox is non-degenerate (w > 0 and h > 0)

Exits with code 0 on success, 1 on validation failure, 2 on usage error.
Prints a per-class summary.

Usage:
    python scripts/validate_labels.py --images data/raw/labelled/images --labels data/raw/labelled/labels
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASS_IDS = {0, 1}
CLASS_NAMES = {0: "bowl_empty", 1: "bowl_not_empty"}


def validate(
    images_dir: Path,
    labels_dir: Path,
    allowed_class_ids: set[int],
) -> tuple[list[str], Counter]:
    errors: list[str] = []
    counts: Counter = Counter()

    if not images_dir.is_dir():
        errors.append(f"images dir does not exist: {images_dir}")
        return errors, counts
    if not labels_dir.is_dir():
        errors.append(f"labels dir does not exist: {labels_dir}")
        return errors, counts

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    label_files = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

    if not images:
        errors.append(f"no images found in {images_dir}")
        return errors, counts

    image_stems = {p.stem for p in images}

    for img in images:
        lbl = label_files.get(img.stem)
        if lbl is None:
            errors.append(f"missing label for image: {img.name}")
            continue
        errors.extend(_validate_label_file(lbl, allowed_class_ids, counts))

    for stem, lbl in label_files.items():
        if stem not in image_stems:
            errors.append(f"orphan label (no matching image): {lbl.name}")

    return errors, counts


def _validate_label_file(
    path: Path,
    allowed_class_ids: set[int],
    counts: Counter,
) -> list[str]:
    errors: list[str] = []
    try:
        # utf-8-sig transparently strips a UTF-8 BOM if one is present (Windows
        # Notepad / PowerShell Set-Content write BOMs by default), and is
        # otherwise identical to utf-8.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        return [f"{path.name}: not valid UTF-8 ({e})"]

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{lineno}: expected 5 fields, got {len(parts)}: {line!r}")
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"{path.name}:{lineno}: class_id is not an integer: {parts[0]!r}")
            continue
        if class_id not in allowed_class_ids:
            errors.append(
                f"{path.name}:{lineno}: class_id {class_id} not in allowed {sorted(allowed_class_ids)}"
            )
            continue
        try:
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{lineno}: bbox values are not all floats: {parts[1:]}")
            continue
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= val <= 1.0):
                errors.append(f"{path.name}:{lineno}: {name}={val} outside [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"{path.name}:{lineno}: degenerate bbox w={w} h={h}")
        counts[class_id] += 1

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO-format labels.")
    parser.add_argument("--images", required=True, help="Directory of image files")
    parser.add_argument("--labels", required=True, help="Directory of .txt label files")
    parser.add_argument(
        "--allowed-class-ids",
        nargs="+",
        type=int,
        default=sorted(DEFAULT_CLASS_IDS),
        help="Allowed class IDs (default: 0 1)",
    )
    args = parser.parse_args()

    errors, counts = validate(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        allowed_class_ids=set(args.allowed_class_ids),
    )

    if counts:
        print("Class distribution:")
        for cid, n in sorted(counts.items()):
            name = CLASS_NAMES.get(cid, f"class_{cid}")
            print(f"  {cid} ({name}): {n}")

    if errors:
        print(f"\n{len(errors)} validation error(s):", file=sys.stderr)
        for e in errors[:50]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
