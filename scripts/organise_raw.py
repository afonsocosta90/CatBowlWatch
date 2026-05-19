"""Organise a flat labelled directory into images/ + labels/ subdirs.

Some labelling tools (LabelImg, CVAT in certain modes) export images and YOLO
.txt label files side by side in one folder. Roboflow's YOLOv8 export already
separates them; running this script on that layout is a no-op.

This script:
  - finds every .jpg/.jpeg/.png/.bmp under --src
  - finds every .txt under --src that is NOT data.yaml / classes.txt / notes.txt
  - matches them by filename stem
  - moves each pair into <dst>/images/ and <dst>/labels/
  - prints a summary, including any orphans (image with no label, or vice versa)

Idempotent: files already in <dst>/images/ or <dst>/labels/ are left alone.

Usage:
    python scripts/organise_raw.py --src data/raw/labelled --dst data/raw/labelled
    python scripts/organise_raw.py --src data/raw/labelled --dst data/raw/labelled --copy
"""

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
META_FILENAMES = {"data.yaml", "classes.txt", "notes.txt", "README.txt", "README.md"}


def find_files(src: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Return (images_by_stem, labels_by_stem) for files directly under src
    OR already inside src/images and src/labels. Files in other subdirs are ignored."""
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}

    candidate_dirs = [src, src / "images", src / "labels"]
    for d in candidate_dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.name in META_FILENAMES:
                continue
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                images.setdefault(p.stem, p)
            elif ext == ".txt":
                labels.setdefault(p.stem, p)

    return images, labels


def organise(src: Path, dst: Path, copy: bool) -> int:
    images, labels = find_files(src)

    img_out = dst / "images"
    lbl_out = dst / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    matched_stems = sorted(images.keys() & labels.keys())
    orphan_images = sorted(images.keys() - labels.keys())
    orphan_labels = sorted(labels.keys() - images.keys())

    moved = 0
    for stem in matched_stems:
        src_img = images[stem]
        src_lbl = labels[stem]
        dst_img = img_out / src_img.name
        dst_lbl = lbl_out / src_lbl.name

        if src_img.resolve() != dst_img.resolve():
            _transfer(src_img, dst_img, copy)
            moved += 1
        if src_lbl.resolve() != dst_lbl.resolve():
            _transfer(src_lbl, dst_lbl, copy)

    print(f"Matched pairs: {len(matched_stems)}  |  moved/copied: {moved}")
    if orphan_images:
        print(f"WARN  {len(orphan_images)} image(s) without a label:")
        for stem in orphan_images[:10]:
            print(f"  {images[stem].name}")
        if len(orphan_images) > 10:
            print(f"  ... and {len(orphan_images) - 10} more")
    if orphan_labels:
        print(f"WARN  {len(orphan_labels)} label(s) without an image:")
        for stem in orphan_labels[:10]:
            print(f"  {labels[stem].name}")
        if len(orphan_labels) > 10:
            print(f"  ... and {len(orphan_labels) - 10} more")

    return 0 if matched_stems else 1


def _transfer(src: Path, dst: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> int:
    parser = argparse.ArgumentParser(description="Organise raw labelled data into images/ and labels/.")
    parser.add_argument("--src", required=True, help="Source directory")
    parser.add_argument("--dst", required=True, help="Destination directory (images/ and labels/ created inside)")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        print(f"Error: --src is not a directory: {src}", file=sys.stderr)
        return 2

    return organise(src, dst, args.copy)


if __name__ == "__main__":
    sys.exit(main())
