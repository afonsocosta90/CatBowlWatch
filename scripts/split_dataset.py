"""Random train/val/test split for the labelled dataset.

Reads `<src>/images/` + `<src>/labels/`, splits randomly with a fixed seed, and
copies (does not move — re-runs are safe) into:
    <dst>/images/{train,val,test}/
    <dst>/labels/{train,val,test}/
Writes `<dst>/data.yaml` (Ultralytics format) at the end.

Default ratios are 0.70 / 0.15 / 0.15. They must sum to 1.0 (±1e-6).

The split is deterministic given --seed. If the destination already contains
the split, it is wiped and rewritten so the split stays consistent with the
current --seed and --ratios.

Usage:
    python scripts/split_dataset.py --src data/raw/labelled --dst data
    python scripts/split_dataset.py --src data/raw/labelled --dst data --ratios 0.8 0.1 0.1 --seed 7
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = ["bowl_empty", "bowl_not_empty"]


def gather_pairs(src: Path) -> list[tuple[Path, Path]]:
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Error: expected {images_dir} and {labels_dir} to exist")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            print(f"WARN  skipping {img.name}: no matching label", file=sys.stderr)
            continue
        pairs.append((img, lbl))
    return pairs


def compute_split_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    # Floor first two, assign the remainder to test so the totals always equal n.
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def split(pairs: list[tuple[Path, Path]], ratios: tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = compute_split_sizes(len(shuffled), ratios)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def reset_split_dirs(dst: Path) -> None:
    for split_name in SPLITS:
        for kind in ("images", "labels"):
            d = dst / kind / split_name
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def write_split(dst: Path, split_name: str, items: list[tuple[Path, Path]]) -> None:
    img_dir = dst / "images" / split_name
    lbl_dir = dst / "labels" / split_name
    for img, lbl in items:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)


def write_data_yaml(dst: Path, class_names: list[str]) -> Path:
    # data.yaml is consumed by Ultralytics' YOLO trainer in Phase 2.
    # `path` is the dataset root; train/val/test are relative to that.
    yaml_path = dst / "data.yaml"
    lines = [
        f"path: {dst.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(class_names)],
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Random seeded train/val/test split.")
    parser.add_argument("--src", required=True, help="Source dir with images/ and labels/")
    parser.add_argument("--dst", required=True, help="Destination dataset root")
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (default: 0.70 0.15 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    total = sum(args.ratios)
    if abs(total - 1.0) > 1e-6:
        print(f"Error: --ratios must sum to 1.0, got {total}", file=sys.stderr)
        return 2

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    pairs = gather_pairs(src)
    if not pairs:
        print(f"Error: no labelled pairs found under {src}", file=sys.stderr)
        return 1

    splits = split(pairs, tuple(args.ratios), args.seed)
    reset_split_dirs(dst)
    for name in SPLITS:
        write_split(dst, name, splits[name])

    yaml_path = write_data_yaml(dst, CLASS_NAMES)

    print(f"Source pairs:   {len(pairs)}")
    for name in SPLITS:
        print(f"  {name:<5} {len(splits[name])}")
    print(f"Wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
