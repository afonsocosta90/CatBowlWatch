"""Tests for scripts/split_dataset.py."""

from pathlib import Path

import pytest

from scripts.split_dataset import (
    compute_split_sizes,
    gather_pairs,
    split,
    write_data_yaml,
)


def test_compute_split_sizes_totals_match_n():
    assert sum(compute_split_sizes(100, (0.7, 0.15, 0.15))) == 100
    assert sum(compute_split_sizes(7, (0.7, 0.15, 0.15))) == 7
    assert sum(compute_split_sizes(0, (0.7, 0.15, 0.15))) == 0


def test_compute_split_sizes_70_15_15_on_100():
    assert compute_split_sizes(100, (0.7, 0.15, 0.15)) == (70, 15, 15)


def test_gather_pairs_picks_only_matched(make_dataset):
    root: Path = make_dataset(n_pairs=4)
    # Orphan image (no label) and orphan label (no image)
    (root / "images" / "no_label.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "labels" / "no_image.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    pairs = gather_pairs(root)
    stems = sorted(p[0].stem for p in pairs)
    assert stems == ["img_0000", "img_0001", "img_0002", "img_0003"]


def test_split_deterministic_with_same_seed(make_dataset):
    root: Path = make_dataset(n_pairs=20)
    pairs = gather_pairs(root)
    a = split(pairs, (0.7, 0.15, 0.15), seed=42)
    b = split(pairs, (0.7, 0.15, 0.15), seed=42)
    for name in ("train", "val", "test"):
        assert [p[0].name for p in a[name]] == [p[0].name for p in b[name]]


def test_split_changes_with_different_seed(make_dataset):
    root: Path = make_dataset(n_pairs=20)
    pairs = gather_pairs(root)
    a = split(pairs, (0.7, 0.15, 0.15), seed=42)
    b = split(pairs, (0.7, 0.15, 0.15), seed=7)
    assert [p[0].name for p in a["train"]] != [p[0].name for p in b["train"]]


def test_split_partitions_every_pair_exactly_once(make_dataset):
    root: Path = make_dataset(n_pairs=37)
    pairs = gather_pairs(root)
    s = split(pairs, (0.7, 0.15, 0.15), seed=1)
    all_stems = [p[0].stem for p in pairs]
    split_stems: list[str] = []
    for name in ("train", "val", "test"):
        split_stems.extend(p[0].stem for p in s[name])
    assert sorted(split_stems) == sorted(all_stems)
    assert len(split_stems) == len(set(split_stems))  # no duplicates


def test_data_yaml_written_with_class_names(tmp_path: Path):
    yaml_path = write_data_yaml(tmp_path, ["bowl_empty", "bowl_not_empty"])
    text = yaml_path.read_text(encoding="utf-8")
    assert "nc: 2" in text
    assert "0: bowl_empty" in text
    assert "1: bowl_not_empty" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text


def test_end_to_end_split_writes_files(make_dataset, tmp_path: Path):
    """Smoke-test the full CLI flow via the main() entrypoint."""
    from scripts.split_dataset import main
    import sys

    src: Path = make_dataset(n_pairs=20)
    dst = tmp_path / "out"

    argv = sys.argv[:]
    sys.argv = [
        "split_dataset.py",
        "--src", str(src),
        "--dst", str(dst),
        "--ratios", "0.7", "0.15", "0.15",
        "--seed", "42",
    ]
    try:
        rc = main()
    finally:
        sys.argv = argv

    assert rc == 0
    assert (dst / "data.yaml").is_file()
    train_imgs = list((dst / "images" / "train").iterdir())
    val_imgs = list((dst / "images" / "val").iterdir())
    test_imgs = list((dst / "images" / "test").iterdir())
    assert len(train_imgs) == 14  # 20 * 0.7
    assert len(val_imgs) == 3     # int(20 * 0.15) = 3
    assert len(test_imgs) == 3    # 20 - 14 - 3
    # Every image has a paired label
    for img in train_imgs + val_imgs + test_imgs:
        split_dir = img.parent.name
        assert (dst / "labels" / split_dir / (img.stem + ".txt")).is_file()
