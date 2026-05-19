"""Tests for scripts/validate_labels.py."""

from pathlib import Path

import pytest

from scripts.validate_labels import validate


def test_clean_dataset_passes(make_dataset):
    root: Path = make_dataset(n_pairs=5)
    errors, counts = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert errors == []
    assert counts[0] == 5


def test_class_id_out_of_range_fails(make_dataset):
    root: Path = make_dataset(
        n_pairs=2,
        labels_per_image=lambda i: ["2 0.5 0.5 0.2 0.2"],
    )
    errors, _ = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert any("class_id 2" in e for e in errors)


def test_coord_out_of_range_fails(make_dataset):
    root: Path = make_dataset(
        n_pairs=1,
        labels_per_image=lambda i: ["0 1.5 0.5 0.2 0.2"],
    )
    errors, _ = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert any("cx=1.5" in e for e in errors)


def test_degenerate_bbox_fails(make_dataset):
    root: Path = make_dataset(
        n_pairs=1,
        labels_per_image=lambda i: ["0 0.5 0.5 0.0 0.2"],
    )
    errors, _ = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert any("degenerate bbox" in e for e in errors)


def test_orphan_label_fails(make_dataset, tmp_path):
    root: Path = make_dataset(n_pairs=2)
    (root / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    errors, _ = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert any("orphan label" in e and "ghost" in e for e in errors)


def test_missing_label_fails(make_dataset):
    root: Path = make_dataset(n_pairs=2)
    (root / "labels" / "img_0000.txt").unlink()
    errors, _ = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert any("missing label" in e and "img_0000" in e for e in errors)


def test_empty_label_allowed_negative_example(make_dataset):
    root: Path = make_dataset(
        n_pairs=2,
        labels_per_image=lambda i: [] if i == 0 else ["0 0.5 0.5 0.2 0.2"],
    )
    errors, counts = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert errors == []
    assert counts[0] == 1


def test_malformed_line_fails(make_dataset):
    root: Path = make_dataset(
        n_pairs=1,
        labels_per_image=lambda i: ["0 0.5 0.5 0.2"],  # only 4 fields
    )
    errors, _ = validate(root / "images", root / "labels", allowed_class_ids={0, 1})
    assert any("expected 5 fields" in e for e in errors)
