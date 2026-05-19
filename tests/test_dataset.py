"""Tests for training/dataset.py BowlDataset."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from training.dataset import BowlDataset


def test_loads_valid_pair(make_dataset):
    root: Path = make_dataset(n_pairs=3)
    ds = BowlDataset(str(root / "images"), str(root / "labels"))
    assert len(ds) == 3
    tensor, targets = ds[0]
    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == torch.float32
    assert len(targets) == 1
    assert targets[0]["class_id"] == 0


def test_skips_image_without_label(make_dataset):
    import numpy as np
    root: Path = make_dataset(n_pairs=2)
    cv2.imwrite(str(root / "images" / "no_label.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    ds = BowlDataset(str(root / "images"), str(root / "labels"))
    assert len(ds) == 2  # the no-label image is skipped


def test_malformed_line_is_skipped_with_warning(make_dataset, caplog):
    import logging
    root: Path = make_dataset(
        n_pairs=1,
        labels_per_image=lambda i: ["0 0.5 0.5 0.2 0.2", "garbage"],
    )
    ds = BowlDataset(str(root / "images"), str(root / "labels"))
    with caplog.at_level(logging.WARNING, logger="training.dataset"):
        _, targets = ds[0]
    assert len(targets) == 1
    assert any("Malformed label line" in rec.message for rec in caplog.records)


def test_empty_label_file_yields_no_targets(make_dataset):
    root: Path = make_dataset(
        n_pairs=1,
        labels_per_image=lambda i: [],
    )
    ds = BowlDataset(str(root / "images"), str(root / "labels"))
    _, targets = ds[0]
    assert targets == []
