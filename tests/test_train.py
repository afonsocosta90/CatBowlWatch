"""Tests for training/train.py argument parsing and config plumbing.

Does not import ultralytics — that's lazy-imported inside main().
"""

from pathlib import Path

import pytest

from training.train import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_BATCH,
    DEFAULT_DATA_YAML,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_OUTPUT_PT,
    DEFAULT_PATIENCE,
    DEFAULT_RUN_NAME,
    DEFAULT_RUN_PROJECT,
    TrainConfig,
    copy_best_weights,
    parse_args,
)


def test_parse_args_defaults():
    cfg = parse_args([])
    assert cfg.data == DEFAULT_DATA_YAML
    assert cfg.weights == DEFAULT_BASE_WEIGHTS
    assert cfg.epochs == DEFAULT_EPOCHS
    assert cfg.batch == DEFAULT_BATCH
    assert cfg.imgsz == DEFAULT_IMGSZ
    assert cfg.patience == DEFAULT_PATIENCE
    assert cfg.device == DEFAULT_DEVICE
    assert cfg.project == DEFAULT_RUN_PROJECT
    assert cfg.name == DEFAULT_RUN_NAME
    assert cfg.output == DEFAULT_OUTPUT_PT


def test_parse_args_overrides():
    cfg = parse_args([
        "--epochs", "5",
        "--batch", "2",
        "--imgsz", "416",
        "--device", "cuda:0",
        "--data", "foo/bar.yaml",
        "--name", "exp42",
    ])
    assert cfg.epochs == 5
    assert cfg.batch == 2
    assert cfg.imgsz == 416
    assert cfg.device == "cuda:0"
    assert cfg.data == "foo/bar.yaml"
    assert cfg.name == "exp42"


def test_to_ultralytics_kwargs_shape():
    cfg = TrainConfig(
        data="d.yaml", epochs=10, batch=4, imgsz=320, patience=5, device="cpu",
        project="proj", name="run", weights="yolov8n.pt", output="out.pt",
    )
    kwargs = cfg.to_ultralytics_kwargs()
    assert set(kwargs.keys()) == {"data", "epochs", "batch", "imgsz", "patience", "device", "project", "name"}
    assert "output" not in kwargs  # output is not a YOLO.train kwarg
    assert "weights" not in kwargs  # weights goes into YOLO(...) ctor, not train()


def test_copy_best_weights_happy_path(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    src = run_dir / "weights" / "best.pt"
    src.write_bytes(b"fake checkpoint bytes")
    dst = tmp_path / "models" / "catbowlwatch.pt"
    out = copy_best_weights(run_dir, dst)
    assert out == dst
    assert dst.is_file()
    assert dst.read_bytes() == b"fake checkpoint bytes"


def test_copy_best_weights_missing_source(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        copy_best_weights(run_dir, tmp_path / "out.pt")
