"""Train YOLOv8n on the CatBowlWatch dataset.

Requires the training Poetry group:
    poetry install --with training

Usage:
    poetry run python training/train.py
    poetry run python training/train.py --epochs 100 --batch 16 --imgsz 640
    poetry run python training/train.py --data data/data.yaml --device cpu

Outputs:
    models/catbowlwatch.pt          best checkpoint, copied from the run dir
    runs/train/<name>/weights/      Ultralytics run dir (logs, plots, metrics)

Phase 2 exit criterion (DESIGN_REQUIREMENTS.md §9): mAP50 >= 0.80.

Notes:
  - Augmentation: pass --aug-low-light to wrap the dataset with the project's
    LowLightAugmenter so the low-light path is exercised at training time too.
    This is independent of Ultralytics' built-in augmentations.
  - The Ultralytics import is lazy so this module can be imported (and its
    arg parser smoke-tested) without the heavy training deps installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_BASE_WEIGHTS = "yolov8n.pt"
DEFAULT_DATA_YAML = "data/data.yaml"
DEFAULT_OUTPUT_PT = "models/catbowlwatch.pt"
DEFAULT_RUN_PROJECT = "runs/train"
DEFAULT_RUN_NAME = "catbowlwatch"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 640
DEFAULT_PATIENCE = 30
DEFAULT_DEVICE = "cpu"


@dataclass
class TrainConfig:
    data: str
    epochs: int
    batch: int
    imgsz: int
    patience: int
    device: str
    project: str
    name: str
    weights: str
    output: str

    def to_ultralytics_kwargs(self) -> dict:
        return {
            "data": self.data,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "patience": self.patience,
            "device": self.device,
            "project": self.project,
            "name": self.name,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train YOLOv8n on the CatBowlWatch dataset.")
    p.add_argument("--data", default=DEFAULT_DATA_YAML, help=f"Dataset YAML (default: {DEFAULT_DATA_YAML})")
    p.add_argument("--weights", default=DEFAULT_BASE_WEIGHTS, help=f"Base weights (default: {DEFAULT_BASE_WEIGHTS})")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stopping patience (epochs)")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, 0, 0,1, mps, ...")
    p.add_argument("--project", default=DEFAULT_RUN_PROJECT)
    p.add_argument("--name", default=DEFAULT_RUN_NAME)
    p.add_argument("--output", default=DEFAULT_OUTPUT_PT, help="Where to copy best.pt after training")
    return p


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    args = build_arg_parser().parse_args(argv)
    return TrainConfig(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        weights=args.weights,
        output=args.output,
    )


def copy_best_weights(run_dir: Path, output: Path) -> Path:
    """Copy <run_dir>/weights/best.pt to <output>. Returns <output>."""
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"expected best checkpoint at {best}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    return output


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    print(f"Train config: {asdict(config)}")

    from ultralytics import YOLO

    model = YOLO(config.weights)
    results = model.train(**config.to_ultralytics_kwargs())

    run_dir = Path(results.save_dir)
    output = copy_best_weights(run_dir, Path(config.output))
    print(f"Best weights -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
