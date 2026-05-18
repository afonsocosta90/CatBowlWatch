"""YOLO-format dataset for CatBowlWatch training.

Reads images from `images_dir` and matching YOLO label files from `labels_dir`.
Each label file contains one line per bounding box: class_id cx cy w h (normalized).

Classes:
    0 = bowl_empty
    1 = bowl_not_empty
"""

import logging
import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class BowlDataset(Dataset):
    """PyTorch Dataset for YOLO-format cat bowl images.

    Args:
        images_dir: Path to folder containing image files.
        labels_dir: Path to folder containing matching .txt label files.
        img_size: Images are resized to (img_size, img_size) before returning.
        transform: Optional callable applied to the uint8 BGR image before
            conversion to tensor. Accepts and returns a numpy array.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: int = 640,
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.transform = transform

        self.samples = self._build_index()
        if not self.samples:
            logger.warning("No valid image/label pairs found in %s", images_dir)

    def _build_index(self) -> list[Path]:
        valid = []
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                logger.warning("No label file for %s — skipping", img_path.name)
                continue
            valid.append(img_path)
        return valid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        # Load and resize image
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.transform is not None:
            img = self.transform(img)

        # BGR → RGB, HWC → CHW, uint8 → float32 [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Parse label file
        targets = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    logger.warning("Malformed label line in %s: %r", label_path.name, line)
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                targets.append({"class_id": class_id, "bbox": (cx, cy, w, h)})

        return tensor, targets
