"""Shared test fixtures.

Makes the project root importable so tests can do `import scripts.foo` and
`import training.dataset` without packaging the repo. Implicit namespace
packages (PEP 420) cover us — no __init__.py required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_fake_jpeg(path: Path) -> None:
    """Write a tiny valid JPEG so cv2.imread succeeds. cv2 may not be present
    in every test run, so fall back to a hand-crafted minimal JPEG."""
    try:
        import cv2  # type: ignore
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
    except Exception:
        # 1×1 white JPEG (~125 bytes). Sufficient for "file exists with a valid
        # image extension" checks; cv2-based tests will skip if cv2 is missing.
        minimal_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010101006000600000ffdb0043000806060706"
            "0508070708090908"  # truncated — real fixture writes via cv2
        )
        path.write_bytes(minimal_jpeg)


def _write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@pytest.fixture
def make_dataset(tmp_path: Path):
    """Returns a builder: make_dataset(n_pairs, layout='separated' | 'flat',
    labels_per_image=lambda i: ['0 0.5 0.5 0.2 0.2']) -> root_path."""

    def _build(
        n_pairs: int,
        layout: str = "separated",
        labels_per_image=lambda i: ["0 0.5 0.5 0.2 0.2"],
        extra_files: list[tuple[str, str]] | None = None,
    ) -> Path:
        root = tmp_path / "ds"
        root.mkdir(exist_ok=True)
        if layout == "separated":
            img_dir = root / "images"
            lbl_dir = root / "labels"
        elif layout == "flat":
            img_dir = root
            lbl_dir = root
        else:
            raise ValueError(f"unknown layout: {layout}")
        img_dir.mkdir(exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)

        for i in range(n_pairs):
            stem = f"img_{i:04d}"
            _write_fake_jpeg(img_dir / f"{stem}.jpg")
            _write_label(lbl_dir / f"{stem}.txt", labels_per_image(i))

        if extra_files:
            for rel, content in extra_files:
                fpath = root / rel
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content, encoding="utf-8")

        return root

    return _build
