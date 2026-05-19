"""Low-light adaptive preprocessing for training and inference.

The same transform parameters MUST be used by the C++ inference preprocessor so
the model sees the same distribution at train and inference time. Keep the
defaults below in sync with the inference-side constants (see
ARCHITECTURE.md §4.2 and §6).

Public API:
    mean_brightness(bgr) -> float                 # 0..255
    low_light_transform(bgr, ...) -> np.ndarray   # unconditional CLAHE
    adaptive_low_light(bgr, ...) -> np.ndarray    # threshold-gated CLAHE
    LowLightAugmenter(...)                         # callable wrapper for Dataset.transform
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_BRIGHTNESS_THRESHOLD: int = 50
DEFAULT_CLAHE_CLIP_LIMIT: float = 2.0
DEFAULT_CLAHE_TILE_GRID: tuple[int, int] = (8, 8)


def mean_brightness(bgr: np.ndarray) -> float:
    """Mean luminance of a BGR uint8 image, returned in [0, 255]."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {bgr.shape}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def low_light_transform(
    bgr: np.ndarray,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
) -> np.ndarray:
    """Grayscale + CLAHE, replicated back to 3-channel BGR.

    Output has the same shape and dtype as the input. All three channels are
    equal (grayscale replication) — this matches the IR-feed look the model is
    being trained to be robust to.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {bgr.shape}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    boosted = clahe.apply(gray)
    return cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)


def adaptive_low_light(
    bgr: np.ndarray,
    threshold: int = DEFAULT_BRIGHTNESS_THRESHOLD,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
) -> np.ndarray:
    """Apply low_light_transform only if mean brightness is below threshold.

    Returns the input unchanged (same object, no copy) above threshold.
    """
    if mean_brightness(bgr) >= threshold:
        return bgr
    return low_light_transform(bgr, clip_limit=clip_limit, tile_grid=tile_grid)


class LowLightAugmenter:
    """Stateful wrapper that reuses the CLAHE object across calls.

    Designed to plug into `BowlDataset(transform=...)`. Stateless from the
    caller's perspective: same input always produces the same output for fixed
    parameters.
    """

    def __init__(
        self,
        threshold: int = DEFAULT_BRIGHTNESS_THRESHOLD,
        clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
        tile_grid: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
    ) -> None:
        self.threshold = threshold
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if mean_brightness(bgr) >= self.threshold:
            return bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boosted = self._clahe.apply(gray)
        return cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)
