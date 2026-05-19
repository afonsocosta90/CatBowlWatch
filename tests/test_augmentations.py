"""Tests for training/augmentations.py low-light adaptive preprocessing."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from training.augmentations import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    LowLightAugmenter,
    adaptive_low_light,
    low_light_transform,
    mean_brightness,
)


def _solid(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _low_contrast_dark(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(20, 40, size=(h, w), dtype=np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def test_mean_brightness_black_and_white():
    assert mean_brightness(_solid(0)) == pytest.approx(0.0, abs=1e-6)
    assert mean_brightness(_solid(255)) == pytest.approx(255.0, abs=1e-6)


def test_mean_brightness_mid():
    assert mean_brightness(_solid(128)) == pytest.approx(128.0, abs=1.0)


def test_mean_brightness_rejects_non_bgr():
    with pytest.raises(ValueError):
        mean_brightness(np.zeros((10, 10), dtype=np.uint8))


def test_low_light_transform_preserves_shape_and_dtype():
    img = _low_contrast_dark()
    out = low_light_transform(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_low_light_transform_replicates_grayscale_across_channels():
    img = _low_contrast_dark()
    out = low_light_transform(img)
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_low_light_transform_improves_contrast():
    img = _low_contrast_dark()
    before = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
    after = float(cv2.cvtColor(low_light_transform(img), cv2.COLOR_BGR2GRAY).std())
    assert after > before


def test_adaptive_low_light_identity_above_threshold():
    bright = _solid(200)
    out = adaptive_low_light(bright, threshold=DEFAULT_BRIGHTNESS_THRESHOLD)
    assert out is bright  # identity return, no copy


def test_adaptive_low_light_applies_below_threshold():
    dark = _low_contrast_dark()
    assert mean_brightness(dark) < DEFAULT_BRIGHTNESS_THRESHOLD
    out = adaptive_low_light(dark, threshold=DEFAULT_BRIGHTNESS_THRESHOLD)
    assert out is not dark
    np.testing.assert_array_equal(out[..., 0], out[..., 1])


def test_augmenter_class_matches_adaptive_low_light():
    dark = _low_contrast_dark()
    aug = LowLightAugmenter()
    np.testing.assert_array_equal(aug(dark), adaptive_low_light(dark))


def test_augmenter_class_identity_for_bright_input():
    bright = _solid(200)
    aug = LowLightAugmenter()
    out = aug(bright)
    assert out is bright


def test_low_light_transform_rejects_non_bgr():
    with pytest.raises(ValueError):
        low_light_transform(np.zeros((10, 10), dtype=np.uint8))
