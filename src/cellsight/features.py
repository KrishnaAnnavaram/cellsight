"""Handcrafted colour, shape and texture features for the light (CPU, no-download) base models.

These features let the full pipeline (dedupe, splits, out-of-fold stacking, evaluation,
explanations) run offline and in CI. The deep backbones in `cellsight.deep` replace them
when PyTorch and timm are installed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .dataset import load_image

FEATURE_GROUPS = ("hue_hist", "sat_hist", "val_hist", "dark_shape", "gradient_hist", "thumbnail")


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    x = img.astype(float) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    nz = diff > 1e-9
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / diff[rm]) % 6
    h[gm] = (b - r)[gm] / diff[gm] + 2
    h[bm] = (r - g)[bm] / diff[bm] + 4
    h = h / 6.0
    s = np.where(mx > 0, diff / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, s, mx], axis=-1)


def image_features(img: np.ndarray) -> np.ndarray:
    """A fixed-length vector for one RGB uint8 image (any size)."""
    hsv = _rgb_to_hsv(img)
    feats = [
        np.histogram(hsv[..., 0], bins=16, range=(0, 1))[0],
        np.histogram(hsv[..., 1], bins=8, range=(0, 1))[0],
        np.histogram(hsv[..., 2], bins=8, range=(0, 1))[0],
    ]
    feats = [f / f.sum() for f in feats]
    gray = img.astype(float).mean(axis=-1)
    dark = gray < np.percentile(gray, 25)
    labels, n_parts = ndimage.label(dark)
    sizes = ndimage.sum(dark, labels, range(1, n_parts + 1)) if n_parts else np.array([0.0])
    big = sizes[sizes >= 0.01 * dark.size]
    shape = np.array([dark.mean(), float(len(big)), float(big.max() / dark.size) if big.size else 0.0,
                      float(np.std(gray[dark])) if dark.any() else 0.0])
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    ghist = np.histogram(grad, bins=8, range=(0, 64))[0].astype(float)
    ghist /= max(ghist.sum(), 1.0)
    h, w = gray.shape
    thumb = gray[: h - h % 8, : w - w % 8].reshape(8, (h - h % 8) // 8, 8, (w - w % 8) // 8).mean(axis=(1, 3))
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return np.concatenate([*feats, shape, ghist, thumb.ravel()]).astype(np.float32)


def feature_matrix(paths, size: int = 64) -> np.ndarray:
    """Decode each image lazily (one at a time) and return an (n, d) feature matrix."""
    return np.stack([image_features(load_image(p, size)) for p in paths])
