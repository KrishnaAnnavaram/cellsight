"""Model-agnostic occlusion maps and a check that the evidence lies on the cell, not on the background."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage


PredictImages = Callable[[list[np.ndarray]], np.ndarray]


def occlusion_map(predict: PredictImages, img: np.ndarray, class_index: int, patch: int = 8,
                  stride: int = 4) -> np.ndarray:
    """Drop of the class log-probability when a patch of the median colour hides each region.

    The log scale keeps the map informative when the probability is close to 1. The map has
    the same size as the image.
    """
    h, w = img.shape[:2]
    fill = np.median(img.reshape(-1, 3), axis=0)
    base = np.log(max(predict([img])[0, class_index], 1e-12))
    heat = np.zeros((h, w))
    count = np.zeros((h, w))
    batch, boxes = [], []
    for y0 in range(0, max(h - patch, 0) + 1, stride):
        for x0 in range(0, max(w - patch, 0) + 1, stride):
            im = img.copy()
            im[y0:y0 + patch, x0:x0 + patch] = fill
            batch.append(im)
            boxes.append((y0, x0))
    probs = np.log(np.clip(predict(batch)[:, class_index], 1e-12, 1.0))
    for (y0, x0), p in zip(boxes, probs):
        heat[y0:y0 + patch, x0:x0 + patch] += base - p
        count[y0:y0 + patch, x0:x0 + patch] += 1
    return heat / np.maximum(count, 1)


def cell_mask(img: np.ndarray) -> np.ndarray:
    """Approximate mask of the central cell.

    Pixels that differ in colour from the border median are candidates. A closing joins the
    nucleus lobes and the cytoplasm, holes are filled, and the component nearest to the image
    centre is kept.
    """
    x = img.astype(float)
    border = np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]])
    dist = np.linalg.norm(x - np.median(border, axis=0), axis=-1)
    mask = dist > np.percentile(np.linalg.norm(border - np.median(border, axis=0), axis=-1), 95) + 10
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    h, w = mask.shape
    centres = ndimage.center_of_mass(mask, labels, range(1, n + 1))
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    score = [np.hypot(cy - h / 2, cx - w / 2) - 0.05 * sz for (cy, cx), sz in zip(centres, sizes)]
    return labels == (int(np.argmin(score)) + 1)


def focus_ratio(heat: np.ndarray, mask: np.ndarray) -> float:
    """(share of positive evidence inside the mask) / (share of the image area inside the mask).

    A value above 1 means that the evidence concentrates on the cell. A value near or below 1
    means that the model uses background or stain as much as the cell.
    """
    pos = np.clip(heat, 0, None)
    if pos.sum() <= 0 or not mask.any():
        return float("nan")
    return float((pos[mask].sum() / pos.sum()) / mask.mean())


def save_overlay(img: np.ndarray, heat: np.ndarray, path: str | Path) -> Path:
    """Write the image and the heat map side by side as a PNG (red = evidence for the class)."""
    pos = np.clip(heat, 0, None)
    pos = pos / pos.max() if pos.max() > 0 else pos
    red = img.astype(float).copy()
    red[..., 0] = np.clip(red[..., 0] * (1 - pos) + 255 * pos, 0, 255)
    red[..., 1:] = red[..., 1:] * (1 - 0.6 * pos[..., None])
    panel = np.concatenate([img, red.astype(np.uint8)], axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).resize((panel.shape[1] * 4, panel.shape[0] * 4), Image.Resampling.NEAREST).save(path)
    return path
