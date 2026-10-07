"""Synthetic stained-cell images with class-specific nucleus shapes, slide stain shifts and duplicates.

The generator copies two properties of the public dataset that matter for evaluation:
augmented near-duplicates of one original cell, and slide-level stain differences. It is
for offline tests and demos only. It is not a model of real blood smears.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .classes import CLASSES


def _disk(yy, xx, cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _ellipse(yy, xx, cy, cx, ry, rx, angle):
    c, s = np.cos(angle), np.sin(angle)
    y, x = yy - cy, xx - cx
    u, v = x * c + y * s, -x * s + y * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1


def draw_cell(label: str, size: int, rng: np.random.Generator, stain: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    s = size / 64.0
    img = np.empty((size, size, 3))
    img[:] = np.array([232, 205, 214]) + rng.normal(0, 4, 3)
    # a few pale red cells in the background
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.uniform(0, size, 2)
        img[_disk(yy, xx, cy, cx, rng.uniform(7, 10) * s)] = [222, 170, 180]
    cy, cx = size / 2 + rng.normal(0, 2 * s, 2)
    radius = {"erythroblast": rng.uniform(13, 19), "monocyte": rng.uniform(20, 27)}.get(label, rng.uniform(17, 25)) * s
    cyto = {"basophil": [190, 160, 205], "erythroblast": [180, 165, 200], "monocyte": [185, 180, 205],
            "myeloblast": [170, 160, 210], "seg_neutrophil": [225, 195, 215]}[label]
    cell = _disk(yy, xx, cy, cx, radius)
    img[cell] = np.asarray(cyto) + rng.normal(0, 12, 3)
    dark = np.array([85, 45, 120])
    nucleus = np.zeros_like(cell)
    if label == "basophil":
        nucleus = _disk(yy, xx, cy, cx, radius * 0.75)
        grains = rng.random((size, size)) < 0.12
        img[cell & grains] = [70, 30, 100]
    elif label == "erythroblast":
        nucleus = _disk(yy, xx, cy + rng.normal(0, 1), cx + rng.normal(0, 1), radius * 0.55)
        dark = np.array([60, 30, 90])
    elif label == "monocyte":
        a = rng.uniform(0, 2 * np.pi)
        nucleus = _ellipse(yy, xx, cy, cx, radius * 0.45, radius * 0.7, a)
        bite = _disk(yy, xx, cy + np.sin(a) * radius * 0.5, cx + np.cos(a) * radius * 0.1, radius * 0.35)
        nucleus &= ~bite
        dark = np.array([120, 85, 150])
    elif label == "myeloblast":
        nucleus = _disk(yy, xx, cy, cx, radius * 0.82)
        dark = np.array([125, 85, 170])
    elif label == "seg_neutrophil":
        n_lobes = rng.integers(3, 5)
        a0 = rng.uniform(0, 2 * np.pi)
        for i in range(n_lobes):
            a = a0 + i * 2 * np.pi / n_lobes
            nucleus |= _disk(yy, xx, cy + np.sin(a) * radius * 0.4, cx + np.cos(a) * radius * 0.4, radius * 0.24)
        dark = np.array([95, 50, 125])
    img[nucleus & cell] = dark + rng.normal(0, 18, 3)
    if label == "myeloblast":
        for _ in range(rng.integers(1, 3)):
            ny, nx = cy + rng.normal(0, radius * 0.25), cx + rng.normal(0, radius * 0.25)
            img[_disk(yy, xx, ny, nx, radius * 0.12) & nucleus] = [165, 130, 200]
    img = img * stain * rng.uniform(0.9, 1.1, 3) + rng.normal(0, 12, img.shape)
    img = ndimage.gaussian_filter(img, sigma=(rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.2), 0))
    return np.clip(img, 0, 255).astype(np.uint8)


def augment_copy(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A near-duplicate: rotation by k*90 degrees, an optional flip and a small brightness change."""
    out = np.rot90(img, rng.integers(0, 4))
    if rng.random() < 0.5:
        out = out[:, ::-1]
    out = out.astype(float) * rng.uniform(0.95, 1.05)
    return np.clip(out, 0, 255).astype(np.uint8)


def generate(out_dir: str | Path, per_class: int = 40, duplicates: int = 2, size: int = 64, n_slides: int = 40,
             seed: int = 42) -> pd.DataFrame:
    """Write PNG files to out_dir/<class>/ and out_dir/metadata.csv. Return the metadata table.

    Each original image gets `duplicates` augmented copies. The metadata has the slide of each
    file (`group`) and the ID of its original (`origin`), so tests can check the dedupe step.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    stains = np.clip(rng.normal(1.0, 0.06, (n_slides, 3)), 0.8, 1.2)
    rows = []
    for label in CLASSES:
        (out / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            slide = int(rng.integers(0, n_slides))
            base = draw_cell(label, size, rng, stains[slide])
            origin = f"{label}_{i:04d}"
            images = [base] + [augment_copy(base, rng) for _ in range(duplicates)]
            for j, im in enumerate(images):
                name = f"{origin}_{j}.png"
                Image.fromarray(im).save(out / label / name)
                rows.append({"file": name, "label": label, "group": f"slide_{slide:02d}", "origin": origin})
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "metadata.csv", index=False)
    return meta
