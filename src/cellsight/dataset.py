"""Lazy image index: one row per file, nothing decoded until a component asks for pixels.

Layout: <data_dir>/<class name>/<image file>. An optional metadata CSV maps a file name to a
`group` (slide, patient or source). Without metadata, the near-duplicate groups from
`dedupe.py` are the groups.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .classes import CLASS_INDEX, CLASSES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetError(ValueError):
    """The image folder breaks the layout contract."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:10]))
        self.errors = errors


@dataclass
class Manifest:
    frame: pd.DataFrame  # columns: path, label, label_index, group, sha256
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def y(self) -> np.ndarray:
        return self.frame["label_index"].to_numpy(dtype=int)

    @property
    def paths(self) -> list[Path]:
        return [Path(p) for p in self.frame["path"]]

    def subset(self, index) -> "Manifest":
        return Manifest(self.frame.iloc[np.asarray(index)].reset_index(drop=True), list(self.warnings))


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Decode one image to RGB uint8 (H, W, 3), optionally resized to size x size."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(data_dir: str | Path, metadata_csv: str | Path | None = None, check_decode: bool = True) -> Manifest:
    root = Path(data_dir)
    if not root.is_dir():
        raise DatasetError([f"{root} is not a folder. See data/README.md or run `cellsight synth`."])
    errors: list[str] = []
    warnings: list[str] = []
    folders = {p.name: p for p in root.iterdir() if p.is_dir()}
    unknown = sorted(set(folders) - set(CLASSES))
    if unknown:
        warnings.append(f"ignored folders that are not classes: {unknown}")
    missing = [c for c in CLASSES if c not in folders]
    if missing:
        errors.append(f"missing class folders: {missing}")
    rows = []
    for cls in CLASSES:
        if cls not in folders:
            continue
        files = sorted(p for p in folders[cls].iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            errors.append(f"class folder {cls} has no images")
        for p in files:
            if check_decode:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except (UnidentifiedImageError, OSError):
                    errors.append(f"cannot decode {p}")
                    continue
            rows.append({"path": str(p), "file": p.name, "label": cls, "label_index": CLASS_INDEX[cls],
                         "sha256": _sha256(p)})
    if errors:
        raise DatasetError(errors)
    frame = pd.DataFrame(rows)
    exact = frame["sha256"].duplicated(keep=False)
    if exact.any():
        warnings.append(f"{int(exact.sum())} files are byte-identical copies (they share one group)")
    frame["group"] = ""
    if metadata_csv is not None:
        meta = pd.read_csv(metadata_csv, dtype=str)
        if not {"file", "group"} <= set(meta.columns):
            raise DatasetError(["metadata CSV needs the columns file, group"])
        mapping = dict(zip(meta["file"], meta["group"]))
        frame["group"] = frame["file"].map(mapping).fillna("")
        n_missing = int((frame["group"] == "").sum())
        if n_missing:
            warnings.append(f"{n_missing} images have no group in the metadata")
    counts = frame["label"].value_counts()
    if counts.max() > 3 * counts.min():
        warnings.append(f"class imbalance: {counts.to_dict()}")
    return Manifest(frame, warnings)
