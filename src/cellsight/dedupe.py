"""Near-duplicate detection with a DCT perceptual hash that is invariant to rotations and flips.

The public dataset holds augmented copies of one original cell (rotations, flips, small
colour changes). An image-level random split puts such copies in train and test. This module
finds them, so the split can keep each duplicate group on one side.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dctn

from .dataset import Manifest


def _dihedral(a: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(r[:, ::-1])
    return out


HASH_SIDE = 16  # a 16 x 16 low-frequency DCT block gives a 256-bit hash


def phash_bits(gray: np.ndarray, side: int = HASH_SIDE) -> np.ndarray:
    """Perceptual hash: sign of the low side x side DCT block against its median (DC excluded), packed to bytes."""
    coeffs = dctn(gray.astype(float), norm="ortho")[:side, :side].ravel()
    bits = coeffs > np.median(coeffs[1:])
    bits[0] = False
    return np.packbits(bits)


def image_hashes(path) -> np.ndarray:
    """Eight hashes (one per rotation and flip of the image), shape (8, 32) uint8."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=float)
    return np.stack([phash_bits(v) for v in _dihedral(gray)])


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit distance between packed hashes (last axis = bytes)."""
    return np.bitwise_count(np.bitwise_xor(a, b)).sum(axis=-1, dtype=np.int32)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def duplicate_groups(hashes: np.ndarray, max_distance: int = 16, block: int = 64) -> np.ndarray:
    """Group indices of near-duplicates. hashes: (n, 8, 32) uint8. Return one group number per image.

    Two images are linked when the original hash of one is within `max_distance` bits of any
    of the eight hashes of the other. Links are transitive (union-find).
    """
    n = hashes.shape[0]
    uf = _UnionFind(n)
    originals = hashes[:, 0]
    for start in range(0, n, block):
        a = originals[start:start + block][:, None, None, :]        # (b, 1, 1, bytes)
        dist = hamming(a, hashes[None, :, :, :]).min(axis=2)         # (b, n)
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows, cols):
            i = start + int(r)
            if i != c:
                uf.union(i, int(c))
    roots = np.array([uf.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def assign_groups(manifest: Manifest, max_distance: int = 16) -> tuple[Manifest, dict]:
    """Add `dup_group` and the final `split_group` column. Return the manifest and a summary.

    split_group = metadata group (slide or patient) when it exists. Duplicate groups that
    span several metadata groups merge them, so no copy of a cell crosses a split.
    """
    hashes = np.stack([image_hashes(p) for p in manifest.frame["path"]])
    dup = duplicate_groups(hashes, max_distance)
    frame = manifest.frame.copy()
    frame["dup_group"] = dup
    # exact byte copies always share a group
    for _, idx in frame.groupby("sha256").groups.items():
        idx = list(idx)
        if len(idx) > 1:
            frame.loc[idx, "dup_group"] = frame.loc[idx[0], "dup_group"]
    meta = frame["group"].fillna("")
    keys = np.where(meta != "", "meta:" + meta, "dup:" + frame["dup_group"].astype(str))
    uf = _UnionFind(len(frame))
    first_by_key: dict[str, int] = {}
    first_by_dup: dict[int, int] = {}
    for i, (k, d) in enumerate(zip(keys, frame["dup_group"])):
        if k in first_by_key:
            uf.union(i, first_by_key[k])
        else:
            first_by_key[k] = i
        if d in first_by_dup:
            uf.union(i, first_by_dup[d])
        else:
            first_by_dup[d] = i
    roots = np.array([uf.find(i) for i in range(len(frame))])
    _, frame["split_group"] = np.unique(roots, return_inverse=True)
    sizes = frame["dup_group"].value_counts()
    summary = {"images": int(len(frame)), "duplicate_groups": int(sizes.size),
               "images_in_duplicate_groups": int(sizes[sizes > 1].sum()),
               "largest_duplicate_group": int(sizes.max()), "split_groups": int(frame["split_group"].nunique()),
               "max_distance": max_distance}
    return Manifest(frame, manifest.warnings), summary
