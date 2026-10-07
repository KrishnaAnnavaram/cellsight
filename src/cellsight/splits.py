"""Group-aware, label-stratified splits. A split group never sits on two sides of a split."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class SplitError(ValueError):
    """The groups do not permit the requested split."""


def assert_disjoint(groups: np.ndarray, a, b) -> None:
    shared = set(np.asarray(groups)[np.asarray(a)]) & set(np.asarray(groups)[np.asarray(b)])
    if shared:
        raise SplitError(f"{len(shared)} groups are on both sides of a split, for example {sorted(shared)[:3]}")


def _splitter(n_splits: int, groups: np.ndarray, seed: int) -> StratifiedGroupKFold:
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise SplitError(f"{n_groups} groups cannot fill {n_splits} folds; add data or lower the fold count")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def holdout_split(y: np.ndarray, groups: np.ndarray, test_fraction: float = 0.2, seed: int = 42):
    """Return (development indices, test indices). The test indices stay untouched until the final evaluation."""
    n_splits = max(2, int(round(1 / test_fraction)))
    dev, test = next(_splitter(n_splits, groups, seed).split(np.zeros(len(y)), y, groups))
    assert_disjoint(groups, dev, test)
    return np.sort(dev), np.sort(test)


def kfold(y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = list(_splitter(k, groups, seed).split(np.zeros(len(y)), y, groups))
    for tr, va in folds:
        assert_disjoint(groups, tr, va)
    return folds
