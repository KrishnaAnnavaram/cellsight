"""Out-of-fold stacking. The meta-model learns only from out-of-fold predictions on development images.

Procedure:
1. Hold out a group-aware test set. Nothing in steps 2-4 receives a test path or a test label.
2. Split the development images into k group-aware folds. For each fold, fit every base model
   on the other folds and predict the held-out fold: the out-of-fold (OOF) probabilities.
3. Fit the meta-model (multinomial logistic regression on log-probabilities) on the OOF matrix.
4. Refit every base model on all development images.
5. Predict the test images once. Score the base models, the probability mean and the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .classes import CLASSES
from .config import Settings
from .dataset import Manifest
from .models import LIGHT_MODELS, BaseModel, FeatureCache, make_model
from .splits import holdout_split, kfold


def stack_features(probas: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.hstack([np.log(np.clip(probas[n], 1e-6, 1.0)) for n in names])


class Stacker:
    def __init__(self, names: list[str], C: float = 1.0, seed: int = 0):
        self.names = list(names)
        self.model = LogisticRegression(C=C, max_iter=5000, random_state=seed)

    def fit(self, oof: dict[str, np.ndarray], y: np.ndarray) -> "Stacker":
        self.model.fit(stack_features(oof, self.names), y)
        return self

    def predict_proba(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        p = self.model.predict_proba(stack_features(probas, self.names))
        out = np.zeros((p.shape[0], len(CLASSES)))
        out[:, self.model.classes_] = p
        return out


def mean_proba(probas: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.mean([probas[n] for n in names], axis=0)


@dataclass
class EnsembleResult:
    names: list[str]
    dev_index: np.ndarray
    test_index: np.ndarray
    oof: dict[str, np.ndarray]
    test_proba: dict[str, np.ndarray]
    stacker: Stacker
    base_models: dict[str, BaseModel] = field(default_factory=dict)


ModelFactory = Callable[[str], BaseModel]


def _inner_val(y: np.ndarray, groups: np.ndarray, seed: int):
    """A group-aware validation part for early stopping of deep models."""
    return holdout_split(y, groups, test_fraction=0.15, seed=seed)


def _fit(model: BaseModel, paths: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> BaseModel:
    if getattr(model, "name", "") in LIGHT_MODELS:
        return model.fit(list(paths), y)  # light models do not use early stopping: train on all rows
    tr, va = _inner_val(y, groups, seed)
    return model.fit(list(paths[tr]), y[tr], list(paths[va]), y[va])


def out_of_fold(paths: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], factory: ModelFactory,
                folds: int, seed: int) -> dict[str, np.ndarray]:
    oof = {n: np.full((len(y), len(CLASSES)), np.nan) for n in names}
    for k, (tr, va) in enumerate(kfold(y, groups, folds, seed)):
        for n in names:
            model = _fit(factory(n), paths[tr], y[tr], groups[tr], seed + k)
            oof[n][va] = model.predict_proba(list(paths[va]))
    for n in names:
        if np.isnan(oof[n]).any():
            raise RuntimeError(f"{n}: some development images have no out-of-fold prediction")
    return oof


def run(manifest: Manifest, names: list[str], settings: Settings,
        factory: ModelFactory | None = None) -> EnsembleResult:
    if "split_group" not in manifest.frame:
        raise ValueError("run dedupe.assign_groups first: the manifest has no split_group column")
    cache = FeatureCache(settings.image_size)
    factory = factory or (lambda n: make_model(n, cache, settings.seed, settings))
    paths = np.asarray(manifest.frame["path"])
    y = manifest.y
    groups = manifest.frame["split_group"].to_numpy()
    dev, test = holdout_split(y, groups, settings.test_fraction, settings.seed)

    oof = out_of_fold(paths[dev], y[dev], groups[dev], names, factory, settings.folds, settings.seed)
    stacker = Stacker(names, seed=settings.seed).fit(oof, y[dev])

    base_models, test_proba = {}, {}
    for n in names:
        base_models[n] = _fit(factory(n), paths[dev], y[dev], groups[dev], settings.seed)
    for n in names:  # the single pass over the test images
        test_proba[n] = base_models[n].predict_proba(list(paths[test]))
    return EnsembleResult(list(names), dev, test, oof, test_proba, stacker, base_models)


@dataclass
class Bundle:
    """The final model: base models fit on all development images plus the OOF-trained stacker."""

    names: list[str]
    base_models: dict[str, BaseModel]
    stacker: Stacker
    image_size: int
    version: str

    def predict_paths(self, paths) -> np.ndarray:
        return self.stacker.predict_proba({n: self.base_models[n].predict_proba(list(paths)) for n in self.names})

    def predict_images(self, images: list[np.ndarray]) -> np.ndarray:
        return self.stacker.predict_proba({n: self.base_models[n].predict_images(images) for n in self.names})


def make_bundle(result: EnsembleResult, settings: Settings) -> Bundle:
    from . import __version__
    for m in result.base_models.values():
        if hasattr(m, "cache"):
            m.cache = FeatureCache(m.cache.size)  # drop cached development features before saving
    return Bundle(result.names, result.base_models, result.stacker, settings.image_size, __version__)
