"""Base models behind one interface. Light models run on CPU with scikit-learn. Deep models live in `deep.py`.

Every base model sees only the rows that the caller gives to `fit`: never the test rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classes import CLASSES
from .features import feature_matrix, image_features


class BaseModel(Protocol):
    name: str

    def fit(self, paths: list, y: np.ndarray, val_paths: list | None = None, val_y: np.ndarray | None = None): ...

    def predict_proba(self, paths: list) -> np.ndarray: ...


class FeatureCache:
    """Decodes each image once per run and keeps only its small feature vector."""

    def __init__(self, size: int = 64):
        self.size = size
        self._store: dict[str, np.ndarray] = {}

    def matrix(self, paths) -> np.ndarray:
        missing = [str(p) for p in paths if str(p) not in self._store]
        if missing:
            for p, row in zip(missing, feature_matrix(missing, self.size)):
                self._store[p] = row
        return np.stack([self._store[str(p)] for p in paths])


def _full_proba(est, X) -> np.ndarray:
    """Probabilities with one column per class, also when a class was absent from the training rows."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    out[:, est.classes_] = p
    return out


@dataclass
class LightModel:
    name: str
    columns: str  # "all", "color" or "thumbnail"
    cache: FeatureCache
    seed: int = 0

    def _select(self, X: np.ndarray) -> np.ndarray:
        if self.columns == "color":
            return X[:, :36]  # hue, saturation and value histograms + dark-region shape
        if self.columns == "thumbnail":
            return X[:, -64:]
        return X

    def _estimator(self):
        if self.name == "hist_logreg":
            return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, C=0.5))])
        if self.name == "shape_rf":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
        if self.name == "thumb_knn":
            return Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))])
        raise ValueError(self.name)

    def fit(self, paths, y, val_paths=None, val_y=None):
        self.est_ = self._estimator().fit(self._select(self.cache.matrix(paths)), y)
        return self

    def predict_proba(self, paths) -> np.ndarray:
        return _full_proba(self.est_, self._select(self.cache.matrix(paths)))

    def predict_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Probabilities for decoded RGB arrays (used by the occlusion explainer)."""
        from PIL import Image
        rows = [image_features(np.asarray(Image.fromarray(im).resize((self.cache.size, self.cache.size),
                                                                     Image.Resampling.BILINEAR))) for im in images]
        return _full_proba(self.est_, self._select(np.stack(rows)))


LIGHT_MODELS = {"hist_logreg": "color", "shape_rf": "all", "thumb_knn": "thumbnail"}


def load_backbone_config() -> dict:
    text = resources.files("cellsight").joinpath("configs/backbones.json").read_text(encoding="utf-8")
    return json.loads(text)


def make_model(name: str, cache: FeatureCache, seed: int = 0, settings=None) -> BaseModel:
    if name in LIGHT_MODELS:
        return LightModel(name, LIGHT_MODELS[name], cache, seed)
    backbones = load_backbone_config()["backbones"]
    if name in backbones or name == "tiny_cnn":
        from .deep import TorchModel  # lazy: needs the 'deep' extra
        return TorchModel(name, seed=seed, device=getattr(settings, "device", "auto"))
    raise ValueError(f"unknown model {name!r}; light: {sorted(LIGHT_MODELS)}, deep: {sorted(backbones)}")
