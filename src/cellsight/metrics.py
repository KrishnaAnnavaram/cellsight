"""Multi-class metrics, per-class recall, calibration and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from .classes import CLASS_INDEX, CLASSES, CRITICAL_CLASS

LABELS = list(range(len(CLASSES)))


def top_label_ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def point_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)),
        "critical_recall": float(recall_score(y, pred, labels=[CLASS_INDEX[CRITICAL_CLASS]], average="macro",
                                              zero_division=0)),
        "ece": top_label_ece(y, proba),
        "log_loss": float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)))),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS))
    except ValueError:
        out["auc_ovr_macro"] = float("nan")
    return out


def per_class(y: np.ndarray, proba: np.ndarray) -> list[dict]:
    pred = proba.argmax(axis=1)
    recall = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=LABELS, average=None, zero_division=0)
    return [{"class": c, "support": int((y == i).sum()), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, c in enumerate(CLASSES)]


def confusion(y: np.ndarray, proba: np.ndarray) -> list[list[int]]:
    return confusion_matrix(y, proba.argmax(axis=1), labels=LABELS).tolist()


def _fast(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Accuracy, macro-F1 and critical-class recall from one confusion matrix (no sklearn overhead)."""
    k = len(CLASSES)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(tp.sum() / cm.sum()), float(f1.mean()), float(recall[CLASS_INDEX[CRITICAL_CLASS]])


def bootstrap(y: np.ndarray, proba: np.ndarray, n_boot: int = 1000, seed: int = 0,
              groups: np.ndarray | None = None) -> dict:
    """95% percentile CIs for accuracy, macro-F1 and critical-class recall.

    With `groups`, whole groups are resampled (a cluster bootstrap), because images of one
    group are not independent.
    """
    rng = np.random.default_rng(seed)
    pred = proba.argmax(axis=1)
    if groups is None:
        units = [np.array([i]) for i in range(len(y))]
    else:
        units = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        stats[b] = _fast(y[idx], pred[idx])
    point = _fast(y, pred)
    names = ("accuracy", "macro_f1", "critical_recall")
    return {n: {"value": point[i], "low": float(np.quantile(stats[:, i], 0.025)),
                "high": float(np.quantile(stats[:, i], 0.975))} for i, n in enumerate(names)}
