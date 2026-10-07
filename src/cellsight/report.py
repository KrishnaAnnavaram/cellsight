"""Evaluation record (JSON) and report (Markdown) for one ensemble run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import NOTICE, __version__, metrics
from .classes import CLASSES, CRITICAL_CLASS
from .config import Settings
from .dataset import Manifest
from .ensemble import EnsembleResult, mean_proba


def evaluate(result: EnsembleResult, manifest: Manifest, settings: Settings) -> dict:
    y = manifest.y
    groups = manifest.frame["split_group"].to_numpy()
    y_dev, y_test, g_test = y[result.dev_index], y[result.test_index], groups[result.test_index]
    oof_rows = {n: metrics.point_metrics(y_dev, p) for n, p in result.oof.items()}
    oof_rows["mean"] = metrics.point_metrics(y_dev, mean_proba(result.oof, result.names))
    candidates = dict(result.test_proba)
    candidates["mean"] = mean_proba(result.test_proba, result.names)
    candidates["stack"] = result.stacker.predict_proba(result.test_proba)
    test_rows = {}
    for name, p in candidates.items():
        test_rows[name] = {**metrics.point_metrics(y_test, p),
                           "ci": metrics.bootstrap(y_test, p, settings.n_bootstrap, settings.seed, groups=g_test)}
    stack = candidates["stack"]
    return {
        "development_images": int(len(result.dev_index)), "test_images": int(len(result.test_index)),
        "test_groups": int(len(np.unique(g_test))),
        "oof_development": oof_rows, "test": test_rows,
        "stack_per_class": metrics.per_class(y_test, stack),
        "stack_confusion": metrics.confusion(y_test, stack),
    }


def _ci(c: dict) -> str:
    return f"{c['value']:.3f} [{c['low']:.3f}, {c['high']:.3f}]"


def render(record: dict, dedupe_summary: dict, settings: Settings, source: str) -> str:
    lines = ["# cellsight evaluation report", "", f"> {NOTICE}", "",
             f"- Images: `{source}`, {dedupe_summary['images']} files",
             f"- Near-duplicates: {dedupe_summary['images_in_duplicate_groups']} images in groups of 2 or more "
             f"(Hamming distance <= {dedupe_summary['max_distance']}, rotations and flips included)",
             f"- Split by: {dedupe_summary.get('split_by', 'group')}. Split groups: {dedupe_summary['split_groups']}. Development {record['development_images']} images, "
             f"test {record['test_images']} images in {record['test_groups']} groups",
             f"- {settings.folds}-fold group-aware out-of-fold stacking, seed {settings.seed}, cellsight {__version__}",
             "", "## Development (out-of-fold) metrics, used for all choices", "",
             "| Model | Accuracy | Macro-F1 | Myeloblast recall | AUC (OvR) | ECE |", "|---|---|---|---|---|---|"]
    for n, m in record["oof_development"].items():
        lines.append(f"| {n} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['critical_recall']:.3f} | "
                     f"{m['auc_ovr_macro']:.3f} | {m['ece']:.3f} |")
    lines += ["", "## Test metrics (one pass, 95% group bootstrap CI)", "",
              "| Model | Accuracy | Macro-F1 | Myeloblast recall | AUC (OvR) | ECE |", "|---|---|---|---|---|---|"]
    for n, m in record["test"].items():
        lines.append(f"| {n} | {_ci(m['ci']['accuracy'])} | {_ci(m['ci']['macro_f1'])} | "
                     f"{_ci(m['ci']['critical_recall'])} | {m['auc_ovr_macro']:.3f} | {m['ece']:.3f} |")
    lines += ["", "## Stack: per-class test results", "", "| Class | Support | Recall | F1 |", "|---|---|---|---|"]
    for r in record["stack_per_class"]:
        mark = " (critical)" if r["class"] == CRITICAL_CLASS else ""
        lines.append(f"| {r['class']}{mark} | {r['support']} | {r['recall']:.3f} | {r['f1']:.3f} |")
    lines += ["", "## Stack: test confusion matrix (rows = true class)", "",
              "| | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for c, row in zip(CLASSES, record["stack_confusion"]):
        lines.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    return "\n".join(lines)


def write(record: dict, dedupe_summary: dict, settings: Settings, source: str, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"report": out_dir / "report.md", "results": out_dir / "results.json"}
    paths["report"].write_text(render(record, dedupe_summary, settings, source), encoding="utf-8")
    paths["results"].write_text(json.dumps({"version": __version__, "notice": NOTICE, "dedupe": dedupe_summary,
                                            "settings": {k: str(v) for k, v in settings.__dict__.items()},
                                            **record}, indent=2), encoding="utf-8")
    return paths
