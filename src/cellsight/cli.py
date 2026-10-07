"""Command line interface: `cellsight <command>`."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import NOTICE, __version__, ensemble, explain, report, synthetic
from .classes import CLASS_INDEX, CLASSES
from .config import ConfigError, Settings
from .dataset import DatasetError, load_image, scan
from .dedupe import assign_groups
from .models import LIGHT_MODELS, load_backbone_config
from .splits import SplitError, holdout_split

DEFAULT_MODELS = "hist_logreg,shape_rf,thumb_knn"


def _settings(args) -> Settings:
    return Settings.from_env().with_overrides(
        seed=getattr(args, "seed", None), folds=getattr(args, "folds", None),
        test_fraction=getattr(args, "test_fraction", None), hash_distance=getattr(args, "hash_distance", None),
        n_bootstrap=getattr(args, "bootstrap", None), device=getattr(args, "device", None))


def _manifest(args, settings: Settings):
    data_dir = args.data_dir or settings.data_dir
    if data_dir is None:
        raise ConfigError("give --data-dir or set CELLSIGHT_DATA_DIR (or run `cellsight demo`)")
    meta = None if getattr(args, "no_metadata", False) else (args.metadata or settings.metadata_csv)
    if meta is None and not getattr(args, "no_metadata", False) and (Path(data_dir) / "metadata.csv").is_file():
        meta = Path(data_dir) / "metadata.csv"
    man = scan(data_dir, meta)
    for w in man.warnings:
        print(f"[cellsight] warning: {w}", file=sys.stderr)
    return man, str(data_dir)


def cmd_synth(args) -> int:
    meta = synthetic.generate(args.out, args.per_class, args.duplicates, args.size, args.slides, args.seed)
    print(f"wrote {len(meta)} synthetic images ({meta['origin'].nunique()} originals) to {args.out}")
    return 0


def cmd_index(args) -> int:
    settings = _settings(args)
    man, _ = _manifest(args, settings)
    print(man.frame["label"].value_counts().reindex(list(CLASSES)).to_string())
    print(f"images: {len(man)}  index: OK")
    return 0


def cmd_dedupe(args) -> int:
    settings = _settings(args)
    man, _ = _manifest(args, settings)
    man, summary = assign_groups(man, settings.hash_distance)
    for k, v in summary.items():
        print(f"{k}: {v}")
    if args.out:
        man.frame.drop(columns=["path"]).to_csv(args.out, index=False)
        print(f"wrote groups to {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    settings = _settings(args)
    man, source = _manifest(args, settings)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    man, summary = assign_groups(man, settings.hash_distance)
    if args.split_by == "image":
        man.frame["split_group"] = np.arange(len(man))  # leakage check only: copies can cross the split
        summary["split_groups"] = len(man)
        summary["split_by"] = "image (leakage check: copies can cross the split)"
        print("[cellsight] WARNING: --split-by image puts near-duplicates on both sides; use it only "
              "to measure the leakage effect", file=sys.stderr)
    else:
        summary["split_by"] = "group"
    print(f"[cellsight] {summary['images_in_duplicate_groups']} images are near-duplicates; "
          f"{summary['split_groups']} split groups", file=sys.stderr)
    result = ensemble.run(man, names, settings)
    record = report.evaluate(result, man, settings)
    out = Path(args.out) if args.out else settings.output_dir
    paths = report.write(record, summary, settings, source, out)
    print(f"{'model':<12}{'test acc [95% CI]':<28}{'macro-F1':<10}{'myeloblast recall':<18}")
    for n, m in record["test"].items():
        ci = m["ci"]["accuracy"]
        print(f"{n:<12}{m['accuracy']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]{'':<6}{m['macro_f1']:<10.3f}"
              f"{m['critical_recall']:<18.3f}")
    print(f"wrote {paths['report']}")
    if args.save_model:
        if any(n not in LIGHT_MODELS for n in names):
            print("[cellsight] --save-model stores light models only; use `train-deep` for backbones", file=sys.stderr)
        else:
            path = Path(args.save_model)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(ensemble.make_bundle(result, settings), path)
            print(f"saved {path}")
    print(NOTICE)
    return 0


def cmd_train_deep(args) -> int:
    from .deep import TorchModel, TrainConfig  # needs the 'deep' extra
    settings = _settings(args)
    man, _ = _manifest(args, settings)
    man, _summary = assign_groups(man, settings.hash_distance)
    groups = man.frame["split_group"].to_numpy()
    dev, _test = holdout_split(man.y, groups, settings.test_fraction, settings.seed)  # test stays unused here
    tr_rel, va_rel = holdout_split(man.y[dev], groups[dev], 0.15, settings.seed)
    paths = np.asarray(man.frame["path"])[dev]
    tc = TrainConfig.from_file()
    if args.epochs:
        tc.max_epochs = args.epochs
    model = TorchModel(args.backbone, settings.seed, settings.device, pretrained=not args.no_pretrained,
                       train_config=tc)
    model.fit(list(paths[tr_rel]), man.y[dev][tr_rel], list(paths[va_rel]), man.y[dev][va_rel])
    for h in model.result_.history:
        print(f"epoch {h['epoch']:>2} {h['stage']:<6} loss {h['train_loss']:.4f} val macro-F1 {h['val_macro_f1']:.4f}")
    print(f"best epoch {model.result_.best_epoch}, validation macro-F1 {model.result_.best_score:.4f}")
    out = model.save(args.out or f"models/{args.backbone}.pt")
    print(f"saved {out} (the test images were not used)")
    return 0


def _load_bundle(path) -> ensemble.Bundle:
    obj = joblib.load(path)
    if not isinstance(obj, ensemble.Bundle):
        raise ConfigError(f"{path} is not a cellsight bundle")
    return obj


def cmd_predict(args) -> int:
    bundle = _load_bundle(args.model)
    proba = bundle.predict_paths(args.images)
    rows = [{"image": p, "prediction": CLASSES[int(np.argmax(r))], "confidence": float(np.max(r)),
             **{f"p_{c}": float(v) for c, v in zip(CLASSES, r)}} for p, r in zip(args.images, proba)]
    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print(table[["image", "prediction", "confidence"]].to_string(index=False))
    return 0


def cmd_explain(args) -> int:
    bundle = _load_bundle(args.model)
    img = load_image(args.image, bundle.image_size)
    proba = bundle.predict_images([img])[0]
    cls = CLASS_INDEX[args.target] if args.target else int(np.argmax(proba))
    heat = explain.occlusion_map(bundle.predict_images, img, cls, args.patch, args.stride)
    ratio = explain.focus_ratio(heat, explain.cell_mask(img))
    out = explain.save_overlay(img, heat, args.out)
    print(f"class {CLASSES[cls]} p={proba[cls]:.3f}  focus ratio {ratio:.2f} (>1: evidence on the cell)")
    print(f"wrote {out}")
    return 0


def cmd_demo(args) -> int:
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="cellsight_demo_"))
    data = work / "images"
    synthetic.generate(data, args.per_class, 2, 64, 40, args.seed or 42)
    print(f"[cellsight] synthetic images in {data}", file=sys.stderr)
    args.data_dir, args.metadata = str(data), None
    args.models = DEFAULT_MODELS
    args.split_by = "group"
    args.out = args.out or str(work / "report")
    args.save_model = str(work / "cellsight_light.joblib")
    return cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellsight", description="Leakage-safe blood-cell image classification.")
    parser.add_argument("--version", action="version", version=f"cellsight {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data-dir")
        p.add_argument("--metadata", help="CSV with columns file, group (slide, patient or source)")
        p.add_argument("--no-metadata", action="store_true", help="ignore metadata.csv: groups = duplicate groups")
        p.add_argument("--seed", type=int)
        p.add_argument("--hash-distance", type=int)
        p.add_argument("--test-fraction", type=float)
        p.add_argument("--folds", type=int)
        p.add_argument("--bootstrap", type=int)
        p.add_argument("--device")

    p = sub.add_parser("synth", help="write synthetic cell images with duplicates and slide groups")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--duplicates", type=int, default=2)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--slides", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("index", help="scan and validate an image folder")
    common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("dedupe", help="find near-duplicate groups (rotation and flip invariant)")
    common(p)
    p.add_argument("--out", help="CSV with dup_group and split_group per file")
    p.set_defaults(func=cmd_dedupe)

    p = sub.add_parser("evaluate", help="dedupe, group split, OOF stacking, one test pass, report")
    common(p)
    p.add_argument("--models", default=DEFAULT_MODELS,
                   help=f"comma list; light: {sorted(LIGHT_MODELS)}; deep: {sorted(load_backbone_config()['backbones'])}")
    p.add_argument("--split-by", choices=["group", "image"], default="group",
                   help="'image' only to measure duplicate leakage")
    p.add_argument("--out")
    p.add_argument("--save-model")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train-deep", help="fine-tune one backbone on development images (extra 'deep')")
    common(p)
    p.add_argument("--backbone", required=True, choices=sorted(load_backbone_config()["backbones"]) + ["tiny_cnn"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_train_deep)

    p = sub.add_parser("predict", help="classify images with a saved bundle")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p.add_argument("images", nargs="+")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("explain", help="occlusion map and cell-focus ratio for one image")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--target", choices=list(CLASSES))
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--out", default="outputs/explain.png")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("demo", help="offline demo on synthetic images with the light models")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--workdir")
    p.add_argument("--out")
    p.add_argument("--seed", type=int)
    p.add_argument("--folds", type=int)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--hash-distance", type=int)
    p.add_argument("--test-fraction", type=float)
    p.add_argument("--device")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, DatasetError, SplitError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
