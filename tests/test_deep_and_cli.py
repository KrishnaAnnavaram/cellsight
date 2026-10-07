import json

import numpy as np
import pytest

from cellsight.cli import main
from cellsight.deep import DataConfig, TrainConfig, preprocess
from cellsight.models import load_backbone_config


def test_seven_backbones_are_configured():
    cfg = load_backbone_config()
    assert len(cfg["backbones"]) == 7
    assert cfg["training"]["selection_metric"] == "macro_f1"


def test_preprocess_uses_the_backbone_normalisation():
    # Problem 4: every backbone gets its own mean and std (the earlier CNNs got none).
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    imagenet = DataConfig(32, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    x = preprocess(img, imagenet, train=False)
    assert x.shape == (3, 32, 32) and x.dtype == np.float32
    expected = (128 / 255 - 0.485) / 0.229
    assert x[0, 0, 0] == pytest.approx(expected, abs=1e-4)


def test_training_augmentation_changes_images_and_eval_does_not():
    # Problem 7: augmentation exists for training only.
    rng_img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    cfg = DataConfig(32, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    a = preprocess(rng_img, cfg, train=False)
    b = preprocess(rng_img, cfg, train=False)
    assert np.array_equal(a, b)
    augmented = [preprocess(rng_img, cfg, train=True, rng=np.random.default_rng(s)) for s in range(4)]
    assert any(not np.allclose(a, t) for t in augmented)


def test_train_config_has_one_budget_for_all_backbones():
    # Problem 5: one epoch cap and early stopping for every backbone.
    tc = TrainConfig.from_file()
    assert tc.max_epochs == 20 and tc.patience == 4 and tc.freeze_epochs == 3


def test_frozen_stage_keeps_batchnorm_statistics():
    # Problem 9: in the frozen stage the BatchNorm running statistics must not move.
    torch = pytest.importorskip("torch")
    from cellsight.deep import build_network, set_stage
    model, _ = build_network("tiny_cnn")
    bn = model.features[1]
    before = bn.running_mean.clone()
    set_stage(model, frozen=True)
    model(torch.randn(8, 3, 64, 64) * 3 + 1)
    assert torch.equal(before, bn.running_mean)
    assert not any(p.requires_grad for p in model.features.parameters())
    set_stage(model, frozen=False)
    model(torch.randn(8, 3, 64, 64) * 3 + 1)
    assert not torch.equal(before, bn.running_mean)


def test_tiny_cnn_training_restores_the_best_epoch(image_dir):
    torch = pytest.importorskip("torch")
    from cellsight.dataset import scan
    from cellsight.dataset import load_image
    from cellsight.deep import TorchModel, grad_cam
    man = scan(image_dir)
    paths, y = list(man.frame["path"]), man.y
    tc = TrainConfig(max_epochs=3, patience=1, batch_size=32, freeze_epochs=1, amp=False)
    m = TorchModel("tiny_cnn", seed=0, device="cpu", pretrained=False, train_config=tc)
    m.fit(paths[::2], y[::2], paths[1::2], y[1::2])
    hist = m.result_.history
    assert hist[0]["stage"] == "frozen" and m.result_.best_score == max(h["val_macro_f1"] for h in hist)
    proba = m.predict_proba(paths[:4])
    assert proba.shape == (4, 5) and np.allclose(proba.sum(1), 1, atol=1e-5)
    cam = grad_cam(m.model_, m.cfg_, load_image(paths[0]), m.model_.features[4])
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1
    m2 = TorchModel("tiny_cnn", seed=0, device="cpu", pretrained=False, train_config=tc)
    m2.fit(paths[::2], y[::2], paths[1::2], y[1::2])
    assert np.allclose(m2.predict_proba(paths[:4]), proba, atol=1e-5)  # seeded runs repeat


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "imgs"
    assert main(["synth", "--out", str(data), "--per-class", "8", "--slides", "10"]) == 0
    assert main(["index", "--data-dir", str(data)]) == 0
    assert main(["dedupe", "--data-dir", str(data), "--out", str(tmp_path / "groups.csv")]) == 0
    out = tmp_path / "rep"
    bundle = tmp_path / "m.joblib"
    assert main(["evaluate", "--data-dir", str(data), "--folds", "2", "--bootstrap", "50", "--models",
                 "hist_logreg,shape_rf", "--out", str(out), "--save-model", str(bundle)]) == 0
    record = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {"hist_logreg", "shape_rf", "mean", "stack"} <= set(record["test"])
    assert "Not a medical device" in (out / "report.md").read_text(encoding="utf-8")
    img = next((data / "myeloblast").glob("*.png"))
    assert main(["predict", "--model", str(bundle), str(img)]) == 0
    assert main(["explain", "--model", str(bundle), "--image", str(img), "--out", str(tmp_path / "e.png")]) == 0
    assert (tmp_path / "e.png").exists()


def test_cli_image_split_is_a_labelled_leakage_check(tmp_path, capsys):
    data = tmp_path / "imgs"
    main(["synth", "--out", str(data), "--per-class", "8", "--slides", "10"])
    out = tmp_path / "rep"
    assert main(["evaluate", "--data-dir", str(data), "--no-metadata", "--split-by", "image", "--folds", "2",
                 "--bootstrap", "50", "--models", "hist_logreg", "--out", str(out)]) == 0
    assert "leakage check" in (out / "report.md").read_text(encoding="utf-8")
    assert "WARNING" in capsys.readouterr().err


def test_cli_errors(tmp_path, capsys):
    assert main(["index", "--data-dir", str(tmp_path / "none")]) == 2
    import joblib
    joblib.dump({"x": 1}, tmp_path / "bad.joblib")
    assert main(["predict", "--model", str(tmp_path / "bad.joblib"), "x.png"]) == 2
