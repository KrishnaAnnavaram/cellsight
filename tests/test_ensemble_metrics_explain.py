import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, recall_score

from cellsight import ensemble, explain, metrics
from cellsight.classes import CLASS_INDEX, CLASSES
from cellsight.config import Settings
from cellsight.dataset import scan
from cellsight.dedupe import assign_groups
from cellsight.models import FeatureCache, make_model


@pytest.fixture(scope="module")
def grouped(image_dir):
    man, _ = assign_groups(scan(image_dir, image_dir / "metadata.csv"))
    return man


def test_stacker_never_sees_test_images_or_labels(grouped, monkeypatch):
    # Problem 1: the earlier stacker was fit and scored on the test set.
    settings = Settings(folds=3, test_fraction=0.25, seed=1)
    cache = FeatureCache(64)
    fitted_paths: list[str] = []

    def factory(name):
        model = make_model(name, cache, 0)
        original = model.fit

        def spy_fit(paths, y, *a, **k):
            fitted_paths.extend(paths)
            return original(paths, y, *a, **k)

        model.fit = spy_fit
        return model

    stacker_rows = []
    original_fit = ensemble.Stacker.fit

    def spy_stacker(self, oof, y):
        stacker_rows.append(len(y))
        return original_fit(self, oof, y)

    monkeypatch.setattr(ensemble.Stacker, "fit", spy_stacker)
    result = ensemble.run(grouped, ["hist_logreg", "shape_rf"], settings, factory=factory)
    test_paths = set(grouped.frame["path"].iloc[result.test_index])
    assert fitted_paths and test_paths.isdisjoint(fitted_paths)
    assert stacker_rows == [len(result.dev_index)]
    for p in result.oof.values():
        assert p.shape == (len(result.dev_index), len(CLASSES)) and np.isfinite(p).all()
    probs = result.stacker.predict_proba(result.test_proba)
    assert probs.shape == (len(result.test_index), len(CLASSES))
    assert np.allclose(probs.sum(axis=1), 1)


def test_run_requires_groups(image_dir):
    with pytest.raises(ValueError, match="dedupe"):
        ensemble.run(scan(image_dir), ["hist_logreg"], Settings(folds=2))


def test_bundle_predicts_paths_and_images(grouped):
    settings = Settings(folds=2, test_fraction=0.25)
    result = ensemble.run(grouped, ["hist_logreg"], settings)
    bundle = ensemble.make_bundle(result, settings)
    path = grouped.frame["path"].iloc[0]
    from cellsight.dataset import load_image
    a = bundle.predict_paths([path])
    b = bundle.predict_images([load_image(path, 64)])
    assert np.allclose(a, b, atol=1e-6)


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 5, 300))
    acc, f1, rec = metrics._fast(y, pred)
    assert acc == pytest.approx(accuracy_score(y, pred))
    assert f1 == pytest.approx(f1_score(y, pred, average="macro"))
    assert rec == pytest.approx(recall_score(y, pred, labels=[CLASS_INDEX["myeloblast"]], average="macro"))


def test_point_metrics_and_ece():
    y = np.repeat(np.arange(5), 10)
    perfect = np.eye(5)[y]
    m = metrics.point_metrics(y, perfect)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["critical_recall"] == 1 and m["ece"] == 0
    uniform = np.full((50, 5), 0.2)
    assert metrics.top_label_ece(y, uniform) == pytest.approx(0.0, abs=1e-9)  # 20% confident, 20% right


def test_group_bootstrap_is_wider_than_image_bootstrap():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 10)
    y = rng.integers(0, 5, 200)
    group_ok = rng.random(20) < 0.7
    pred = np.where(np.repeat(group_ok, 10), y, (y + 1) % 5)
    proba = np.eye(5)[pred]
    plain = metrics.bootstrap(y, proba, 300, 0)["accuracy"]
    grouped = metrics.bootstrap(y, proba, 300, 0, groups=groups)["accuracy"]
    assert grouped["high"] - grouped["low"] > plain["high"] - plain["low"]


def test_per_class_and_confusion():
    y = np.array([0, 1, 2, 3, 4, 3])
    proba = np.eye(5)[np.array([0, 1, 2, 3, 4, 2])]
    rows = metrics.per_class(y, proba)
    assert rows[3]["recall"] == 0.5 and rows[3]["class"] == "myeloblast"
    assert metrics.confusion(y, proba)[3][2] == 1


def _centre_predictor(images):
    """A fake model: class 0 probability depends only on the dark centre of the image."""
    out = []
    for im in images:
        h, w = im.shape[:2]
        centre = im[h // 2 - 6:h // 2 + 6, w // 2 - 6:w // 2 + 6].mean() / 255.0
        p0 = 1 - centre
        out.append([p0] + [(1 - p0) / 4] * 4)
    return np.array(out)


def _disk_image():
    img = np.full((48, 48, 3), 230, dtype=np.uint8)
    yy, xx = np.mgrid[0:48, 0:48]
    img[(yy - 24) ** 2 + (xx - 24) ** 2 <= 100] = [80, 40, 120]
    return img


def test_occlusion_finds_the_evidence_on_the_cell():
    # Problem 10: an explanation that shows where the evidence lies.
    img = _disk_image()
    heat = explain.occlusion_map(_centre_predictor, img, 0, patch=6, stride=3)
    assert heat.shape == (48, 48)
    assert heat[18:30, 18:30].mean() > heat[:8, :8].mean()
    mask = explain.cell_mask(img)
    assert mask[24, 24] and not mask[2, 2]
    assert explain.focus_ratio(heat, mask) > 1.5


def test_focus_ratio_edge_cases(tmp_path):
    assert np.isnan(explain.focus_ratio(np.zeros((4, 4)), np.ones((4, 4), bool)))
    path = explain.save_overlay(_disk_image(), np.random.default_rng(0).random((48, 48)), tmp_path / "x.png")
    assert path.exists()
