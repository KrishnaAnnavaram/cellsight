import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cellsight.classes import CLASSES
from cellsight.config import ConfigError, Settings
from cellsight.dataset import DatasetError, scan
from cellsight.dedupe import assign_groups, hamming, image_hashes
from cellsight.features import image_features
from cellsight.models import FeatureCache
from cellsight.splits import SplitError, assert_disjoint, holdout_split, kfold


def test_settings():
    s = Settings.from_env({"CELLSIGHT_FOLDS": "3", "CELLSIGHT_HASH_DISTANCE": "10", "CELLSIGHT_DATA_DIR": "x"})
    assert s.folds == 3 and s.hash_distance == 10 and str(s.data_dir) == "x"
    for bad in ({"CELLSIGHT_FOLDS": "1"}, {"CELLSIGHT_TEST_FRACTION": "0.9"}, {"CELLSIGHT_SEED": "x"}):
        with pytest.raises(ConfigError):
            Settings.from_env(bad)


def test_scan_reads_all_classes_and_metadata(image_dir):
    man = scan(image_dir, image_dir / "metadata.csv")
    assert len(man) == 5 * 12 * 3
    assert set(man.frame["label"]) == set(CLASSES)
    assert man.frame["group"].str.startswith("slide_").all()


def test_scan_errors(tmp_path):
    with pytest.raises(DatasetError):
        scan(tmp_path / "missing")
    (tmp_path / "basophil").mkdir()
    (tmp_path / "basophil" / "bad.png").write_bytes(b"not an image")
    with pytest.raises(DatasetError) as err:
        scan(tmp_path)
    text = " ".join(err.value.errors)
    assert "missing class folders" in text and "cannot decode" in text


def test_byte_identical_copies_are_reported(tmp_path, image_dir):
    for c in CLASSES:
        (tmp_path / c).mkdir()
        src = sorted((image_dir / c).glob("*.png"))[0]
        (tmp_path / c / "a.png").write_bytes(src.read_bytes())
        (tmp_path / c / "b.png").write_bytes(src.read_bytes())
    man = scan(tmp_path)
    assert any("byte-identical" in w for w in man.warnings)


def test_hash_is_invariant_to_rotation_and_flip(image_dir):
    path = sorted((image_dir / "monocyte").glob("*_0.png"))[0]
    img = np.asarray(Image.open(path))
    rotated = path.parent.parent / "rot_tmp.png"
    Image.fromarray(np.rot90(img)[:, ::-1].copy()).save(rotated)
    a, b = image_hashes(path), image_hashes(rotated)
    assert hamming(a[0][None, :], b).min() <= 2
    rotated.unlink()


def test_duplicates_are_grouped_with_their_original(image_dir):
    # Problem 2: augmented copies of one cell must share one group.
    man = scan(image_dir, image_dir / "metadata.csv")
    man, summary = assign_groups(man, max_distance=16)
    meta = pd.read_csv(image_dir / "metadata.csv").set_index("file")
    frame = man.frame.assign(origin=man.frame["file"].map(meta["origin"]))
    assert (frame.groupby("origin")["dup_group"].nunique() == 1).all()
    # different cells stay apart in most cases
    assert summary["duplicate_groups"] >= 0.9 * frame["origin"].nunique()


def test_split_group_merges_duplicates_and_slides(image_dir):
    man, _ = assign_groups(scan(image_dir, image_dir / "metadata.csv"))
    f = man.frame
    assert (f.groupby("dup_group")["split_group"].nunique() == 1).all()
    assert (f.groupby("group")["split_group"].nunique() == 1).all()


def test_without_metadata_the_duplicate_groups_are_the_split_groups(image_dir):
    # Problem 3: with no slide IDs, the duplicate groups still keep copies together.
    man, summary = assign_groups(scan(image_dir), 16)
    assert summary["split_groups"] == summary["duplicate_groups"]


def test_group_splits_never_share_a_group(image_dir):
    man, _ = assign_groups(scan(image_dir))
    y, g = man.y, man.frame["split_group"].to_numpy()
    dev, test = holdout_split(y, g, 0.25, seed=0)
    assert set(g[dev]).isdisjoint(g[test]) and len(dev) + len(test) == len(y)
    for tr, va in kfold(y[dev], g[dev], 3, seed=0):
        assert set(g[dev][tr]).isdisjoint(g[dev][va])


def test_split_errors():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(SplitError):
        kfold(y, np.array([0, 0, 1, 1]), k=3)
    with pytest.raises(SplitError):
        assert_disjoint(np.array([0, 0, 1]), [0], [1])


def test_features_are_fixed_length_and_deterministic(image_dir):
    path = sorted((image_dir / "basophil").glob("*.png"))[0]
    img = np.asarray(Image.open(path).convert("RGB"))
    a, b = image_features(img), image_features(img)
    assert a.shape == (108,) and np.array_equal(a, b)
    assert image_features(np.zeros((40, 48, 3), dtype=np.uint8)).shape == (108,)


def test_feature_cache_keeps_vectors_not_images(image_dir):
    # Problem 8: images are decoded one at a time and only small vectors stay in memory.
    paths = sorted(str(p) for p in (image_dir / "erythroblast").glob("*.png"))[:5]
    cache = FeatureCache(64)
    X = cache.matrix(paths)
    assert X.shape == (5, 108)
    assert all(v.nbytes == 108 * 4 for v in cache._store.values())
