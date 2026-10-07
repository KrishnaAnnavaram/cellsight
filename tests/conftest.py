import pytest

from cellsight import synthetic


@pytest.fixture(scope="session")
def image_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("cells")
    synthetic.generate(root, per_class=12, duplicates=2, size=64, n_slides=12, seed=3)
    return root
