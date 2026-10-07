"""cellsight: leakage-safe blood-cell image classification with out-of-fold stacking.

The core package needs numpy, pandas, scipy, scikit-learn and Pillow. The deep backbones
(PyTorch and timm) are an optional extra and are imported lazily in `cellsight.deep`.
"""

__version__ = "0.1.0"

NOTICE = (
    "Research software. Not a medical device and not a diagnostic tool. A qualified person must "
    "review every result. Scores on one public dataset do not transfer to other labs, stains or microscopes."
)
