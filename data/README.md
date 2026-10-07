# Data

Git does not store images in this repository. Put the images in `data/` (git ignores everything here
except this file), or point `CELLSIGHT_DATA_DIR` at the folder.

## Source

| Item | Value |
|---|---|
| Name | Blood Cell Images for Cancer Detection |
| URL | https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection |
| License | Read the license and the terms on the dataset page before use. Do not redistribute the images |
| Size | 5 classes, 1,000 images per class (5,000 images) |
| Known issue | The set seems to contain augmented copies of a smaller pool of original cells. Run `cellsight dedupe` before you trust any split |
| Metadata | No slide, patient or source IDs are published |

## Download

1. Sign in to Kaggle and accept the dataset terms.
2. Download the archive and extract it to `data/blood_cells/`.
3. Check that the folder names match the class names below.
4. Run `cellsight index --data-dir data/blood_cells`.

With the Kaggle CLI: `kaggle datasets download -d sumithsingh/blood-cell-images-for-cancer-detection -p data --unzip`.

## Expected layout

```
data/blood_cells/
  basophil/        *.png | *.jpg | *.jpeg | *.bmp | *.tif
  erythroblast/
  monocyte/
  myeloblast/      the leukemia-relevant class
  seg_neutrophil/
  metadata.csv     optional: columns file, group (slide, patient or source)
```

| Rule | Result |
|---|---|
| A class folder is missing or empty | Error |
| A file cannot be decoded | Error |
| A folder that is not a class | Warning, ignored |
| Byte-identical files | Warning, the files share one group |
| An image without a group in `metadata.csv` | Warning, the duplicate group is used |

If you have slide or patient IDs, write them to `metadata.csv`. The split then keeps each slide on one side.

## Offline data

The tests and the demo use synthetic images. They need no download:

```bash
cellsight synth --out data/synthetic --per-class 40 --duplicates 2 --slides 40
```
