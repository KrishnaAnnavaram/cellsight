"""The five cell classes. The order is the column order of every probability matrix."""

CLASSES: tuple[str, ...] = ("basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil")
CRITICAL_CLASS = "myeloblast"  # the leukemia-relevant class; its recall is reported on its own
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
