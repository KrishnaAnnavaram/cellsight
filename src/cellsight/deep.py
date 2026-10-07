"""Deep backbones (optional extra: pip install 'cellsight[deep]'). torch and timm are imported lazily.

Fixes against the earlier notebook:
* Each backbone gets its own input size and normalisation from its timm data config.
* Training images get flips, 90-degree rotations, brightness/contrast and stain jitter.
* All backbones share one epoch cap. Early stopping on validation macro-F1 restores the best epoch.
* In the frozen stage the backbone runs in eval mode, so BatchNorm statistics do not move.
* Seeds are set for Python, NumPy and torch. Images are decoded lazily, one batch at a time.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .classes import CLASSES
from .models import load_backbone_config


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError("deep backbones need the optional extra: pip install 'cellsight[deep]'") from exc
    return torch


@dataclass(frozen=True)
class DataConfig:
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass
class TrainConfig:
    max_epochs: int = 20
    patience: int = 4
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_full: float = 1e-4
    weight_decay: float = 1e-4
    freeze_epochs: int = 3
    amp: bool = True

    @classmethod
    def from_file(cls) -> "TrainConfig":
        cfg = load_backbone_config()["training"]
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


@dataclass
class TrainResult:
    best_epoch: int
    best_score: float
    history: list[dict] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _tiny_cnn(n_classes: int):
    torch = require_torch()
    nn = torch.nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, n_classes))

        def forward(self, x):
            return self.head(self.features(x))

        def get_classifier(self):
            return self.head

    return TinyCNN()


def build_network(name: str, n_classes: int = len(CLASSES), pretrained: bool = True):
    """Return (model, DataConfig). tiny_cnn needs only torch. The 7 backbones need timm."""
    if name == "tiny_cnn":
        return _tiny_cnn(n_classes), DataConfig(64, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    spec = load_backbone_config()["backbones"].get(name)
    if spec is None:
        raise ValueError(f"unknown backbone {name!r}")
    try:
        import timm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the timm backbones need: pip install 'cellsight[deep]'") from exc
    model = timm.create_model(spec["timm_name"], pretrained=pretrained, num_classes=n_classes, drop_rate=0.3)
    dc = timm.data.resolve_model_data_config(model)
    return model, DataConfig(int(dc["input_size"][-1]), tuple(dc["mean"]), tuple(dc["std"]))


def preprocess(img: np.ndarray, cfg: DataConfig, train: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    """uint8 HxWx3 -> float32 CxHxW, resized and normalised with the backbone's own mean and std."""
    im = Image.fromarray(img).resize((cfg.input_size, cfg.input_size), Image.Resampling.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    if train:
        rng = rng or np.random.default_rng()
        x = np.rot90(x, int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            x = x[:, ::-1]
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)          # brightness and contrast
        x = x * rng.uniform(0.93, 1.07, size=(1, 1, 3))                    # stain (per-channel) jitter
        x = np.clip(x, 0.0, 1.0)
    x = (x - np.asarray(cfg.mean, dtype=np.float32)) / np.asarray(cfg.std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def _batches(paths, labels, cfg, train, batch_size, rng):
    torch = require_torch()
    from .dataset import load_image
    order = rng.permutation(len(paths)) if train else np.arange(len(paths))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        xb = np.stack([preprocess(load_image(paths[i]), cfg, train, rng) for i in idx])
        yb = None if labels is None else torch.as_tensor(np.asarray(labels)[idx], dtype=torch.long)
        yield torch.from_numpy(xb), yb


def _device(device: str):
    torch = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_stage(model, frozen: bool) -> None:
    """Frozen stage: backbone parameters fixed and backbone in eval mode (BatchNorm stats fixed); head trains."""
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = (not frozen) or id(p) in head_ids
    if frozen:
        model.eval()
        head.train()
    else:
        model.train()


def predict_network(model, cfg: DataConfig, paths, device, batch_size: int = 64) -> np.ndarray:
    torch = require_torch()
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in _batches(list(paths), None, cfg, False, batch_size, np.random.default_rng(0)):
            out.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def train_network(model, cfg: DataConfig, train_paths, train_y, val_paths, val_y, tc: TrainConfig,
                  seed: int = 0, device: str = "auto") -> TrainResult:
    """Two-stage fine-tuning with early stopping on validation macro-F1. Restores the best weights."""
    torch = require_torch()
    from sklearn.metrics import f1_score
    seed_everything(seed)
    dev = _device(device)
    model.to(dev)
    rng = np.random.default_rng(seed)
    use_amp = tc.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = TrainResult(best_epoch=-1, best_score=-1.0)
    best_state = None
    stale = 0
    optimizer = None
    for epoch in range(tc.max_epochs):
        frozen = epoch < tc.freeze_epochs
        if epoch == 0 or epoch == tc.freeze_epochs:
            set_stage(model, frozen)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=tc.lr_head if frozen else tc.lr_full,
                                          weight_decay=tc.weight_decay)
        set_stage(model, frozen)
        losses = []
        for xb, yb in _batches(list(train_paths), train_y, cfg, True, tc.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = loss_fn(model(xb.to(dev)), yb.to(dev))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        proba = predict_network(model, cfg, val_paths, dev)
        score = float(f1_score(val_y, proba.argmax(1), average="macro", labels=list(range(len(CLASSES))),
                               zero_division=0))
        best.history.append({"epoch": epoch, "stage": "frozen" if frozen else "full",
                             "train_loss": float(np.mean(losses)) if losses else float("nan"), "val_macro_f1": score})
        if score > best.best_score:
            best.best_score, best.best_epoch = score, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= tc.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


class TorchModel:
    """BaseModel adapter: fit/predict_proba on image paths."""

    def __init__(self, name: str, seed: int = 0, device: str = "auto", pretrained: bool = True,
                 train_config: TrainConfig | None = None):
        self.name = name
        self.seed = seed
        self.device = device
        self.pretrained = pretrained
        self.train_config = train_config or TrainConfig.from_file()

    def fit(self, paths, y, val_paths=None, val_y=None):
        if val_paths is None:
            raise ValueError("deep models need a validation set for early stopping (pass val_paths)")
        seed_everything(self.seed)  # before the head (and tiny_cnn) weights are initialised
        self.model_, self.cfg_ = build_network(self.name, pretrained=self.pretrained)
        self.result_ = train_network(self.model_, self.cfg_, list(paths), np.asarray(y), list(val_paths),
                                     np.asarray(val_y), self.train_config, self.seed, self.device)
        return self

    def predict_proba(self, paths) -> np.ndarray:
        return predict_network(self.model_, self.cfg_, list(paths), _device(self.device))

    def predict_images(self, images: list[np.ndarray]) -> np.ndarray:
        torch = require_torch()
        dev = _device(self.device)
        self.model_.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.stack([preprocess(im, self.cfg_, False) for im in images])).to(dev)
            return torch.softmax(self.model_(x), dim=1).cpu().numpy()

    def save(self, path: str | Path) -> Path:
        torch = require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "state_dict": self.model_.state_dict(), "data_config": self.cfg_.__dict__,
                    "classes": list(CLASSES), "history": self.result_.history}, path)
        return path


def grad_cam(model, cfg: DataConfig, img: np.ndarray, target_layer, class_index: int | None = None,
             channels_last: bool = False) -> np.ndarray:
    """Grad-CAM heat map (input_size x input_size, values in [0, 1]) for one image."""
    torch = require_torch()
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        x = torch.from_numpy(preprocess(img, cfg, False)[None]).requires_grad_(True)
        logits = model(x)
        idx = int(logits.argmax(1)) if class_index is None else class_index
        model.zero_grad()
        logits[0, idx].backward()
        a, g = acts["a"], grads["g"]
        if a.dim() == 4 and channels_last:  # Swin stages give (B, H, W, C)
            a, g = a.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        if a.dim() == 3:  # transformer tokens (B, N, C): drop the class token, make a square grid
            n = a.shape[1]
            side = int(np.sqrt(n))
            a, g = a[:, n - side * side:], g[:, n - side * side:]
            a = a.transpose(1, 2).reshape(1, -1, side, side)
            g = g.transpose(1, 2).reshape(1, -1, side, side)
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1))[0]
        cam = torch.nn.functional.interpolate(cam[None, None], size=(cfg.input_size, cfg.input_size),
                                              mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        h1.remove()
        h2.remove()
