"""Settings from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """A setting has an invalid value."""


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                v = v.strip().strip("'\"")
                if v:
                    out[k.strip()] = v
    return out


def _num(env: Mapping[str, str], key: str, default, cast, low, high=None):
    raw = env.get(key)
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} has an invalid value {raw!r}") from exc
    if value < low or (high is not None and value > high):
        raise ConfigError(f"{key} must be in [{low}, {high}], got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path | None = None
    metadata_csv: Path | None = None
    output_dir: Path = Path("outputs")
    seed: int = 42
    image_size: int = 64
    hash_distance: int = 16
    test_fraction: float = 0.2
    folds: int = 5
    n_bootstrap: int = 1000
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0.05 <= self.test_fraction <= 0.5:
            raise ConfigError("test_fraction must be in [0.05, 0.5]")
        if self.folds < 2:
            raise ConfigError("folds must be >= 2")
        if not 0 <= self.hash_distance <= 128:
            raise ConfigError("hash_distance must be in [0, 128] (bits of a 256-bit hash)")

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: Path | None = Path(".env")) -> "Settings":
        merged: dict[str, str] = {}
        if env is None:
            if dotenv is not None:
                merged.update(_read_dotenv(dotenv))
            merged.update(os.environ)
        else:
            merged.update(env)
        data = merged.get("CELLSIGHT_DATA_DIR")
        meta = merged.get("CELLSIGHT_METADATA")
        return cls(
            data_dir=Path(data) if data else None,
            metadata_csv=Path(meta) if meta else None,
            output_dir=Path(merged.get("CELLSIGHT_OUTPUT_DIR") or "outputs"),
            seed=_num(merged, "CELLSIGHT_SEED", 42, int, 0),
            image_size=_num(merged, "CELLSIGHT_IMAGE_SIZE", 64, int, 16, 1024),
            hash_distance=_num(merged, "CELLSIGHT_HASH_DISTANCE", 16, int, 0, 128),
            test_fraction=_num(merged, "CELLSIGHT_TEST_FRACTION", 0.2, float, 0.05, 0.5),
            folds=_num(merged, "CELLSIGHT_FOLDS", 5, int, 2),
            n_bootstrap=_num(merged, "CELLSIGHT_BOOTSTRAP", 1000, int, 50),
            device=merged.get("CELLSIGHT_DEVICE") or "auto",
        )
