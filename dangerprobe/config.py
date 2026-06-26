"""Load and validate config.yaml. Single entry point for all runtime settings.

Usage:
    from dangerprobe.config import load_config
    cfg = load_config()                  # reads ./config.yaml
    mc = cfg.model_config()              # resolved model block for cfg.active
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the dangerprobe package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class ModelConfig:
    name: str
    device: str
    dtype: str
    layers: list[int]
    use_chat_template: bool


class Config:
    """Thin typed wrapper over the parsed YAML dict."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.active: str = raw["active"]
        if self.active not in raw["model"]:
            raise ValueError(
                f"active='{self.active}' has no matching model block; "
                f"options: {list(raw['model'])}"
            )

    # --- convenience accessors -------------------------------------------
    def model_config(self) -> ModelConfig:
        m = self.raw["model"][self.active]
        return ModelConfig(
            name=m["name"],
            device=m["device"],
            dtype=m["dtype"],
            layers=list(m["layers"]),
            use_chat_template=bool(m["use_chat_template"]),
        )

    @property
    def pooling_methods(self) -> list[str]:
        return list(self.raw["pooling"]["methods"])

    @property
    def thresholds(self) -> dict[str, float]:
        return self.raw["thresholds"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def split(self) -> dict[str, Any]:
        return self.raw["split"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        """Resolve a path from the `paths` block against the repo root."""
        rel = self.raw["paths"][key]
        return REPO_ROOT / rel

    def dataset_out_path(self) -> Path:
        return REPO_ROOT / self.dataset["out_path"]


_REQUIRED_TOP_KEYS = {
    "active",
    "model",
    "pooling",
    "probe",
    "thresholds",
    "dataset",
    "split",
    "paths",
    "seed",
}


def _validate(raw: dict[str, Any]) -> None:
    missing = _REQUIRED_TOP_KEYS - set(raw)
    if missing:
        raise ValueError(f"config.yaml missing keys: {sorted(missing)}")
    th = raw["thresholds"]
    if not (0.0 <= th["safe"] <= th["block"] <= 1.0):
        raise ValueError(
            f"thresholds must satisfy 0 <= safe <= block <= 1, got {th}"
        )
    for name, block in raw["model"].items():
        for k in ("name", "device", "dtype", "layers", "use_chat_template"):
            if k not in block:
                raise ValueError(f"model.{name} missing '{k}'")


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _validate(raw)
    return Config(raw)


if __name__ == "__main__":
    cfg = load_config()
    mc = cfg.model_config()
    print(f"active={cfg.active}  model={mc.name}  device={mc.device}  layers={mc.layers}")
    print(f"pooling={cfg.pooling_methods}  thresholds={cfg.thresholds}")
