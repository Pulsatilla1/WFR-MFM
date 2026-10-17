from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("data", "model", "transport", "training")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and attach its repository root."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_root"]) / path


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove local bookkeeping keys before saving a checkpoint."""
    return {key: value for key, value in config.items() if not key.startswith("_")}
