from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import MeanFlowModel


def save_checkpoint(
    path: str | Path,
    model: MeanFlowModel,
    *,
    config: dict[str, Any],
    feature_names: list[str] | tuple[str, ...],
    times: list[float] | tuple[float, ...],
    embedding_names: list[str] | tuple[str, ...] = (),
    embedding_mean: list[float] | None = None,
    embedding_std: list[float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "model": {
            "dimension": model.dimension,
            "condition_dimension": model.condition_dimension,
            "hidden_dim": int(config["model"]["hidden_dim"]),
            "hidden_layers": int(config["model"]["hidden_layers"]),
            "activation": str(config["model"]["activation"]),
        },
        "config": config,
        "feature_names": list(feature_names),
        "times": [float(value) for value in times],
        "embedding_names": list(embedding_names),
        "embedding_mean": embedding_mean,
        "embedding_std": embedding_std,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[MeanFlowModel, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = MeanFlowModel(**payload["model"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
