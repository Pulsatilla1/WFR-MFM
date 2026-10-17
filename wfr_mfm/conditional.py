from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from .checkpoint import save_checkpoint
from .config import public_config, resolve_path
from .data import ConditionalData, load_conditional_data
from .models import MeanFlowModel
from .training import PreparedInterval, interval_batch, resolve_device, seed_everything
from .transport import solve_wfr_plan


@dataclass(frozen=True)
class ConditionalInterval:
    label: int
    embedding: np.ndarray
    interval: PreparedInterval


def _condition_delta(
    data: ConditionalData, label: int, transport_config: dict[str, Any]
) -> float:
    base = float(transport_config.get("delta_base", 1.0))
    mode = str(transport_config.get("delta_mode", "fixed"))
    if mode == "fixed":
        return base
    if mode == "endpoint_ratio":
        return base * data.endpoints[label].shape[0] / data.control.shape[0]
    raise ValueError(f"Unknown conditional delta mode: {mode}")


def prepare_conditional_intervals(
    data: ConditionalData,
    transport_config: dict[str, Any],
    rng: np.random.Generator,
) -> list[ConditionalInterval]:
    prepared = []
    bandwidth = float(transport_config.get("bandwidth", 0.0))
    for label, endpoint in data.endpoints.items():
        delta = _condition_delta(data, label, transport_config)
        plan = solve_wfr_plan(
            data.control,
            endpoint,
            delta=delta,
            mini_batch=bool(transport_config.get("mini_batch", False)),
            chunk_size=int(transport_config.get("chunk_size", 1500)),
            rng=rng,
        )
        prepared.append(
            ConditionalInterval(
                label,
                data.embeddings[label],
                PreparedInterval(0.0, 1.0, plan, delta, bandwidth),
            )
        )
    return prepared


def fit_conditional(
    model: MeanFlowModel,
    intervals: list[ConditionalInterval],
    training_config: dict[str, Any],
    *,
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    model.to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    epochs = int(training_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    batch_size = int(training_config["batch_size"])
    condition_batch_size = min(
        int(training_config.get("condition_batch_size", len(intervals))), len(intervals)
    )
    p_diff = float(training_config["p_diff"])
    lambda_growth = float(training_config["lambda_growth"])
    log_every = int(training_config.get("log_every", 20))
    history: list[dict[str, float]] = []
    progress = trange(epochs, desc="Training conditional WFR-MFM", unit="epoch")

    for epoch in progress:
        optimizer.zero_grad(set_to_none=True)
        chosen = rng.choice(len(intervals), size=condition_batch_size, replace=False)
        condition_losses = []
        velocity_losses = []
        growth_losses = []
        for index in chosen:
            item = intervals[int(index)]
            condition = torch.as_tensor(item.embedding, device=device, dtype=torch.float32)
            position, time, terminal_time, weight, velocity_target, growth_target = interval_batch(
                item.interval,
                model,
                batch_size=batch_size,
                p_diff=p_diff,
                device=device,
                rng=rng,
                condition=condition,
            )
            velocity, growth = model(position, time, terminal_time, condition)
            velocity_loss = ((velocity - velocity_target).square() * weight).mean()
            growth_loss = ((growth - growth_target).square() * weight).mean()
            condition_losses.append(velocity_loss + lambda_growth * growth_loss)
            velocity_losses.append(velocity_loss)
            growth_losses.append(growth_loss)
        loss = torch.stack(condition_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        loss.backward()
        if training_config.get("gradient_clip") is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip"])
            )
        optimizer.step()
        scheduler.step()
        record = {
            "epoch": float(epoch + 1),
            "loss": float(loss.detach()),
            "velocity_loss": float(torch.stack(velocity_losses).mean().detach()),
            "growth_loss": float(torch.stack(growth_losses).mean().detach()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if epoch == 0 or (epoch + 1) % log_every == 0:
            progress.set_postfix(loss=f"{record['loss']:.5g}")
    return history


def train_conditional_from_config(
    config: dict[str, Any],
    output: str | Path,
    *,
    device_value: str = "auto",
) -> Path:
    rng = seed_everything(int(config.get("seed", 0)))
    device = resolve_device(device_value)
    data_config = config["data"]
    data = load_conditional_data(
        resolve_path(config, data_config["path"]),
        resolve_path(config, data_config["embedding_path"]),
        condition_column=str(data_config.get("condition_column", "samples")),
        control_label=int(data_config.get("control_label", 0)),
        dimension=int(data_config["dimension"]),
    )
    model = MeanFlowModel(
        data.dimension,
        condition_dimension=data.condition_dimension,
        hidden_dim=int(config["model"]["hidden_dim"]),
        hidden_layers=int(config["model"]["hidden_layers"]),
        activation=str(config["model"]["activation"]),
    )
    intervals = prepare_conditional_intervals(data, config["transport"], rng)
    history = fit_conditional(
        model, intervals, config["training"], device=device, rng=rng
    )
    output = Path(output)
    save_checkpoint(
        output,
        model,
        config=public_config(config),
        feature_names=data.feature_names,
        times=(0.0, 1.0),
        embedding_names=data.embedding_names,
        embedding_mean=data.embedding_mean.tolist(),
        embedding_std=data.embedding_std.tolist(),
    )
    pd.DataFrame(history).to_csv(output.with_suffix(".history.csv"), index=False)
    return output
