from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from .checkpoint import save_checkpoint
from .config import public_config, resolve_path
from .data import SnapshotData, load_snapshots
from .models import MeanFlowModel
from .objectives import mean_flow_targets
from .paths import traveling_dirac_path
from .time_sampling import sample_time_pairs
from .transport import WFRPlan, solve_wfr_plan


@dataclass(frozen=True)
class PreparedInterval:
    start: float
    stop: float
    plan: WFRPlan
    delta: float
    bandwidth: float = 0.0


def resolve_device(value: str | torch.device = "auto") -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def prepare_intervals(
    data: SnapshotData,
    transport_config: dict[str, Any],
    rng: np.random.Generator,
) -> list[PreparedInterval]:
    delta = float(transport_config["delta"])
    bandwidth = float(transport_config.get("bandwidth", 0.0))
    return [
        PreparedInterval(
            start,
            stop,
            solve_wfr_plan(
                source,
                target,
                delta=delta,
                mini_batch=bool(transport_config.get("mini_batch", False)),
                chunk_size=int(transport_config.get("chunk_size", 1500)),
                rng=rng,
            ),
            delta,
            bandwidth,
        )
        for start, stop, source, target in zip(
            data.times[:-1], data.times[1:], data.snapshots[:-1], data.snapshots[1:]
        )
    ]


def interval_batch(
    interval: PreparedInterval,
    model: MeanFlowModel,
    *,
    batch_size: int,
    p_diff: float,
    device: torch.device,
    rng: np.random.Generator,
    condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target, mass0, mass1 = interval.plan.sample(batch_size, rng)
    source_tensor = torch.as_tensor(source, device=device, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, device=device, dtype=torch.float32)
    mass0_tensor = torch.as_tensor(mass0, device=device, dtype=torch.float32)
    mass1_tensor = torch.as_tensor(mass1, device=device, dtype=torch.float32)
    relative_t, relative_terminal = sample_time_pairs(
        batch_size, p_diff=p_diff, device=device
    )
    length = interval.stop - interval.start
    time = interval.start + length * relative_t
    terminal_time = interval.start + length * relative_terminal
    path = traveling_dirac_path(
        relative_t,
        length,
        source_tensor,
        target_tensor,
        mass0_tensor,
        mass1_tensor,
        interval.delta,
    )
    time = time[path.feasible]
    terminal_time = terminal_time[path.feasible]
    if path.position.shape[0] == 0:
        raise RuntimeError("No feasible WFR pair remained in a sampled batch")
    position = path.position
    if interval.bandwidth > 0:
        position = position + interval.bandwidth * torch.randn_like(position)
    velocity_target, growth_target = mean_flow_targets(
        model,
        position,
        time,
        terminal_time,
        path.velocity,
        path.growth,
        condition=condition,
    )
    return (
        position,
        time,
        terminal_time,
        path.relative_mass,
        velocity_target,
        growth_target,
    )


def fit_unconditional(
    model: MeanFlowModel,
    intervals: list[PreparedInterval],
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
    lambda_growth = float(training_config["lambda_growth"])
    batch_size = int(training_config["batch_size"])
    p_diff = float(training_config["p_diff"])
    log_every = int(training_config.get("log_every", 50))
    history: list[dict[str, float]] = []

    progress = trange(epochs, desc="Training WFR-MFM", unit="epoch")
    for epoch in progress:
        optimizer.zero_grad(set_to_none=True)
        batches = [
            interval_batch(
                interval,
                model,
                batch_size=batch_size,
                p_diff=p_diff,
                device=device,
                rng=rng,
            )
            for interval in intervals
        ]
        position, time, terminal_time, weight, velocity_target, growth_target = (
            torch.cat(parts, dim=0) for parts in zip(*batches)
        )
        velocity, growth = model(position, time, terminal_time)
        velocity_loss = ((velocity - velocity_target).square() * weight).mean()
        growth_loss = ((growth - growth_target).square() * weight).mean()
        loss = velocity_loss + lambda_growth * growth_loss
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
            "velocity_loss": float(velocity_loss.detach()),
            "growth_loss": float(growth_loss.detach()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if epoch == 0 or (epoch + 1) % log_every == 0:
            progress.set_postfix(loss=f"{record['loss']:.5g}")
    return history


def train_from_config(
    config: dict[str, Any],
    output: str | Path,
    *,
    device_value: str = "auto",
    hold_out: list[float] | None = None,
) -> Path:
    seed = int(config.get("seed", 0))
    rng = seed_everything(seed)
    device = resolve_device(device_value)
    data_config = config["data"]
    data = load_snapshots(
        resolve_path(config, data_config["path"]),
        time_column=str(data_config.get("time_column", "samples")),
        dimension=int(data_config["dimension"]),
        feature_columns=data_config.get("feature_columns"),
    )
    if hold_out:
        data = data.without_times(hold_out)
    model = MeanFlowModel(
        data.dimension,
        hidden_dim=int(config["model"]["hidden_dim"]),
        hidden_layers=int(config["model"]["hidden_layers"]),
        activation=str(config["model"]["activation"]),
    )
    intervals = prepare_intervals(data, config["transport"], rng)
    history = fit_unconditional(
        model, intervals, config["training"], device=device, rng=rng
    )
    output = Path(output)
    save_checkpoint(
        output,
        model,
        config=public_config(config),
        feature_names=data.feature_names,
        times=data.times,
    )
    pd.DataFrame(history).to_csv(output.with_suffix(".history.csv"), index=False)
    return output
