from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch

from .models import MeanFlowModel


@torch.no_grad()
def rollout(
    model: MeanFlowModel,
    initial_positions: torch.Tensor,
    times: Sequence[float],
    *,
    initial_masses: torch.Tensor | None = None,
    steps_per_interval: int = 1,
    condition: torch.Tensor | None = None,
    clamp_log_mass: tuple[float, float] | None = (-30.0, 30.0),
) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
    """Algorithm 1 with one or more additive-consistency steps per interval."""
    times = [float(value) for value in times]
    if len(times) < 2 or any(right <= left for left, right in pairwise(times)):
        raise ValueError("times must contain at least two strictly increasing values")
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be positive")
    position = initial_positions
    if initial_masses is None:
        mass = torch.ones((position.shape[0], 1), device=position.device, dtype=position.dtype)
    else:
        mass = initial_masses.reshape(-1, 1).to(position)
    outputs = {times[0]: (position.clone(), mass.clone())}

    for start, stop in pairwise(times):
        sub_times = torch.linspace(
            start,
            stop,
            steps_per_interval + 1,
            device=position.device,
            dtype=position.dtype,
        )
        for left, right in pairwise(sub_times):
            batch = position.shape[0]
            t = left.expand(batch, 1)
            terminal_time = right.expand(batch, 1)
            velocity, growth = model(position, t, terminal_time, condition)
            step = right - left
            position = position + step * velocity
            log_multiplier = step * growth
            if clamp_log_mass is not None:
                log_multiplier = log_multiplier.clamp(*clamp_log_mass)
            mass = mass * torch.exp(log_multiplier)
        outputs[stop] = (position.clone(), mass.clone())
    return outputs
