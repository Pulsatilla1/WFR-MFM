from __future__ import annotations

import torch
from torch.func import jvp

from .models import MeanFlowModel


def mean_flow_targets(
    model: MeanFlowModel,
    position: torch.Tensor,
    time: torch.Tensor,
    terminal_time: torch.Tensor,
    instantaneous_velocity: torch.Tensor,
    instantaneous_growth: torch.Tensor,
    *,
    condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the stopped-gradient derivative-identity targets in Sec. 4.2."""
    interval = terminal_time - time
    direction_x = interval * instantaneous_velocity
    direction_t = interval
    condition = None if condition is None else condition.detach()
    terminal_time = terminal_time.detach()

    def velocity_field(x_value: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
        return model.mean_velocity(x_value, t_value, terminal_time, condition)

    def growth_field(x_value: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
        return model.mean_growth(x_value, t_value, terminal_time, condition)

    with torch.enable_grad():
        _, velocity_derivative = jvp(
            velocity_field,
            (position.detach(), time.detach()),
            (direction_x.detach(), direction_t.detach()),
        )
        _, growth_derivative = jvp(
            growth_field,
            (position.detach(), time.detach()),
            (direction_x.detach(), direction_t.detach()),
        )
    velocity_target = instantaneous_velocity + velocity_derivative
    growth_target = instantaneous_growth + growth_derivative
    return velocity_target.detach(), growth_target.detach()
