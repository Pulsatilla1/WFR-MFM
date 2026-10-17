from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WFRPathBatch:
    position: torch.Tensor
    velocity: torch.Tensor
    growth: torch.Tensor
    relative_mass: torch.Tensor
    feasible: torch.Tensor


def traveling_dirac_path(
    relative_time: torch.Tensor,
    interval_length: float,
    source: torch.Tensor,
    target: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    delta: float,
) -> WFRPathBatch:
    """Evaluate the WFR traveling-Dirac path and its instantaneous fields."""
    if interval_length <= 0:
        raise ValueError("interval_length must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    displacement = target - source
    distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
    feasible = (distance[:, 0] < torch.pi * delta)
    source = source[feasible]
    target = target[feasible]
    source_mass = source_mass[feasible].clamp_min(1e-12)
    target_mass = target_mass[feasible].clamp_min(1e-12)
    relative_time = relative_time[feasible]
    displacement = target - source
    distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
    direction = displacement / distance.clamp_min(1e-12)

    tau = torch.tan(distance / (2.0 * delta))
    scale = torch.sqrt(source_mass * target_mass / (1.0 + tau.square()))
    omega = 2.0 * delta * tau * scale
    omega_vector = omega * direction
    coefficient_a = target_mass + source_mass - 2.0 * scale
    coefficient_b = source_mass - scale
    inverse = 2.0 * delta / omega.clamp_min(1e-12)
    angle = torch.atan((coefficient_a * relative_time - coefficient_b) * inverse)
    initial_angle = torch.atan(-coefficient_b * inverse)
    position = source + omega_vector * inverse * (angle - initial_angle)

    mass = (
        coefficient_a * relative_time.square()
        - 2.0 * coefficient_b * relative_time
        + source_mass
    ).clamp_min(1e-12)
    mass_derivative = 2.0 * coefficient_a * relative_time - 2.0 * coefficient_b
    growth = (mass_derivative / mass) / interval_length
    velocity = (omega_vector / mass) / interval_length
    relative_mass = mass / source_mass
    return WFRPathBatch(position, velocity, growth, relative_mass, feasible)
