from __future__ import annotations

import torch


@torch.no_grad()
def sample_time_pairs(
    batch_size: int,
    *,
    p_diff: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 4: sample ordered uniform pairs mixed with diagonal pairs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= p_diff <= 1.0:
        raise ValueError("p_diff must lie in [0, 1]")
    first = torch.rand((batch_size, 1), device=device, dtype=dtype)
    second = torch.rand((batch_size, 1), device=device, dtype=dtype)
    t = torch.minimum(first, second)
    terminal_time = torch.maximum(first, second)
    diagonal = torch.rand((batch_size, 1), device=device) >= p_diff
    same_time = torch.rand((batch_size, 1), device=device, dtype=dtype)
    t = torch.where(diagonal, same_time, t)
    terminal_time = torch.where(diagonal, same_time, terminal_time)
    return t.clamp(eps, 1.0 - eps), terminal_time.clamp(eps, 1.0 - eps)
