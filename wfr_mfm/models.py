from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    choices: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
    }
    try:
        return choices[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown activation: {name}") from error


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        activation: str = "leakyrelu",
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        activation_type = _activation(activation)
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(width, hidden_dim), activation_type()))
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MeanFlowModel(nn.Module):
    """Mean velocity and mass-growth fields over a finite interval."""

    def __init__(
        self,
        dimension: int,
        *,
        condition_dimension: int = 0,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        activation: str = "leakyrelu",
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.condition_dimension = int(condition_dimension)
        input_dim = self.dimension + self.condition_dimension + 2
        self.velocity = MLP(
            input_dim, self.dimension, hidden_dim, hidden_layers, activation
        )
        self.growth = MLP(input_dim, 1, hidden_dim, hidden_layers, activation)

    def _inputs(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        terminal_time: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        t = t.reshape(-1, 1).expand(batch, 1)
        terminal_time = terminal_time.reshape(-1, 1).expand(batch, 1)
        pieces = [x]
        if self.condition_dimension:
            if condition is None:
                raise ValueError("A condition embedding is required by this model")
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            condition = condition.expand(batch, -1)
            if condition.shape[1] != self.condition_dimension:
                raise ValueError("Condition embedding has the wrong dimension")
            pieces.append(condition)
        pieces.extend((t, terminal_time))
        return torch.cat(pieces, dim=1)

    def mean_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        terminal_time: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity(self._inputs(x, t, terminal_time, condition))

    def mean_growth(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        terminal_time: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.growth(self._inputs(x, t, terminal_time, condition))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        terminal_time: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs(x, t, terminal_time, condition)
        return self.velocity(inputs), self.growth(inputs)
