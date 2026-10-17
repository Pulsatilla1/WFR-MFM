from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import ot


def wfr_cost(source: np.ndarray, target: np.ndarray, delta: float) -> np.ndarray:
    """Pairwise WFR traveling-Dirac cost used by the OET coupling."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    distances = ot.dist(source, target, metric="euclidean")
    angles = np.minimum(distances / (2.0 * delta), np.pi / 2.0)
    cosine_squared = np.cos(angles) ** 2
    return -np.log(np.maximum(cosine_squared, 1e-12))


@dataclass
class CouplingBlock:
    source_indices: np.ndarray
    target_indices: np.ndarray
    gamma0: np.ndarray
    gamma1: np.ndarray

    @property
    def mass(self) -> float:
        return float(self.gamma0.sum())

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        probabilities = self.gamma0.reshape(-1).astype(np.float64)
        total = probabilities.sum()
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("Encountered an empty OET coupling block")
        probabilities /= total
        flat = rng.choice(probabilities.size, size=count, replace=True, p=probabilities)
        local_source, local_target = np.unravel_index(flat, self.gamma0.shape)
        return (
            self.source_indices[local_source],
            self.target_indices[local_target],
            self.gamma0[local_source, local_target],
            self.gamma1[local_source, local_target],
        )


@dataclass
class WFRPlan:
    source: np.ndarray
    target: np.ndarray
    blocks: list[CouplingBlock]

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        masses = np.asarray([block.mass for block in self.blocks], dtype=np.float64)
        if masses.sum() <= 0:
            raise RuntimeError("The OET plan has zero total mass")
        block_ids = rng.choice(len(self.blocks), size=batch_size, p=masses / masses.sum())
        source_indices = np.empty(batch_size, dtype=np.int64)
        target_indices = np.empty(batch_size, dtype=np.int64)
        mass0 = np.empty(batch_size, dtype=np.float32)
        mass1 = np.empty(batch_size, dtype=np.float32)
        for block_id in np.unique(block_ids):
            positions = np.flatnonzero(block_ids == block_id)
            i, j, m0, m1 = self.blocks[int(block_id)].sample(len(positions), rng)
            source_indices[positions] = i
            target_indices[positions] = j
            mass0[positions] = m0
            mass1[positions] = m1
        return (
            self.source[source_indices],
            self.target[target_indices],
            mass0[:, None],
            mass1[:, None],
        )


def _solve_block(
    source: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    delta: float,
) -> CouplingBlock:
    source_part = source[source_indices].astype(np.float64, copy=False)
    target_part = target[target_indices].astype(np.float64, copy=False)
    a = np.ones(source_part.shape[0], dtype=np.float64)
    b = np.ones(target_part.shape[0], dtype=np.float64)
    cost = wfr_cost(source_part, target_part, delta)
    coupling = np.asarray(
        ot.unbalanced.mm_unbalanced(a, b, cost, reg_m=(1.0, 1.0)), dtype=np.float64
    )
    row_sum = coupling.sum(axis=1, keepdims=True)
    column_sum = coupling.sum(axis=0, keepdims=True)
    gamma0 = coupling * (a[:, None] / np.maximum(row_sum, 1e-12))
    gamma1 = coupling * (b[None, :] / np.maximum(column_sum, 1e-12))
    return CouplingBlock(
        source_indices=source_indices,
        target_indices=target_indices,
        gamma0=gamma0.astype(np.float32),
        gamma1=gamma1.astype(np.float32),
    )


def solve_wfr_plan(
    source: np.ndarray,
    target: np.ndarray,
    *,
    delta: float,
    mini_batch: bool = False,
    chunk_size: int = 1500,
    rng: np.random.Generator | None = None,
) -> WFRPlan:
    """Solve the WFR OET problem, optionally using paired random blocks."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be 2D arrays with equal feature dimension")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("source and target must be non-empty")
    rng = rng or np.random.default_rng()
    if not mini_batch:
        source_indices = np.arange(source.shape[0])
        target_indices = np.arange(target.shape[0])
        return WFRPlan(
            source,
            target,
            [_solve_block(source, target, source_indices, target_indices, delta)],
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    block_count = ceil(max(source.shape[0], target.shape[0]) / chunk_size)
    block_count = max(1, min(block_count, source.shape[0], target.shape[0]))
    source_groups = np.array_split(rng.permutation(source.shape[0]), block_count)
    target_groups = np.array_split(rng.permutation(target.shape[0]), block_count)
    blocks = [
        _solve_block(source, target, source_group, target_group, delta)
        for source_group, target_group in zip(source_groups, target_groups)
    ]
    return WFRPlan(source, target, blocks)
