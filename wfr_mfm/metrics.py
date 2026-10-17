from __future__ import annotations

import numpy as np
import ot


def weighted_wasserstein1(
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_weights: np.ndarray | None = None,
) -> float:
    if predicted_weights is None:
        predicted_weights = np.ones(predicted.shape[0], dtype=np.float64)
    predicted_weights = np.asarray(predicted_weights, dtype=np.float64).reshape(-1)
    predicted_weights = predicted_weights / predicted_weights.sum()
    observed_weights = np.full(observed.shape[0], 1.0 / observed.shape[0])
    cost = ot.dist(predicted.astype(np.float64), observed.astype(np.float64), metric="euclidean")
    return float(ot.emd2(predicted_weights, observed_weights, cost))


def relative_mass_error(
    predicted_masses: np.ndarray,
    initial_count: int,
    observed_count: int,
) -> float:
    predicted_ratio = float(np.asarray(predicted_masses).sum()) / float(initial_count)
    observed_ratio = float(observed_count) / float(initial_count)
    return abs(predicted_ratio - observed_ratio) / max(observed_ratio, 1e-12)
