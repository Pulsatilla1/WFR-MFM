from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnapshotData:
    times: tuple[float, ...]
    snapshots: tuple[np.ndarray, ...]
    feature_names: tuple[str, ...]
    time_column: str

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(array.shape[0] for array in self.snapshots)

    def without_times(self, excluded: Iterable[float]) -> SnapshotData:
        excluded_set = {float(value) for value in excluded}
        keep = [index for index, time in enumerate(self.times) if time not in excluded_set]
        if len(keep) < 2:
            raise ValueError("At least two snapshots are required after hold-out filtering")
        return SnapshotData(
            times=tuple(self.times[index] for index in keep),
            snapshots=tuple(self.snapshots[index] for index in keep),
            feature_names=self.feature_names,
            time_column=self.time_column,
        )


def _numeric_features(
    frame: pd.DataFrame,
    time_column: str,
    dimension: int | None,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        columns = list(feature_columns)
    else:
        columns = [
            column
            for column in frame.columns
            if column != time_column and pd.api.types.is_numeric_dtype(frame[column])
        ]
    if dimension is not None:
        if len(columns) < dimension:
            raise ValueError(f"Requested {dimension} features but found {len(columns)}")
        columns = columns[:dimension]
    if not columns:
        raise ValueError("No numeric feature columns were found")
    return columns


def load_snapshots(
    path: str | Path,
    *,
    time_column: str = "samples",
    dimension: int | None = None,
    feature_columns: list[str] | None = None,
) -> SnapshotData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if time_column not in frame.columns:
        raise ValueError(f"{path} has no '{time_column}' column")
    columns = _numeric_features(frame, time_column, dimension, feature_columns)
    if frame[[time_column, *columns]].isnull().any().any():
        raise ValueError(f"NaN values found in {path}")
    times = tuple(float(value) for value in sorted(frame[time_column].unique()))
    if len(times) < 2:
        raise ValueError("At least two distinct snapshots are required")
    snapshots = tuple(
        frame.loc[frame[time_column] == time, columns].to_numpy(dtype=np.float32, copy=True)
        for time in times
    )
    if any(snapshot.shape[0] == 0 for snapshot in snapshots):
        raise ValueError("Empty snapshot encountered")
    return SnapshotData(times, snapshots, tuple(columns), time_column)


@dataclass(frozen=True)
class ConditionalData:
    control: np.ndarray
    endpoints: dict[int, np.ndarray]
    embeddings: dict[int, np.ndarray]
    feature_names: tuple[str, ...]
    embedding_names: tuple[str, ...]
    embedding_mean: np.ndarray
    embedding_std: np.ndarray
    control_label: int

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def condition_dimension(self) -> int:
        return len(self.embedding_names)


def load_conditional_data(
    cell_path: str | Path,
    embedding_path: str | Path,
    *,
    condition_column: str = "samples",
    control_label: int = 0,
    dimension: int | None = None,
) -> ConditionalData:
    cells = pd.read_csv(cell_path)
    embeddings = pd.read_csv(embedding_path)
    for frame, path in ((cells, cell_path), (embeddings, embedding_path)):
        if condition_column not in frame.columns:
            raise ValueError(f"{path} has no '{condition_column}' column")

    feature_names = _numeric_features(cells, condition_column, dimension, None)
    embedding_names = _numeric_features(embeddings, condition_column, None, None)
    control = cells.loc[
        cells[condition_column] == control_label, feature_names
    ].to_numpy(dtype=np.float32, copy=True)
    if control.size == 0:
        raise ValueError(f"Control label {control_label} is absent from {cell_path}")

    endpoint_labels = sorted(
        int(value) for value in cells[condition_column].unique() if int(value) != control_label
    )
    endpoints = {
        label: cells.loc[cells[condition_column] == label, feature_names].to_numpy(
            dtype=np.float32, copy=True
        )
        for label in endpoint_labels
    }
    embedding_rows = embeddings.set_index(condition_column)
    missing = [label for label in endpoint_labels if label not in embedding_rows.index]
    if missing:
        raise ValueError(f"Missing embeddings for conditions: {missing[:10]}")

    if embedding_rows.index.has_duplicates:
        raise ValueError(f"Duplicate condition labels found in {embedding_path}")

    # Match the manuscript source scripts: fit the embedding normalizer on the
    # complete training embedding table, including the control row when present.
    normalizer_values = embeddings.loc[:, embedding_names].to_numpy(dtype=np.float32, copy=True)
    mean = normalizer_values.mean(axis=0)
    std = normalizer_values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    raw = np.stack(
        [embedding_rows.loc[label, embedding_names].to_numpy(dtype=np.float32) for label in endpoint_labels]
    )
    normalized = (raw - mean) / std
    embedding_map = {label: normalized[index] for index, label in enumerate(endpoint_labels)}

    return ConditionalData(
        control=control,
        endpoints=endpoints,
        embeddings=embedding_map,
        feature_names=tuple(feature_names),
        embedding_names=tuple(embedding_names),
        embedding_mean=mean.astype(np.float32),
        embedding_std=std,
        control_label=control_label,
    )
