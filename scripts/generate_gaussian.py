"""Generate the 1000D Gaussian-mixture benchmark described in Appendix C.7."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gaussian(rng: np.random.Generator, count: int, mean: np.ndarray, std: float) -> np.ndarray:
    return rng.normal(mean, std, size=(count, mean.size)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/gaussian_1000d.csv")
    parser.add_argument("--dimension", type=int, default=1000)
    parser.add_argument("--std", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dimension < 2:
        raise ValueError("dimension must be at least two")
    rng = np.random.default_rng(args.seed)
    lower = np.zeros(args.dimension, dtype=np.float32)
    upper = lower.copy()
    upper[1] = 2.5
    left = lower.copy()
    left[0] = -1.0
    right = lower.copy()
    right[0] = 1.0

    initial = np.concatenate(
        (gaussian(rng, 100, upper, args.std), gaussian(rng, 400, lower, args.std))
    )
    target = np.concatenate(
        (
            gaussian(rng, 1000, upper, args.std),
            gaussian(rng, 200, left, args.std),
            gaussian(rng, 200, right, args.std),
        )
    )
    columns = [f"x{index}" for index in range(1, args.dimension + 1)]
    frame = pd.concat(
        (
            pd.DataFrame(initial, columns=columns).assign(samples=0.0),
            pd.DataFrame(target, columns=columns).assign(samples=1.0),
        ),
        ignore_index=True,
    )
    frame = frame[["samples", *columns]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} cells to {output.resolve()}")


if __name__ == "__main__":
    main()
