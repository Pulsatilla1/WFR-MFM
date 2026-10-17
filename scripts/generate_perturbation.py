"""Generate the three-gene perturbation benchmark described in Appendix C.12."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm.auto import trange

ALPHA = np.array([1.0, 5.0, 10.0])
GAMMA = np.array([3.5, 1.0, 8.0])
DEGRADATION = np.ones(3)
NOISE = np.array([0.1, 0.01, 0.05])
RHO3 = 2.0


def drift(state: np.ndarray, rho1: float, rho2: float) -> np.ndarray:
    x1, x2, x3 = state.T
    dx1 = (rho1 + ALPHA[0] * x1**2) / (
        1.0 + ALPHA[0] * x1**2 + GAMMA[1] * x2**2 + GAMMA[2] * x3**2
    ) - DEGRADATION[0] * x1
    dx2 = (rho2 + ALPHA[1] * x2**2) / (
        1.0 + GAMMA[0] * x1**2 + ALPHA[1] * x2**2 + GAMMA[2] * x3**2
    ) - DEGRADATION[1] * x2
    dx3 = (RHO3 + ALPHA[2] * x3**2) / (1.0 + ALPHA[2] * x3**2) - DEGRADATION[2] * x3
    return np.stack((dx1, dx2, dx3), axis=1)


def equilibrium() -> np.ndarray:
    state = np.ones((1, 3), dtype=np.float64)
    for _ in range(20_000):
        state += 0.01 * drift(state, 100.0, 100.0)
        np.maximum(state, 0.0, out=state)
    return state[0]


def simulate(
    initial: np.ndarray,
    rho1: float,
    rho2: float,
    alpha_growth: float,
    *,
    duration: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    state = initial.copy()
    for _ in range(round(duration / dt)):
        if not len(state):
            break
        state += dt * drift(state, rho1, rho2)
        state += rng.normal(size=state.shape) * (np.sqrt(dt) * NOISE)
        np.maximum(state, 0.0, out=state)
        death_rate = (-alpha_growth) * state[:, 2] ** 2 / (1.0 + state[:, 2] ** 2)
        death_probability = 1.0 - np.exp(-death_rate * dt)
        state = state[rng.random(len(state)) > death_probability]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=int, default=100)
    parser.add_argument("--cells", type=int, default=10_000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--data-output", required=True)
    parser.add_argument("--embedding-output", required=True)
    args = parser.parse_args()
    if args.conditions < 1 or args.cells < 1 or args.dt <= 0 or args.duration <= 0:
        raise ValueError("conditions, cells, duration, and dt must be positive")

    rng = np.random.default_rng(args.seed)
    mean = equilibrium()
    initial = rng.multivariate_normal(
        mean, np.diag([0.05, 0.001, 0.01]), size=args.cells
    ).astype(np.float32)
    np.maximum(initial, 0.0, out=initial)
    parameters = []
    for index in range(args.conditions):
        label = args.label_offset + index + 1
        rho1 = float(np.exp(rng.uniform(np.log(1.0), np.log(1000.0))))
        rho2 = float(np.exp(rng.uniform(np.log(1.0), np.log(1000.0))))
        alpha_growth = float(rng.uniform(-0.4, -0.1))
        parameters.append((label, rho1, rho2, alpha_growth))

    data_output = Path(args.data_output)
    embedding_output = Path(args.embedding_output)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    embedding_output.parent.mkdir(parents=True, exist_ok=True)
    with embedding_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("samples", "rho1", "rho2", "alpha_g"))
        writer.writerow((0, 100.0, 100.0, 0.0))
        writer.writerows(parameters)

    with data_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("samples", "X1", "X2", "X3"))
        writer.writerows((0, *row) for row in initial)
        seeds = np.random.SeedSequence(args.seed).spawn(args.conditions)
        for index in trange(args.conditions, desc="Simulating perturbations"):
            label, rho1, rho2, alpha_growth = parameters[index]
            endpoint = simulate(
                initial,
                rho1,
                rho2,
                alpha_growth,
                duration=args.duration,
                dt=args.dt,
                rng=np.random.default_rng(seeds[index]),
            )
            writer.writerows((label, *row) for row in endpoint)
    print(f"Saved cells to {data_output.resolve()}")
    print(f"Saved embeddings to {embedding_output.resolve()}")


if __name__ == "__main__":
    main()
