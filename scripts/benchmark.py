"""Benchmark WFR-MFM speed and accuracy over different inference step counts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wfr_mfm.checkpoint import load_checkpoint
from wfr_mfm.data import load_snapshots
from wfr_mfm.inference import rollout
from wfr_mfm.metrics import relative_mass_error, weighted_wasserstein1
from wfr_mfm.training import resolve_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 30])
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    if model.condition_dimension:
        raise ValueError("This benchmark expects an unconditional checkpoint")
    data_config = payload["config"]["data"]
    data = load_snapshots(
        args.data,
        time_column=str(data_config.get("time_column", "samples")),
        feature_columns=list(payload["feature_names"]),
    )
    initial = torch.as_tensor(data.snapshots[0], device=device, dtype=torch.float32)
    initial_count = initial.shape[0]
    records = []

    for steps in args.steps:
        if steps < 1:
            raise ValueError("All step counts must be positive")
        for _ in range(args.warmup):
            rollout(model, initial, data.times, steps_per_interval=steps)
        synchronize(device)
        durations = []
        for _ in range(args.runs):
            synchronize(device)
            start = time.perf_counter()
            predictions = rollout(model, initial, data.times, steps_per_interval=steps)
            synchronize(device)
            durations.append(time.perf_counter() - start)

        predictions = rollout(model, initial, data.times, steps_per_interval=steps)
        w1_values = []
        rme_values = []
        for snapshot_time, observed in zip(data.times[1:], data.snapshots[1:]):
            position, mass = predictions[snapshot_time]
            position_np = position.detach().cpu().numpy()
            mass_np = mass.detach().cpu().numpy().reshape(-1)
            w1_values.append(weighted_wasserstein1(position_np, observed, mass_np))
            rme_values.append(
                relative_mass_error(mass_np, initial_count, observed.shape[0])
            )
        records.append(
            {
                "steps_per_interval": steps,
                "time_mean_seconds": float(np.mean(durations)),
                "time_sd_seconds": float(np.std(durations, ddof=1)) if args.runs > 1 else 0.0,
                "w1_mean": float(np.mean(w1_values)),
                "rme_mean": float(np.mean(rme_values)),
                "runs": args.runs,
            }
        )
        print(pd.DataFrame(records[-1:], index=[0]).to_string(index=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved benchmark: {output.resolve()}")


if __name__ == "__main__":
    main()
