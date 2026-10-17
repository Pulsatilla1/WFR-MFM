from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoint import load_checkpoint
from .conditional import train_conditional_from_config
from .config import load_config
from .data import load_snapshots
from .inference import rollout
from .metrics import relative_mass_error, weighted_wasserstein1
from .training import resolve_device, train_from_config


def _write_rollout(
    output: str | Path,
    predictions: dict[float, tuple[torch.Tensor, torch.Tensor]],
    feature_names: Sequence[str],
    *,
    condition: int | None = None,
) -> None:
    frames = []
    for time, (positions, masses) in predictions.items():
        frame = pd.DataFrame(positions.detach().cpu().numpy(), columns=feature_names)
        frame.insert(0, "weight", masses.detach().cpu().numpy().reshape(-1))
        frame.insert(0, "particle", np.arange(frame.shape[0]))
        frame.insert(0, "time", time)
        if condition is not None:
            frame.insert(0, "condition", condition)
        frames.append(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    train_from_config(
        config,
        args.output,
        device_value=args.device,
        hold_out=args.hold_out,
    )
    print(f"Saved checkpoint: {Path(args.output).resolve()}")


def command_train_conditional(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    train_conditional_from_config(config, args.output, device_value=args.device)
    print(f"Saved checkpoint: {Path(args.output).resolve()}")


def _load_inference_data(
    data_path: str | Path, payload: dict, time_column: str | None = None
):
    config_data = payload["config"]["data"]
    return load_snapshots(
        data_path,
        time_column=time_column or str(config_data.get("time_column", "samples")),
        feature_columns=list(payload["feature_names"]),
    )


def command_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    if model.condition_dimension:
        raise ValueError("Use infer-conditional for a conditional checkpoint")
    data = _load_inference_data(args.data, payload, args.time_column)
    initial = torch.as_tensor(data.snapshots[0], device=device, dtype=torch.float32)
    predictions = rollout(
        model,
        initial,
        data.times,
        steps_per_interval=args.steps_per_interval,
    )
    _write_rollout(args.output, predictions, data.feature_names)
    print(f"Saved predictions: {Path(args.output).resolve()}")


def command_evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    if model.condition_dimension:
        raise ValueError("This evaluator expects an unconditional checkpoint")
    data = _load_inference_data(args.data, payload, args.time_column)
    initial = torch.as_tensor(data.snapshots[0], device=device, dtype=torch.float32)
    predictions = rollout(
        model,
        initial,
        data.times,
        steps_per_interval=args.steps_per_interval,
    )
    records = []
    initial_count = data.snapshots[0].shape[0]
    for time, observed in zip(data.times[1:], data.snapshots[1:]):
        predicted, mass = predictions[time]
        predicted_np = predicted.detach().cpu().numpy()
        mass_np = mass.detach().cpu().numpy().reshape(-1)
        records.append(
            {
                "time": time,
                "w1": weighted_wasserstein1(predicted_np, observed, mass_np),
                "rme": relative_mass_error(mass_np, initial_count, observed.shape[0]),
                "steps_per_interval": args.steps_per_interval,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(f"Saved metrics: {output.resolve()}")


def _control_frame(
    path: str | Path,
    feature_names: list[str],
    control_label: int,
    condition_column: str,
) -> np.ndarray:
    frame = pd.read_csv(path)
    if condition_column in frame.columns:
        frame = frame.loc[frame[condition_column] == control_label]
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Control file is missing features: {missing}")
    return frame[feature_names].to_numpy(dtype=np.float32)


def command_infer_conditional(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    if not model.condition_dimension:
        raise ValueError("Checkpoint is not conditional")
    condition_column = str(payload["config"]["data"].get("condition_column", "samples"))
    control_label = int(payload["config"]["data"].get("control_label", 0))
    control = _control_frame(
        args.control,
        list(payload["feature_names"]),
        control_label,
        condition_column,
    )
    embedding_frame = pd.read_csv(args.embeddings).set_index(condition_column)
    mean = np.asarray(payload["embedding_mean"], dtype=np.float32)
    std = np.asarray(payload["embedding_std"], dtype=np.float32)
    initial = torch.as_tensor(control, device=device, dtype=torch.float32)
    output_frames = []
    for condition_id in args.conditions:
        if condition_id not in embedding_frame.index:
            raise ValueError(f"Condition {condition_id} is absent from the embedding table")
        raw = embedding_frame.loc[condition_id, payload["embedding_names"]].to_numpy(
            dtype=np.float32
        )
        condition = torch.as_tensor((raw - mean) / std, device=device)
        predictions = rollout(
            model,
            initial,
            (0.0, 1.0),
            steps_per_interval=args.steps_per_interval,
            condition=condition,
        )
        for time, (positions, masses) in predictions.items():
            frame = pd.DataFrame(
                positions.detach().cpu().numpy(), columns=payload["feature_names"]
            )
            frame.insert(0, "weight", masses.detach().cpu().numpy().reshape(-1))
            frame.insert(0, "particle", np.arange(frame.shape[0]))
            frame.insert(0, "time", time)
            frame.insert(0, "condition", condition_id)
            output_frames.append(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(output_frames, ignore_index=True).to_csv(output, index=False)
    print(f"Saved predictions: {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wfr-mfm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train an unconditional WFR-MFM model")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--hold-out", type=float, nargs="*")
    train.set_defaults(function=command_train)

    conditional = subparsers.add_parser(
        "train-conditional", help="Train the perturbation-conditioned model"
    )
    conditional.add_argument("--config", required=True)
    conditional.add_argument("--output", required=True)
    conditional.add_argument("--device", default="auto")
    conditional.set_defaults(function=command_train_conditional)

    infer = subparsers.add_parser("infer", help="Run unconditional inference")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--data", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--steps-per-interval", type=int, default=1)
    infer.add_argument("--time-column")
    infer.add_argument("--device", default="auto")
    infer.set_defaults(function=command_infer)

    evaluate = subparsers.add_parser("evaluate", help="Compute W1 and RME")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--steps-per-interval", type=int, default=1)
    evaluate.add_argument("--time-column")
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(function=command_evaluate)

    infer_conditional = subparsers.add_parser(
        "infer-conditional", help="Infer one or more perturbation conditions"
    )
    infer_conditional.add_argument("--checkpoint", required=True)
    infer_conditional.add_argument("--control", required=True)
    infer_conditional.add_argument("--embeddings", required=True)
    infer_conditional.add_argument("--conditions", type=int, nargs="+", required=True)
    infer_conditional.add_argument("--output", required=True)
    infer_conditional.add_argument("--steps-per-interval", type=int, default=1)
    infer_conditional.add_argument("--device", default="auto")
    infer_conditional.set_defaults(function=command_infer_conditional)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
