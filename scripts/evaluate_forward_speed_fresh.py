from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.models.forward_speed_gru import ForwardSpeedGRU
from src.preprocessing.dataset import load_synchronized_drive
from train_forward_speed import _candidate_starts, _features


def _metrics(estimate: np.ndarray, reference: np.ndarray, time: np.ndarray) -> dict:
    error = estimate - reference
    distance = np.r_[0.0, np.cumsum(0.5 * (estimate[1:] + estimate[:-1]) * np.diff(time))]
    reference_distance = np.r_[0.0, np.cumsum(0.5 * (reference[1:] + reference[:-1]) * np.diff(time))]
    return {
        "speed_mae_mps": float(np.mean(np.abs(error))),
        "speed_rmse_mps": float(np.sqrt(np.mean(error**2))),
        "final_speed_error_mps": float(error[-1]),
        "estimated_distance_m": float(distance[-1]),
        "reference_distance_m": float(reference_distance[-1]),
        "final_distance_error_m": float(distance[-1] - reference_distance[-1]),
    }


def run(config_path: Path, split: str = "fresh_test") -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["experiment"]
    spec = settings[split]
    drive = load_synchronized_drive(spec["smartphone_csv"], spec["reference_csv"])
    length = settings["sequence_samples"]
    candidates = _candidate_starts(
        drive.reference_speed_mps, length, settings["minimum_start_speed_mps"],
        settings["minimum_moving_fraction"],
    )
    if not len(candidates):
        raise ValueError("fresh test session has no qualifying interval")
    # Lock the first qualifying interval before running model inference.
    start = int(candidates[0])
    stop = start + length
    time = drive.time_s[start:stop] - drive.time_s[start]
    reference = drive.reference_speed_mps[start:stop]

    checkpoint_path = Path(config["outputs"]["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ForwardSpeedGRU(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    features = _features(drive, start, length)
    normalized = (features - np.asarray(checkpoint["feature_mean"])) / np.asarray(checkpoint["feature_std"])
    with torch.inference_mode():
        mean, std = model(
            torch.tensor(normalized[None], dtype=torch.float32),
            torch.tensor([features[0, 6]], dtype=torch.float32),
        )
    learned = mean[0].numpy()
    uncertainty = std[0].numpy()
    constant = np.full(length, reference[0])
    lower, upper = learned - 1.96 * uncertainty, learned + 1.96 * uncertainty
    coverage = float(np.mean((reference >= lower) & (reference <= upper)))
    learned_metrics = _metrics(learned, reference, time)
    learned_metrics.update(
        {
            "mean_predicted_std_mps": float(np.mean(uncertainty)),
            "empirical_95_percent_interval_coverage": coverage,
        }
    )
    baseline_metrics = _metrics(constant, reference, time)

    experiment_dir = Path(config["outputs"]["experiment_dir"])
    plots_dir = Path(config["outputs"]["plots_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(
        {
            "time_s": time,
            "reference_speed_mps": reference,
            "constant_speed_baseline_mps": constant,
            "learned_mean_speed_mps": learned,
            "learned_std_mps": uncertainty,
            "learned_95_lower_mps": lower,
            "learned_95_upper_mps": upper,
        }
    )
    trace.to_csv(experiment_dir / f"fresh_{spec['name']}_trace.csv", index=False)
    report = {
        "model": "ForwardSpeedGRU with Gaussian mean and standard deviation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data_split": {
            "train": [item["name"] for item in settings["train"]],
            "validation": settings["validation"]["name"],
            "evaluated_split": split,
            "evaluated_session": spec["name"],
            "fresh_test_influenced_training_or_tuning": False if split == "fresh_test" else None,
        },
        "locked_fresh_interval": {
            "selection": "first qualifying 20-second interval; chosen before inference",
            "start_index": start,
            "end_index_inclusive": stop - 1,
            "start_time_s": float(drive.time_s[start]),
            "end_time_s": float(drive.time_s[stop - 1]),
            "duration_s": float(time[-1]),
        },
        "corrected_safe_baseline": {
            "definition": "Hold the last trusted GNSS speed constant; avoids the unsupported gyroscope body-axis and acceleration projection assumptions found by the audit.",
            **baseline_metrics,
        },
        "learned_forward_speed": learned_metrics,
        "change_from_safe_baseline": {
            "speed_mae_mps": learned_metrics["speed_mae_mps"] - baseline_metrics["speed_mae_mps"],
            "speed_rmse_mps": learned_metrics["speed_rmse_mps"] - baseline_metrics["speed_rmse_mps"],
            "absolute_final_distance_error_m": abs(learned_metrics["final_distance_error_m"]) - abs(baseline_metrics["final_distance_error_m"]),
        },
    }
    (experiment_dir / f"evaluation_{spec['name']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    learned_distance = np.r_[0.0, np.cumsum(0.5 * (learned[1:] + learned[:-1]) * np.diff(time))]
    constant_distance = np.r_[0.0, np.cumsum(0.5 * (constant[1:] + constant[:-1]) * np.diff(time))]
    reference_distance = np.r_[0.0, np.cumsum(0.5 * (reference[1:] + reference[:-1]) * np.diff(time))]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time, reference, color="black", linewidth=2, label="S1 reference")
    axes[0].plot(time, constant, color="0.45", linestyle="--", label="Safe constant-speed baseline")
    axes[0].plot(time, learned, color="#0072B2", linewidth=2, label="Learned mean")
    axes[0].fill_between(time, lower, upper, color="#56B4E9", alpha=.25, label="Learned 95% interval")
    axes[0].set(ylabel="Speed (m/s)", title=f"{spec['name']} forward-speed test ({split})"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(time, reference_distance, color="black", linewidth=2, label="Reference")
    axes[1].plot(time, constant_distance, color="0.45", linestyle="--", label="Safe baseline")
    axes[1].plot(time, learned_distance, color="#0072B2", linewidth=2, label="Learned")
    axes[1].set(xlabel="Time (s)", ylabel="Integrated distance (m)", title="Speed-only distance consequence"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plots_dir / f"fresh_{spec['name']}_speed_and_distance.png", dpi=240); plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/forward_speed_v1.yaml"))
    parser.add_argument("--split", choices=("development_test", "fresh_test"), default="fresh_test")
    args = parser.parse_args(); print(json.dumps(run(args.config, args.split), indent=2))


if __name__ == "__main__": main()
