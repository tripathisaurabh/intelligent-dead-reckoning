from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import position_errors, trajectory_metrics
from src.ins.iekf import IEKFInput, IEKFNoise, run_iekf
from src.preprocessing.alignment import estimate_mounting_alignment
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu


def _noise_from_config(config: dict) -> IEKFNoise:
    process = config["process_covariance"]
    measurement = config["measurement_covariance"]
    initial = config["initial_covariance"]
    return IEKFNoise(
        gyro=process["gyro"],
        accelerometer=process["accelerometer"],
        gyro_bias=process["gyro_bias"],
        accelerometer_bias=process["accelerometer_bias"],
        car_to_imu_rotation=process["car_to_imu_rotation"],
        car_to_imu_translation=process["car_to_imu_translation"],
        lateral_velocity=measurement["lateral_velocity"],
        upward_velocity=measurement["upward_velocity"],
        initial_roll_pitch=initial["roll_pitch"],
        initial_velocity=initial["velocity"],
        initial_gyro_bias=initial["gyro_bias"],
        initial_accelerometer_bias=initial["accelerometer_bias"],
        initial_car_to_imu_rotation=initial["car_to_imu_rotation"],
        initial_car_to_imu_translation=initial["car_to_imu_translation"],
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rotation_change_deg(initial: np.ndarray, final: np.ndarray) -> float:
    cosine = np.clip((np.trace(final @ initial.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    baseline_config = yaml.safe_load(
        Path(experiment["baseline_config"]).read_text(encoding="utf-8")
    )
    iekf_config = yaml.safe_load(
        Path(experiment["iekf_config"]).read_text(encoding="utf-8")
    )
    base_experiment = baseline_config["experiment"]
    drive = load_synchronized_drive(
        base_experiment["smartphone_csv"],
        base_experiment["reference_csv"],
        maximum_gap_s=baseline_config["validation"]["maximum_gap_s"],
    )
    biases = json.loads(Path(experiment["baseline_biases"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(experiment["baseline_metadata"]).read_text(encoding="utf-8"))
    gyro_bias_ypr = np.asarray(biases["gyro_ypr_bias_rps"], dtype=float)
    gyro_bias_body = gyro_bias_ypr[[2, 1, 0]]
    gyro_body_full = drive.gyro_ypr_rps[:, [2, 1, 0]]

    alignment = estimate_mounting_alignment(
        drive.time_s,
        drive.gravity_device_mps2,
        gyro_body_full - gyro_bias_body,
        **config["alignment"],
    )
    if alignment.end_index_exclusive > int(metadata["start_index"]):
        raise AssertionError("mounting alignment leaks into the blackout")

    start = int(metadata["start_index"])
    end = int(metadata["end_index_inclusive"])
    selection = slice(start, end + 1)
    time = drive.time_s[selection] - drive.time_s[start]
    reference_enu = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )
    result = run_iekf(
        IEKFInput(
            time_s=time,
            gyro_body_rps=gyro_body_full[selection],
            specific_force_body_mps2=drive.accel_device_mps2[selection],
            initial_speed_mps=float(drive.reference_speed_mps[start]),
            initial_bearing_deg=float(drive.reference_heading_deg[start]),
            initial_gyro_bias_body_rps=gyro_bias_body,
            initial_accel_bias_body_mps2=np.asarray(
                biases["residual_accel_device_bias_mps2"], dtype=float
            ),
            initial_rotation_car_to_imu=alignment.rotation_car_to_imu,
        ),
        _noise_from_config(iekf_config["iekf"]),
    )

    metrics = trajectory_metrics(result.position_enu_m, reference_enu)
    metrics.update(
        {
            "method": "fixed_covariance_iekf_with_gravity_mounting_alignment",
            "blackout_start_time_s": float(metadata["start_time_s"]),
            "blackout_end_time_s": float(metadata["end_time_s"]),
            "blackout_duration_s": float(metadata["duration_s"]),
            "samples": len(time),
        }
    )
    previous = json.loads(Path(experiment["iekf_comparison"]).read_text(encoding="utf-8"))
    baseline_metrics = previous["baseline"]
    iekf_metrics = previous["iekf"]
    refinement = {
        "rotation_change_deg": _rotation_change_deg(
            result.rotation_car_to_imu[0], result.rotation_car_to_imu[-1]
        ),
        "translation_change_m": float(
            np.linalg.norm(
                result.translation_car_to_imu_m[-1] - result.translation_car_to_imu_m[0]
            )
        ),
        "initial_rotation_car_to_imu": result.rotation_car_to_imu[0],
        "final_rotation_car_to_imu": result.rotation_car_to_imu[-1],
        "initial_translation_car_to_imu_m": result.translation_car_to_imu_m[0],
        "final_translation_car_to_imu_m": result.translation_car_to_imu_m[-1],
    }
    comparison = {
        "baseline": baseline_metrics,
        "iekf": iekf_metrics,
        "iekf_with_alignment": metrics,
        "change_from_step2": {
            "final_position_error_m": metrics["final_position_error_m"]
            - iekf_metrics["final_position_error_m"],
            "position_rmse_m": metrics["position_rmse_m"] - iekf_metrics["position_rmse_m"],
            "drift_percentage_points": metrics["drift_percent"] - iekf_metrics["drift_percent"],
        },
        "alignment": asdict(alignment),
        "online_extrinsic_refinement": refinement,
        "interpretation": (
            "Vta08 was already essentially level. The estimated sub-0.01-degree tilt "
            "leaves Step 2 performance flat, with a small numerical worsening."
        ),
    }

    outputs = config["outputs"]
    experiment_dir = Path(outputs["experiment_dir"])
    metrics_path = Path(outputs["metrics"])
    plots_dir = Path(outputs["plots_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([baseline_metrics, iekf_metrics, metrics]).to_csv(metrics_path, index=False)
    (experiment_dir / "comparison.json").write_text(
        json.dumps(_jsonable(comparison), indent=2), encoding="utf-8"
    )
    (experiment_dir / "alignment_estimate.json").write_text(
        json.dumps(_jsonable(asdict(alignment) | {"online_refinement": refinement}), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_since_blackout_s": time,
            "east_m": result.position_enu_m[:, 0],
            "north_m": result.position_enu_m[:, 1],
            "up_m": result.position_enu_m[:, 2],
            "velocity_east_mps": result.velocity_enu_mps[:, 0],
            "velocity_north_mps": result.velocity_enu_mps[:, 1],
            "velocity_up_mps": result.velocity_enu_mps[:, 2],
            "lateral_velocity_mps": result.lateral_upward_velocity_mps[:, 0],
            "upward_velocity_car_mps": result.lateral_upward_velocity_mps[:, 1],
        }
    ).to_csv(experiment_dir / "aligned_iekf_trajectory.csv", index=False)

    baseline = pd.read_csv(experiment["baseline_trajectory"])
    iekf = pd.read_csv(experiment["iekf_trajectory"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(reference_enu[:, 0], reference_enu[:, 1], linewidth=2.2, label="VBOX reference")
    ax.plot(baseline["east_m"], baseline["north_m"], linewidth=1.3, label="Step 1 strapdown")
    ax.plot(iekf["east_m"], iekf["north_m"], linewidth=1.5, label="Step 2 IEKF")
    ax.plot(
        result.position_enu_m[:, 0],
        result.position_enu_m[:, 1],
        linewidth=1.7,
        linestyle="--",
        label="Step 3 IEKF + alignment",
    )
    ax.scatter([0], [0], label="Blackout start", zorder=5)
    ax.set(
        title="Vta08 Blackout: Mounting Alignment Comparison",
        xlabel="East (m)",
        ylabel="North (m)",
    )
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "trajectory_comparison.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time,
        position_errors(baseline[["east_m", "north_m", "up_m"]].to_numpy(), reference_enu),
        label="Step 1 strapdown",
    )
    ax.plot(
        time,
        position_errors(iekf[["east_m", "north_m", "up_m"]].to_numpy(), reference_enu),
        label="Step 2 IEKF",
    )
    ax.plot(time, position_errors(result.position_enu_m, reference_enu), label="Step 3 aligned IEKF")
    ax.set(
        title="Position Error During Identical GNSS Blackout",
        xlabel="Time since blackout (s)",
        ylabel="Horizontal error (m)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "position_error_comparison.png", dpi=240)
    plt.close(fig)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 3 mounting alignment on Vta08")
    parser.add_argument("--config", type=Path, default=Path("configs/alignment_vta08.yaml"))
    args = parser.parse_args()
    print(json.dumps(_jsonable(run(args.config)), indent=2))


if __name__ == "__main__":
    main()
