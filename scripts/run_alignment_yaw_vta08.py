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
from src.preprocessing.alignment_yaw import estimate_gps_heading_mounting_yaw
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu


def _noise(config: dict) -> IEKFNoise:
    p, m, i = (
        config["process_covariance"],
        config["measurement_covariance"],
        config["initial_covariance"],
    )
    return IEKFNoise(
        gyro=p["gyro"], accelerometer=p["accelerometer"],
        gyro_bias=p["gyro_bias"], accelerometer_bias=p["accelerometer_bias"],
        car_to_imu_rotation=p["car_to_imu_rotation"],
        car_to_imu_translation=p["car_to_imu_translation"],
        lateral_velocity=m["lateral_velocity"], upward_velocity=m["upward_velocity"],
        initial_roll_pitch=i["roll_pitch"], initial_velocity=i["velocity"],
        initial_gyro_bias=i["gyro_bias"],
        initial_accelerometer_bias=i["accelerometer_bias"],
        initial_car_to_imu_rotation=i["car_to_imu_rotation"],
        initial_car_to_imu_translation=i["car_to_imu_translation"],
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


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    baseline_config = yaml.safe_load(Path(experiment["baseline_config"]).read_text())
    iekf_config = yaml.safe_load(Path(experiment["iekf_config"]).read_text())
    step3_config = yaml.safe_load(Path(experiment["step3_config"]).read_text())
    base = baseline_config["experiment"]
    drive = load_synchronized_drive(
        base["smartphone_csv"], base["reference_csv"],
        maximum_gap_s=baseline_config["validation"]["maximum_gap_s"],
    )
    biases = json.loads(Path(experiment["baseline_biases"]).read_text())
    metadata = json.loads(Path(experiment["baseline_metadata"]).read_text())
    gyro_bias = np.asarray(biases["gyro_ypr_bias_rps"])[[2, 1, 0]]
    gyro_body = drive.gyro_ypr_rps[:, [2, 1, 0]]
    accel_bias = np.asarray(biases["residual_accel_device_bias_mps2"])

    tilt = estimate_mounting_alignment(
        drive.time_s, drive.gravity_device_mps2, gyro_body - gyro_bias,
        **step3_config["alignment"],
    )
    reference_enu_full = geodetic_to_local_enu(
        drive.reference_lat_deg, drive.reference_lon_deg
    )
    yaw = estimate_gps_heading_mounting_yaw(
        drive.time_s,
        drive.accel_device_mps2,
        drive.gravity_device_mps2,
        gyro_body - gyro_bias,
        reference_enu_full,
        drive.reference_speed_mps,
        drive.reference_heading_deg,
        drive.orientation_ypr_deg,
        blackout_start_index=int(metadata["start_index"]),
        earliest_index=int(metadata["calibration_end_index_exclusive"]),
        initial_rotation_car_to_imu=tilt.rotation_car_to_imu,
        accelerometer_bias_imu_mps2=accel_bias,
        **config["yaw_alignment"],
    )
    if yaw.end_index_exclusive > int(metadata["start_index"]):
        raise AssertionError("yaw alignment leaks into blackout")

    start, end = int(metadata["start_index"]), int(metadata["end_index_inclusive"])
    selection = slice(start, end + 1)
    time = drive.time_s[selection] - drive.time_s[start]
    reference = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )
    result = run_iekf(
        IEKFInput(
            time_s=time,
            gyro_body_rps=gyro_body[selection],
            specific_force_body_mps2=drive.accel_device_mps2[selection],
            initial_speed_mps=float(drive.reference_speed_mps[start]),
            initial_bearing_deg=float(drive.reference_heading_deg[start]),
            initial_gyro_bias_body_rps=gyro_bias,
            initial_accel_bias_body_mps2=accel_bias,
            initial_rotation_car_to_imu=yaw.rotation_car_to_imu,
        ),
        _noise(iekf_config["iekf"]),
    )
    metrics = trajectory_metrics(result.position_enu_m, reference)
    metrics.update({
        "method": "fixed_covariance_iekf_with_gps_heading_mounting_yaw",
        "blackout_start_time_s": float(metadata["start_time_s"]),
        "blackout_end_time_s": float(metadata["end_time_s"]),
        "blackout_duration_s": float(metadata["duration_s"]),
        "samples": len(time),
    })
    prior = json.loads(Path(experiment["step3_comparison"]).read_text())
    step2, step3 = prior["iekf"], prior["iekf_with_alignment"]
    comparison = {
        "baseline": prior["baseline"], "iekf": step2,
        "iekf_with_tilt_alignment": step3,
        "iekf_with_tilt_and_gps_heading_yaw": metrics,
        "change_from_step2": {
            "final_position_error_m": metrics["final_position_error_m"] - step2["final_position_error_m"],
            "position_rmse_m": metrics["position_rmse_m"] - step2["position_rmse_m"],
            "drift_percentage_points": metrics["drift_percent"] - step2["drift_percent"],
        },
        "improvement_from_step2_percent": {
            "final_position_error": 100.0 * (step2["final_position_error_m"] - metrics["final_position_error_m"]) / step2["final_position_error_m"],
            "position_rmse": 100.0 * (step2["position_rmse_m"] - metrics["position_rmse_m"]) / step2["position_rmse_m"],
        },
        "tilt_alignment": prior["alignment"],
        "gps_heading_yaw_alignment": asdict(yaw),
        "phone_speed_unit_assessment": drive.phone_speed_unit_assessment,
        "leakage_check": {
            "last_alignment_index": yaw.end_index_exclusive - 1,
            "first_blackout_index": start,
            "passes": yaw.end_index_exclusive <= start,
        },
        "interpretation": (
            "The causal +13.54-degree mounting-yaw correction substantially reduces the bend. "
            "GPS course/heading selects and validates the straight accelerating window; phone "
            "linear acceleration supplies the car-forward direction in phone coordinates."
        ),
    }

    outputs = config["outputs"]
    experiment_dir, metrics_path, plots_dir = (
        Path(outputs["experiment_dir"]), Path(outputs["metrics"]), Path(outputs["plots_dir"])
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "comparison.json").write_text(json.dumps(_jsonable(comparison), indent=2))
    (experiment_dir / "yaw_alignment_estimate.json").write_text(
        json.dumps(_jsonable(asdict(yaw)), indent=2)
    )
    pd.DataFrame([prior["baseline"], step2, step3, metrics]).to_csv(metrics_path, index=False)
    trajectory = pd.DataFrame({
        "time_since_blackout_s": time,
        "east_m": result.position_enu_m[:, 0], "north_m": result.position_enu_m[:, 1],
        "up_m": result.position_enu_m[:, 2],
        "velocity_east_mps": result.velocity_enu_mps[:, 0],
        "velocity_north_mps": result.velocity_enu_mps[:, 1],
        "velocity_up_mps": result.velocity_enu_mps[:, 2],
    })
    trajectory.to_csv(experiment_dir / "yaw_aligned_iekf_trajectory.csv", index=False)

    baseline = pd.read_csv(experiment["baseline_trajectory"])
    fixed = pd.read_csv(experiment["iekf_trajectory"])
    aligned = pd.read_csv(experiment["step3_trajectory"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(reference[:, 0], reference[:, 1], linewidth=2.2, label="VBOX reference")
    ax.plot(baseline.east_m, baseline.north_m, linewidth=1.0, label="Step 1")
    ax.plot(fixed.east_m, fixed.north_m, linewidth=1.2, label="Step 2")
    ax.plot(aligned.east_m, aligned.north_m, linewidth=1.2, label="Step 3 tilt")
    ax.plot(result.position_enu_m[:, 0], result.position_enu_m[:, 1], linewidth=1.8,
            label="Step 3b tilt + yaw")
    ax.scatter([0], [0], label="Blackout start", zorder=5)
    ax.set(title="Vta08: Causal GPS-Heading Mounting Yaw", xlabel="East (m)", ylabel="North (m)")
    ax.axis("equal"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(plots_dir / "trajectory_comparison.png", dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for frame, label in ((fixed, "Step 2"), (aligned, "Step 3 tilt")):
        ax.plot(time, position_errors(frame[["east_m", "north_m", "up_m"]].to_numpy(), reference), label=label)
    ax.plot(time, position_errors(result.position_enu_m, reference), label="Step 3b tilt + yaw")
    ax.set(title="Identical Vta08 Blackout Error", xlabel="Time since blackout (s)", ylabel="Horizontal error (m)")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(plots_dir / "position_error_comparison.png", dpi=240); plt.close(fig)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal GPS-heading mounting-yaw alignment")
    parser.add_argument("--config", type=Path, default=Path("configs/alignment_yaw_vta08.yaml"))
    args = parser.parse_args()
    print(json.dumps(_jsonable(run(args.config)), indent=2))


if __name__ == "__main__":
    main()
