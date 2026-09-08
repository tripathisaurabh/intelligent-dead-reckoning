from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.run_adapter_vta08 import _load_model, _model_outputs, _noise, _stationary_bias
from src.ins.adaptive_iekf import run_adaptive_iekf
from src.ins.iekf import IEKFInput
from src.preprocessing.alignment import estimate_mounting_alignment
from src.preprocessing.alignment_yaw import estimate_gps_heading_mounting_yaw
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu
from train import mounting_rotation


def _wrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _distance(position: np.ndarray) -> np.ndarray:
    increments = np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1)
    return np.r_[0.0, np.cumsum(increments)]


def _vehicle_heading(result) -> np.ndarray:
    car_to_enu = result.rotation_imu_to_enu @ result.rotation_car_to_imu
    forward_enu = car_to_enu[:, :, 0]
    return np.rad2deg(np.arctan2(forward_enu[:, 0], forward_enu[:, 1])) % 360.0


def _first_sustained(time: np.ndarray, condition: np.ndarray, duration_s: float = 1.0):
    for start in range(len(time)):
        stop = int(np.searchsorted(time, time[start] + duration_s, side="left"))
        if stop < len(time) and np.all(condition[start : stop + 1]):
            return float(time[start])
    return None


def _run_adaptive(drive, start, end, gyro_bias, accel_bias, car_rotation, model, context, noise):
    selection = slice(start, end + 1)
    indices = np.arange(start, end + 1)
    time = drive.time_s[selection] - drive.time_s[start]
    gyro = drive.gyro_ypr_rps[:, [2, 1, 0]]
    data = IEKFInput(
        time_s=time,
        gyro_body_rps=gyro[selection],
        specific_force_body_mps2=drive.accel_device_mps2[selection],
        initial_speed_mps=float(drive.reference_speed_mps[start]),
        initial_bearing_deg=float(drive.reference_heading_deg[start]),
        initial_gyro_bias_body_rps=gyro_bias,
        initial_accel_bias_body_mps2=accel_bias,
        initial_rotation_car_to_imu=car_rotation,
    )
    outputs = _model_outputs(
        model, gyro, drive.accel_device_mps2, gyro_bias, accel_bias, indices, context
    )
    return time, run_adaptive_iekf(data, outputs, noise)


def _diagnose(name: str, drive, start: int, end: int, result, output_root: Path) -> dict:
    selection = slice(start, end + 1)
    time = result.time_s
    reference_position = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )
    estimated_speed = np.linalg.norm(result.velocity_enu_mps[:, :2], axis=1)
    reference_speed = drive.reference_speed_mps[selection]
    estimated_distance = _distance(result.position_enu_m)
    reference_distance = _distance(reference_position)
    estimated_heading = _vehicle_heading(result)
    reference_heading = drive.reference_heading_deg[selection] % 360.0
    heading_error = _wrap_degrees(estimated_heading - reference_heading)
    position_error = np.linalg.norm(
        result.position_enu_m[:, :2] - reference_position[:, :2], axis=1
    )
    speed_error = estimated_speed - reference_speed
    distance_error = estimated_distance - reference_distance
    safe_speed = np.full_like(reference_speed, reference_speed[0])
    safe_distance = np.r_[
        0.0,
        np.cumsum(0.5 * (safe_speed[1:] + safe_speed[:-1]) * np.diff(time)),
    ]

    # These thresholds are diagnostic markers, not accuracy claims. Requiring a
    # full second avoids declaring failure from one noisy reference sample.
    first_speed = _first_sustained(time, np.abs(speed_error) > 2.0)
    first_heading = _first_sustained(time, np.abs(heading_error) > 10.0)
    first_position = _first_sustained(time, position_error > 5.0)
    if first_speed is None and first_heading is None:
        first_failure = "neither threshold was sustained"
    elif first_heading is None or (first_speed is not None and first_speed < first_heading):
        first_failure = "speed"
    elif first_speed is None or first_heading < first_speed:
        first_failure = "direction"
    else:
        first_failure = "speed and direction together"

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(
        {
            "time_since_blackout_s": time,
            "estimated_speed_mps": estimated_speed,
            "reference_speed_mps": reference_speed,
            "estimated_distance_m": estimated_distance,
            "reference_distance_m": reference_distance,
            "estimated_heading_deg": estimated_heading,
            "reference_heading_deg": reference_heading,
            "signed_heading_error_deg": heading_error,
            "position_error_m": position_error,
        }
    )
    trace.to_csv(output_dir / "diagnostic_trace.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(time, reference_speed, color="black", linewidth=2, label="Reference")
    axes[0, 0].plot(time, estimated_speed, color="#d95f02", linewidth=1.8, label="CNN-IEKF")
    axes[0, 0].set(ylabel="Speed (m/s)", title="Speed")
    axes[0, 0].legend()
    axes[0, 1].plot(time, reference_distance, color="black", linewidth=2, label="Reference")
    axes[0, 1].plot(time, estimated_distance, color="#7570b3", linewidth=1.8, label="CNN-IEKF")
    axes[0, 1].set(ylabel="Distance travelled (m)", title="Cumulative distance")
    axes[0, 1].legend()
    reference_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(reference_heading)))
    estimated_unwrapped = reference_unwrapped[0] + np.rad2deg(
        np.unwrap(np.deg2rad(_wrap_degrees(estimated_heading - reference_heading[0])))
    )
    axes[1, 0].plot(time, reference_unwrapped, color="black", linewidth=2, label="Reference")
    axes[1, 0].plot(time, estimated_unwrapped, color="#1b9e77", linewidth=1.8, label="CNN-IEKF")
    axes[1, 0].set(xlabel="Time since blackout (s)", ylabel="Heading (deg)", title="Vehicle heading")
    axes[1, 0].legend()
    axes[1, 1].plot(time, position_error, color="#e7298a", linewidth=2)
    axes[1, 1].axhline(5.0, color="0.4", linestyle="--", linewidth=1, label="5 m marker")
    axes[1, 1].set(xlabel="Time since blackout (s)", ylabel="Horizontal error (m)", title="Position error")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(f"{name} Baseline v1 — map matching disabled", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "kinematics_diagnostic.png", dpi=240)
    plt.close(figure)

    summary = {
        "session": name,
        "map_matching": "disabled",
        "blackout": {
            "start_index": start,
            "end_index_inclusive": end,
            "start_time_s": float(drive.time_s[start]),
            "end_time_s": float(drive.time_s[end]),
            "duration_s": float(time[-1]),
        },
        "threshold_definition": {
            "persistence_s": 1.0,
            "speed_absolute_error_mps": 2.0,
            "heading_absolute_error_deg": 10.0,
            "position_error_m": 5.0,
        },
        "first_sustained_threshold_crossing_s": {
            "speed": first_speed,
            "heading": first_heading,
            "position": first_position,
        },
        "first_failure": first_failure,
        "metrics": {
            "speed_mae_mps": float(np.mean(np.abs(speed_error))),
            "speed_final_error_mps": float(speed_error[-1]),
            "distance_final_error_m": float(distance_error[-1]),
            "heading_mae_deg": float(np.mean(np.abs(heading_error))),
            "heading_final_error_deg": float(heading_error[-1]),
            "position_final_error_m": float(position_error[-1]),
            "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        },
        "corrected_axis_independent_forward_baseline": {
            "definition": "Hold the trusted blackout-entry speed constant; no unsupported gyro-axis or acceleration projection is used.",
            "speed_mae_mps": float(np.mean(np.abs(safe_speed - reference_speed))),
            "speed_rmse_mps": float(np.sqrt(np.mean((safe_speed - reference_speed) ** 2))),
            "final_speed_error_mps": float(safe_speed[-1] - reference_speed[-1]),
            "estimated_distance_m": float(safe_distance[-1]),
            "reference_distance_m": float(reference_distance[-1]),
            "final_distance_error_m": float(safe_distance[-1] - reference_distance[-1]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(output_root: Path) -> dict:
    manifest = json.loads(Path("baselines/v1/manifest.json").read_text(encoding="utf-8"))
    adapter_config = yaml.safe_load(Path("configs/adapter_cnn.yaml").read_text(encoding="utf-8"))
    s3c_config = yaml.safe_load(Path("configs/cross_session_s3c.yaml").read_text(encoding="utf-8"))
    model = _load_model(Path(manifest["model"]["checkpoint"]))
    context = adapter_config["model"]["context_samples"]
    noise = _noise(yaml.safe_load(Path("configs/iekf_vta08.yaml").read_text(encoding="utf-8"))["iekf"])

    vta_spec = adapter_config["experiment"]["test"]
    vta = load_synchronized_drive(vta_spec["smartphone_csv"], vta_spec["reference_csv"])
    vta_lock = manifest["locked_blackouts"]["Vta08"]
    biases = json.loads(Path("results/metrics/baseline_vta08_biases.json").read_text(encoding="utf-8"))
    vta_gyro_bias = np.asarray(biases["gyro_ypr_bias_rps"])[[2, 1, 0]]
    vta_accel_bias = np.asarray(biases["residual_accel_device_bias_mps2"])
    _, vta_result = _run_adaptive(
        vta, vta_lock["start_index"], vta_lock["end_index_inclusive"],
        vta_gyro_bias, vta_accel_bias, mounting_rotation(adapter_config["mounting"]),
        model, context, noise,
    )

    s3_spec = s3c_config["experiment"]
    s3c = load_synchronized_drive(
        s3_spec["smartphone_csv"], s3_spec["reference_csv"],
        maximum_gap_s=s3_spec["maximum_gap_s"],
    )
    s3_lock = manifest["locked_blackouts"]["S3c"]
    s3_gyro_bias, s3_accel_bias, calibration_end = _stationary_bias(s3c)
    alignment_settings = yaml.safe_load(Path(s3c_config["inputs"]["alignment_config"]).read_text(encoding="utf-8"))["alignment"]
    yaw_settings = yaml.safe_load(Path(s3c_config["inputs"]["yaw_alignment_config"]).read_text(encoding="utf-8"))["yaw_alignment"]
    gyro_full = s3c.gyro_ypr_rps[:, [2, 1, 0]]
    tilt = estimate_mounting_alignment(
        s3c.time_s, s3c.gravity_device_mps2, gyro_full - s3_gyro_bias, **alignment_settings
    )
    yaw = estimate_gps_heading_mounting_yaw(
        s3c.time_s, s3c.accel_device_mps2, s3c.gravity_device_mps2,
        gyro_full - s3_gyro_bias,
        geodetic_to_local_enu(s3c.reference_lat_deg, s3c.reference_lon_deg),
        s3c.reference_speed_mps, s3c.reference_heading_deg, s3c.orientation_ypr_deg,
        blackout_start_index=s3_lock["start_index"], earliest_index=calibration_end,
        initial_rotation_car_to_imu=tilt.rotation_car_to_imu,
        accelerometer_bias_imu_mps2=s3_accel_bias, **yaw_settings,
    )
    if yaw.end_index_exclusive > s3_lock["start_index"]:
        raise AssertionError("S3c mounting alignment leaked into the blackout")
    _, s3_result = _run_adaptive(
        s3c, s3_lock["start_index"], s3_lock["end_index_inclusive"],
        s3_gyro_bias, s3_accel_bias, yaw.rotation_car_to_imu, model, context, noise,
    )

    report = {
        "baseline": manifest["name"],
        "diagnostic_only": "No parameters were tuned from either locked interval.",
        "sessions": {
            "Vta08": _diagnose("Vta08", vta, vta_lock["start_index"], vta_lock["end_index_inclusive"], vta_result, output_root),
            "S3c": _diagnose("S3c", s3c, s3_lock["start_index"], s3_lock["end_index_inclusive"], s3_result, output_root),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Baseline v1 without map matching")
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/baseline_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
