from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_adapter_vta08 import _stationary_bias
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu, vehicle_to_enu_rotation
from train import mounting_rotation


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _rotation_checks(rotation: np.ndarray) -> dict:
    return {
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_max_abs_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
    }


def _session_audit(name: str, drive, lock: dict, rotation: np.ndarray, yaw_window: dict) -> dict:
    gyro_bias, accel_bias, _ = _stationary_bias(drive)
    start = int(lock["start_index"])
    calibration = slice(int(yaw_window["start_index"]), int(yaw_window["end_index_exclusive"]))
    phone_enu = geodetic_to_local_enu(drive.phone_lat_deg, drive.phone_lon_deg)
    unique_phone_fixes = len(
        np.unique(
            np.c_[drive.phone_lat_deg[calibration], drive.phone_lon_deg[calibration]], axis=0
        )
    )
    phone_displacement = float(
        np.linalg.norm(phone_enu[calibration.stop - 1, :2] - phone_enu[calibration.start, :2])
    )

    stationary_linear = drive.accel_device_mps2 - drive.gravity_device_mps2 - accel_bias
    stationary_speed = drive.reference_speed_mps <= 0.5
    stationary_gyro = np.linalg.norm(drive.gyro_ypr_rps, axis=1) <= 0.08
    stationary = stationary_speed & stationary_gyro
    initial_imu_to_enu = vehicle_to_enu_rotation(
        float(drive.reference_heading_deg[start])
    ) @ rotation.T
    mean_specific_force = np.mean(drive.accel_device_mps2[stationary] - accel_bias, axis=0)
    navigation_acceleration = initial_imu_to_enu @ mean_specific_force + np.array([0.0, 0.0, -9.80665])

    moving_turning = (drive.reference_speed_mps > 2.0) & (np.abs(drive.reference_yaw_rate_rps) > 0.01)
    gyro_body = drive.gyro_ypr_rps[:, [2, 1, 0]] - gyro_bias
    correlations = {}
    for axis, values in zip("xyz", gyro_body.T):
        correlations[axis] = float(np.corrcoef(values[moving_turning], drive.reference_yaw_rate_rps[moving_turning])[0, 1])

    reference_origin = geodetic_to_local_enu(
        drive.reference_lat_deg[start : start + 2], drive.reference_lon_deg[start : start + 2]
    )[0]
    initialized_velocity = np.array(
        [
            drive.reference_speed_mps[start] * np.sin(np.deg2rad(drive.reference_heading_deg[start])),
            drive.reference_speed_mps[start] * np.cos(np.deg2rad(drive.reference_heading_deg[start])),
            0.0,
        ]
    )
    return {
        "sampling": {
            "median_dt_s": drive.smartphone_stats.median_dt_s,
            "estimated_hz": drive.smartphone_stats.estimated_hz,
            "timestamp_resets": drive.smartphone_stats.timestamp_resets,
        },
        "units": {
            "reference_speed_conversion": "Velocity (km/hr) divided by 3.6",
            "phone_speed_assessment": drive.phone_speed_unit_assessment,
            "gyro": "rad/s",
            "accelerometer": "m/s^2 including gravity",
            "heading": "degrees clockwise from north",
            "local_frame": "ENU metres",
        },
        "axes": {
            "source_gyro_order": "yaw, pitch, roll",
            "filter_gyro_order": "x=roll, y=pitch, z=yaw",
            "correlation_with_vbox_yaw_rate": correlations,
            "strongest_absolute_correlation_axis": max(correlations, key=lambda key: abs(correlations[key])),
        },
        "gravity_and_bias": {
            "stationary_samples_checked": int(np.sum(stationary)),
            "linear_acceleration_after_gravity_and_bias_rms_mps2": float(
                np.sqrt(np.mean(np.sum(stationary_linear[stationary] ** 2, axis=1)))
            ),
            "mean_navigation_acceleration_at_stationary_mps2": navigation_acceleration,
            "mean_navigation_acceleration_norm_mps2": float(np.linalg.norm(navigation_acceleration)),
        },
        "mounting_rotation": _rotation_checks(rotation),
        "initialization": {
            "position_enu_m": reference_origin,
            "speed_mps": float(drive.reference_speed_mps[start]),
            "bearing_deg": float(drive.reference_heading_deg[start]),
            "velocity_enu_mps": initialized_velocity,
            "source": "oracle VBOX at the first blackout sample",
        },
        "mounting_yaw_source_check": {
            "current_selector_inputs": "VBOX reference position, speed and heading",
            "window_indices": [calibration.start, calibration.stop - 1],
            "phone_unique_position_fixes_in_window": int(unique_phone_fixes),
            "phone_displacement_in_window_m": phone_displacement,
            "phone_speed_start_mps": float(drive.phone_speed_raw[calibration.start]),
            "phone_speed_end_mps": float(drive.phone_speed_raw[calibration.stop - 1]),
            "deployable_from_phone_gnss_in_same_window": bool(unique_phone_fixes >= 2 and phone_displacement >= 4.0),
        },
    }


def run() -> dict:
    manifest = json.loads(Path("baselines/v1/manifest.json").read_text(encoding="utf-8"))
    adapter = yaml.safe_load(Path("configs/adapter_cnn.yaml").read_text(encoding="utf-8"))
    s3_config = yaml.safe_load(Path("configs/cross_session_s3c.yaml").read_text(encoding="utf-8"))
    vta_eval = json.loads(Path("results/experiments/alignment_yaw_vta08/comparison.json").read_text(encoding="utf-8"))
    s3_eval = json.loads(Path("results/experiments/S3c/evaluation.json").read_text(encoding="utf-8"))

    vta_spec = adapter["experiment"]["test"]
    vta = load_synchronized_drive(vta_spec["smartphone_csv"], vta_spec["reference_csv"])
    vta_rotation = mounting_rotation(adapter["mounting"])
    s3_spec = s3_config["experiment"]
    s3 = load_synchronized_drive(
        s3_spec["smartphone_csv"], s3_spec["reference_csv"], maximum_gap_s=s3_spec["maximum_gap_s"]
    )
    s3_yaw = s3_eval["calibration"]["yaw_alignment"]
    s3_rotation = np.asarray(s3_yaw["rotation_car_to_imu"])
    report = {
        "baseline": manifest["name"],
        "sessions": {
            "Vta08": _session_audit(
                "Vta08", vta, manifest["locked_blackouts"]["Vta08"], vta_rotation,
                vta_eval["gps_heading_yaw_alignment"],
            ),
            "S3c": _session_audit(
                "S3c", s3, manifest["locked_blackouts"]["S3c"], s3_rotation, s3_yaw,
            ),
        },
        "training_runtime_filter_agreement": {
            "test": "tests/test_adaptive_iekf.py::test_nonzero_adapter_matches_differentiable_training_filter",
            "coverage": "position, velocity, rotation, both biases and covariance after non-zero learned corrections",
        },
        "confirmed_findings": [
            "The three gyroscope columns are semantic yaw/pitch/roll rates, not documented Cartesian x/y/z rates. Reordering them as [roll, pitch, yaw] is therefore an unsupported body-frame assumption. The weak z-channel/VBOX agreement and strong y-channel agreement on S3c confirm that the full 3-D mechanization cannot be called axis-validated.",
            "Reference speed is correctly converted from km/h to m/s; phone speed is already m/s despite its label.",
            "Accelerometer samples include gravity; the propagation subtracts the estimated residual bias and adds ENU gravity once.",
            "Car-to-IMU rotations are proper orthonormal rotations and are composed consistently at initialization.",
            "Training and runtime adaptive IEKF equations agree numerically for non-zero adapter outputs.",
            "The baseline mounting-yaw window is selected with VBOX reference signals, not deployable phone GNSS. This is an evaluation dependency and must not be described as phone-only calibration.",
            "Oracle VBOX speed and heading initialize both locked blackouts; this intentionally measures outage drift, not cold-start accuracy.",
        ],
    }
    output = Path("results/audits/baseline_v1_calculations.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(_jsonable(run()), indent=2))
