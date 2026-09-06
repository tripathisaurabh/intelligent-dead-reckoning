from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.blackout import select_blackout_interval
from src.evaluation.metrics import position_errors, trajectory_metrics
from src.ins.strapdown import InsInput, propagate_basic_ins
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu
from src.preprocessing.stationary import (
    estimate_biases_from_stationary_run,
    stationary_mask,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _save_lines(time_s, values, labels, title, ylabel, path, spans=None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for column, label in zip(np.asarray(values).T, labels):
        ax.plot(time_s, column, linewidth=0.8, label=label)
    for start, end, label in spans or []:
        ax.axvspan(start, end, alpha=0.18, label=label)
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _heading_validation(reference_enu: np.ndarray, heading_deg: np.ndarray, speed_mps: np.ndarray) -> dict:
    delta = np.diff(reference_enu[:, :2], axis=0)
    course = np.rad2deg(np.arctan2(delta[:, 0], delta[:, 1])) % 360.0
    reference_heading = heading_deg[1:] % 360.0
    difference = np.abs((course - reference_heading + 180.0) % 360.0 - 180.0)
    usable = (speed_mps[1:] > 5.0) & (np.linalg.norm(delta, axis=1) > 0.2)
    return {
        "convention": "VBOX heading is treated as degrees clockwise from North",
        "moving_samples_compared": int(np.sum(usable)),
        "median_course_heading_difference_deg": float(np.median(difference[usable])),
        "p95_course_heading_difference_deg": float(np.percentile(difference[usable], 95)),
    }


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    validation = config["validation"]
    stationary_config = config["stationary"]
    outputs = config["outputs"]

    experiment_dir = Path(outputs["experiment_dir"])
    metrics_dir = Path(outputs["metrics_dir"])
    plots_dir = Path(outputs["plots_dir"])
    for directory in (experiment_dir, metrics_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    drive = load_synchronized_drive(
        experiment["smartphone_csv"],
        experiment["reference_csv"],
        maximum_gap_s=validation["maximum_gap_s"],
    )
    if drive.smartphone_stats.timestamp_resets or drive.smartphone_stats.gaps_over_limit:
        raise ValueError("Vta08 fails the timestamp contract")

    reference_enu_full = geodetic_to_local_enu(drive.reference_lat_deg, drive.reference_lon_deg)
    stationary = stationary_mask(
        drive.reference_speed_mps,
        drive.gyro_ypr_rps,
        drive.accel_device_mps2,
        gravity_mps2=validation["gravity_mps2"],
        maximum_reference_speed_mps=stationary_config["maximum_reference_speed_mps"],
        maximum_gyro_norm_rps=stationary_config["maximum_gyro_norm_rps"],
        gravity_magnitude_tolerance_mps2=stationary_config["gravity_magnitude_tolerance_mps2"],
    )
    bias = estimate_biases_from_stationary_run(
        drive.time_s,
        drive.accel_device_mps2,
        drive.gravity_device_mps2,
        drive.gyro_ypr_rps,
        stationary,
        minimum_duration_s=stationary_config["minimum_duration_s"],
    )
    blackout = select_blackout_interval(
        drive.time_s,
        drive.reference_speed_mps,
        earliest_index=bias.end_index_exclusive,
        duration_s=experiment["blackout_duration_s"],
        minimum_speed_mps=experiment["minimum_speed_mps"],
        required_moving_fraction=experiment["required_moving_fraction"],
        maximum_gap_s=validation["maximum_gap_s"],
        context_before_s=experiment["context_before_s"],
        context_after_s=experiment["context_after_s"],
    )
    if bias.end_index_exclusive > blackout.start_index:
        raise AssertionError("Calibration data leaks beyond blackout initialization")

    start = blackout.start_index
    stop = blackout.end_index_inclusive + 1
    selection = slice(start, stop)
    reference_enu = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )

    # Reference data enters only this immutable initialization boundary. The INS
    # receives no reference arrays, which enforces blackout isolation structurally.
    ins_input = InsInput(
        time_s=drive.time_s[selection] - drive.time_s[start],
        accel_device_mps2=drive.accel_device_mps2[selection],
        gravity_device_mps2=drive.gravity_device_mps2[selection],
        gyro_ypr_rps=drive.gyro_ypr_rps[selection],
        accel_bias_device_mps2=bias.residual_accel_device_bias_mps2,
        gyro_bias_ypr_rps=bias.gyro_ypr_bias_rps,
        initial_speed_mps=float(drive.reference_speed_mps[start]),
        initial_bearing_deg=float(drive.reference_heading_deg[start]),
    )
    ins = propagate_basic_ins(ins_input)
    metrics = trajectory_metrics(ins.position_enu_m, reference_enu)
    metrics.update(
        {
            "method": "unfiltered_calibrated_imu_basic_ins",
            "blackout_start_time_s": blackout.start_time_s,
            "blackout_end_time_s": blackout.end_time_s,
            "blackout_duration_s": blackout.duration_s,
            "samples": stop - start,
        }
    )

    heading_check = _heading_validation(
        reference_enu_full, drive.reference_heading_deg, drive.reference_speed_mps
    )
    quality = {
        "experiment": experiment["name"],
        "smartphone_source": drive.source_smartphone,
        "reference_source": drive.source_reference,
        "smartphone_sampling": asdict(drive.smartphone_stats),
        "reference_sampling": asdict(drive.reference_stats),
        "missing_values": {
            "smartphone": drive.smartphone_missing_values,
            "reference": drive.reference_missing_values,
        },
        "phone_speed_unit_assessment": drive.phone_speed_unit_assessment,
        "accelerometer_mps2": {
            "minimum_xyz": np.min(drive.accel_device_mps2, axis=0),
            "maximum_xyz": np.max(drive.accel_device_mps2, axis=0),
        },
        "gyroscope_rps": {
            "minimum_ypr": np.min(drive.gyro_ypr_rps, axis=0),
            "maximum_ypr": np.max(drive.gyro_ypr_rps, axis=0),
        },
        "gravity_magnitude_mps2": {
            "mean": float(np.mean(np.linalg.norm(drive.gravity_device_mps2, axis=1))),
            "std": float(np.std(np.linalg.norm(drive.gravity_device_mps2, axis=1))),
        },
        "reference_speed_mps": {
            "minimum": float(np.min(drive.reference_speed_mps)),
            "maximum": float(np.max(drive.reference_speed_mps)),
            "mean": float(np.mean(drive.reference_speed_mps)),
        },
        "heading_validation": heading_check,
        "frame_contract": {
            "device_to_vehicle": "IO-VNBD mounting assumption: X forward, Y left, Z up",
            "vehicle_to_navigation": "vehicle X-forward/Y-left/Z-up to ENU",
            "reference_heading": "clockwise from North",
            "gyro_mapping": "source [yaw,pitch,roll] -> body XYZ [roll,pitch,yaw]",
        },
    }
    bias_payload = asdict(bias)
    blackout_payload = asdict(blackout) | {
        "selection_rule": "earliest qualifying interval after calibration; independent of INS error",
        "calibration_end_index_exclusive": bias.end_index_exclusive,
        "distance_travelled_m": metrics["distance_travelled_m"],
    }

    _save_json(experiment_dir / "data_quality_report.json", quality)
    _save_json(metrics_dir / "baseline_vta08_biases.json", bias_payload)
    _save_json(experiment_dir / "experiment_metadata.json", blackout_payload)
    pd.DataFrame([metrics]).to_csv(metrics_dir / "baseline_vta08_metrics.csv", index=False)

    pd.DataFrame(
        {
            "time_s": drive.time_s,
            "east_m": reference_enu_full[:, 0],
            "north_m": reference_enu_full[:, 1],
            "up_m": reference_enu_full[:, 2],
            "speed_mps": drive.reference_speed_mps,
            "heading_deg_clockwise_from_north": drive.reference_heading_deg,
        }
    ).to_csv(experiment_dir / "vbox_enu.csv", index=False)
    pd.DataFrame(
        {
            "time_since_blackout_s": ins.time_s,
            "east_m": ins.position_enu_m[:, 0],
            "north_m": ins.position_enu_m[:, 1],
            "up_m": ins.position_enu_m[:, 2],
            "velocity_east_mps": ins.velocity_enu_mps[:, 0],
            "velocity_north_mps": ins.velocity_enu_mps[:, 1],
            "velocity_up_mps": ins.velocity_enu_mps[:, 2],
            "heading_deg_clockwise_from_north": ins.heading_deg,
        }
    ).to_csv(experiment_dir / "basic_ins_trajectory.csv", index=False)
    pd.DataFrame(
        {
            "time_since_blackout_s": ins.time_s,
            "east_m": reference_enu[:, 0],
            "north_m": reference_enu[:, 1],
            "up_m": reference_enu[:, 2],
        }
    ).to_csv(experiment_dir / "blackout_reference_enu.csv", index=False)

    spans = [(blackout.start_time_s, blackout.end_time_s, "selected blackout")]
    _save_lines(
        drive.time_s,
        drive.accel_device_mps2,
        ["X", "Y", "Z"],
        "Vta08 Raw Smartphone Accelerometer",
        "Acceleration (m/s²)",
        plots_dir / "raw_acceleration_xyz.png",
        spans,
    )
    _save_lines(
        drive.time_s,
        drive.gyro_ypr_rps,
        ["Yaw", "Pitch", "Roll"],
        "Vta08 Raw Smartphone Gyroscope",
        "Angular rate (rad/s)",
        plots_dir / "raw_gyroscope_ypr.png",
        spans,
    )
    _save_lines(
        drive.time_s,
        drive.gravity_device_mps2,
        ["X", "Y", "Z"],
        "Vta08 Device-Frame Gravity Vector",
        "Gravity (m/s²)",
        plots_dir / "gravity_xyz.png",
    )
    _save_lines(
        drive.time_s,
        np.column_stack((np.linalg.norm(drive.accel_device_mps2, axis=1),)),
        ["|acceleration|"],
        "Vta08 Acceleration Magnitude",
        "Acceleration magnitude (m/s²)",
        plots_dir / "acceleration_magnitude.png",
        spans,
    )
    _save_lines(
        drive.time_s,
        np.column_stack((np.linalg.norm(drive.gyro_ypr_rps, axis=1),)),
        ["|angular rate|"],
        "Vta08 Gyroscope Magnitude",
        "Angular-rate magnitude (rad/s)",
        plots_dir / "gyroscope_magnitude.png",
        spans,
    )
    _save_lines(
        drive.time_s,
        np.column_stack((drive.reference_speed_mps,)),
        ["VBOX speed"],
        "Vta08 Reference Speed and Selected Blackout",
        "Speed (m/s)",
        plots_dir / "vbox_speed_blackout.png",
        spans,
    )
    corrected_device_accel = (
        drive.accel_device_mps2 - drive.gravity_device_mps2 - bias.residual_accel_device_bias_mps2
    )
    _save_lines(
        drive.time_s,
        corrected_device_accel,
        ["X", "Y", "Z"],
        "Gravity-Removed, Bias-Corrected Device Acceleration",
        "Linear acceleration (m/s²)",
        plots_dir / "gravity_removed_acceleration.png",
        spans,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(reference_enu[:, 0], reference_enu[:, 1], label="VBOX reference", linewidth=2)
    ax.plot(ins.position_enu_m[:, 0], ins.position_enu_m[:, 1], label="Basic INS", linewidth=1.5)
    ax.scatter([0], [0], marker="o", label="Blackout start")
    ax.scatter(reference_enu[-1, 0], reference_enu[-1, 1], marker="x", s=70, label="Reference end")
    ax.scatter(ins.position_enu_m[-1, 0], ins.position_enu_m[-1, 1], marker="x", s=70, label="INS end")
    ax.set(title="20-Second GNSS Blackout: VBOX Reference vs Basic INS", xlabel="East (m)", ylabel="North (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "trajectory_comparison.png", dpi=240)
    plt.close(fig)

    errors = position_errors(ins.position_enu_m, reference_enu)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ins.time_s, errors, linewidth=2, label="Basic INS position error")
    ax.set(title="Position Error During GNSS Blackout", xlabel="Time since blackout (s)", ylabel="Horizontal error (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "position_error_vs_time.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(reference_enu_full[:, 0], reference_enu_full[:, 1], linewidth=1.2)
    ax.scatter(reference_enu_full[start, 0], reference_enu_full[start, 1], label="Blackout start")
    ax.scatter(reference_enu_full[stop - 1, 0], reference_enu_full[stop - 1, 1], label="Blackout end")
    ax.set(title="Vta08 VBOX Reference Route", xlabel="East (m)", ylabel="North (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "vbox_reference_route.png", dpi=240)
    plt.close(fig)

    return {"quality": quality, "bias": bias_payload, "blackout": blackout_payload, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible Vta08 basic INS blackout baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_vta08.yaml"))
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
