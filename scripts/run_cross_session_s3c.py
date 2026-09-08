from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.run_adapter_vta08 import _load_model, _model_outputs, _noise, _stationary_bias
from src.evaluation.blackout import select_blackout_interval
from src.evaluation.metrics import position_errors, trajectory_metrics
from src.ins.adaptive_iekf import run_adaptive_iekf
from src.ins.iekf import IEKFInput, run_iekf
from src.ins.strapdown import InsInput, propagate_basic_ins
from src.mapmatch.hmm_matcher import HMMMapMatcher, OSMRoadNetwork
from src.preprocessing.alignment import estimate_mounting_alignment
from src.preprocessing.alignment_yaw import estimate_gps_heading_mounting_yaw
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu


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


def _metric(result, reference: np.ndarray, method: str, interval) -> dict:
    values = trajectory_metrics(result.position_enu_m, reference)
    values.update(
        {
            "method": method,
            "blackout_start_time_s": interval.start_time_s,
            "blackout_end_time_s": interval.end_time_s,
            "blackout_duration_s": interval.duration_s,
            "samples": len(reference),
        }
    )
    return values


def _iekf_input(
    drive,
    selection: slice,
    time: np.ndarray,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    rotation: np.ndarray | None = None,
) -> IEKFInput:
    kwargs = {}
    if rotation is not None:
        kwargs["initial_rotation_car_to_imu"] = rotation
    return IEKFInput(
        time_s=time,
        gyro_body_rps=drive.gyro_ypr_rps[selection][:, [2, 1, 0]],
        specific_force_body_mps2=drive.accel_device_mps2[selection],
        initial_speed_mps=float(drive.reference_speed_mps[selection.start]),
        initial_bearing_deg=float(drive.reference_heading_deg[selection.start]),
        initial_gyro_bias_body_rps=gyro_bias,
        initial_accel_bias_body_mps2=accel_bias,
        **kwargs,
    )


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment, inputs, outputs = config["experiment"], config["inputs"], config["outputs"]
    drive = load_synchronized_drive(
        experiment["smartphone_csv"],
        experiment["reference_csv"],
        maximum_gap_s=experiment["maximum_gap_s"],
    )
    if drive.smartphone_stats.timestamp_resets or drive.reference_stats.timestamp_resets:
        raise ValueError("cross-session confidence test requires monotonic synchronized timestamps")

    gyro_bias, accel_bias, calibration_end = _stationary_bias(drive)
    blackout = select_blackout_interval(
        drive.time_s,
        drive.reference_speed_mps,
        earliest_index=calibration_end,
        duration_s=experiment["blackout_duration_s"],
        minimum_speed_mps=experiment["minimum_speed_mps"],
        required_moving_fraction=experiment["required_moving_fraction"],
        maximum_gap_s=experiment["maximum_gap_s"],
        context_before_s=experiment["context_before_s"],
        context_after_s=0.0,
    )
    start, stop = blackout.start_index, blackout.end_index_inclusive + 1
    selection = slice(start, stop)
    indices = np.arange(start, stop)
    time = drive.time_s[selection] - drive.time_s[start]
    reference = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )

    alignment_settings = yaml.safe_load(
        Path(inputs["alignment_config"]).read_text(encoding="utf-8")
    )["alignment"]
    yaw_settings = yaml.safe_load(
        Path(inputs["yaw_alignment_config"]).read_text(encoding="utf-8")
    )["yaw_alignment"]
    gyro_body_full = drive.gyro_ypr_rps[:, [2, 1, 0]]
    tilt = estimate_mounting_alignment(
        drive.time_s,
        drive.gravity_device_mps2,
        gyro_body_full - gyro_bias,
        **alignment_settings,
    )
    yaw = estimate_gps_heading_mounting_yaw(
        drive.time_s,
        drive.accel_device_mps2,
        drive.gravity_device_mps2,
        gyro_body_full - gyro_bias,
        geodetic_to_local_enu(drive.reference_lat_deg, drive.reference_lon_deg),
        drive.reference_speed_mps,
        drive.reference_heading_deg,
        drive.orientation_ypr_deg,
        blackout_start_index=start,
        earliest_index=calibration_end,
        initial_rotation_car_to_imu=tilt.rotation_car_to_imu,
        accelerometer_bias_imu_mps2=accel_bias,
        **yaw_settings,
    )
    if yaw.end_index_exclusive > start:
        raise AssertionError("mounting-yaw alignment leaked into the blackout")

    noise = _noise(
        yaml.safe_load(Path(inputs["iekf_config"]).read_text(encoding="utf-8"))["iekf"]
    )
    basic = propagate_basic_ins(
        InsInput(
            time_s=time,
            accel_device_mps2=drive.accel_device_mps2[selection],
            gravity_device_mps2=drive.gravity_device_mps2[selection],
            gyro_ypr_rps=drive.gyro_ypr_rps[selection],
            accel_bias_device_mps2=accel_bias,
            gyro_bias_ypr_rps=gyro_bias[[2, 1, 0]],
            initial_speed_mps=float(drive.reference_speed_mps[start]),
            initial_bearing_deg=float(drive.reference_heading_deg[start]),
        )
    )
    fixed = run_iekf(_iekf_input(drive, selection, time, gyro_bias, accel_bias), noise)
    tilt_result = run_iekf(
        _iekf_input(drive, selection, time, gyro_bias, accel_bias, tilt.rotation_car_to_imu),
        noise,
    )
    yaw_result = run_iekf(
        _iekf_input(drive, selection, time, gyro_bias, accel_bias, yaw.rotation_car_to_imu),
        noise,
    )

    adapter_config = yaml.safe_load(
        Path(inputs["adapter_config"]).read_text(encoding="utf-8")
    )
    model = _load_model(Path(inputs["model"]))
    adapter_outputs = _model_outputs(
        model,
        gyro_body_full,
        drive.accel_device_mps2,
        gyro_bias,
        accel_bias,
        indices,
        adapter_config["model"]["context_samples"],
    )
    adaptive = run_adaptive_iekf(
        _iekf_input(drive, selection, time, gyro_bias, accel_bias, yaw.rotation_car_to_imu),
        adapter_outputs,
        noise,
    )

    rows = [
        _metric(basic, reference, "step1_plain_strapdown", blackout),
        _metric(fixed, reference, "step2_fixed_iekf_nhc", blackout),
        _metric(tilt_result, reference, "step3_tilt_aligned_iekf", blackout),
        _metric(yaw_result, reference, "step3b_causal_yaw_aligned_iekf", blackout),
        _metric(adaptive, reference, "step4_cnn_adapted_iekf", blackout),
    ]
    step5 = dict(rows[-1])
    step5["method"] = "step5_gnss_suppressed_during_blackout"
    rows.append(step5)

    matched = None
    network = None
    map_status: dict = {
        "status": "not_run",
        "reason": "The S3c OSM extract is not available locally.",
        "required_file": inputs["osm_xml"],
    }
    osm_path = Path(inputs["osm_xml"])
    if osm_path.exists():
        network = OSMRoadNetwork.from_xml(
            osm_path,
            origin_lat_deg=float(drive.reference_lat_deg[start]),
            origin_lon_deg=float(drive.reference_lon_deg[start]),
        )
        corrected_gyro = (
            drive.gyro_ypr_rps[selection][:, [2, 1, 0]]
            - 0.02 * adapter_outputs[:, 2:5]
            - gyro_bias
        )
        heading = np.zeros(len(time))
        heading[0] = drive.reference_heading_deg[start]
        heading[1:] = (
            heading[0] + np.rad2deg(np.cumsum(corrected_gyro[1:, 2] * np.diff(time)))
        ) % 360.0
        matcher_settings = yaml.safe_load(
            Path(inputs["matcher_config"]).read_text(encoding="utf-8")
        )["matcher"]
        matched = HMMMapMatcher(network, **matcher_settings).match(
            adaptive.position_enu_m, heading
        )
        matched_metric = trajectory_metrics(matched.position_enu_m, reference)
        matched_metric.update(
            {
                "method": "step6_osm_hmm_map_matching",
                "blackout_start_time_s": blackout.start_time_s,
                "blackout_end_time_s": blackout.end_time_s,
                "blackout_duration_s": blackout.duration_s,
                "samples": len(reference),
            }
        )
        rows.append(matched_metric)
        map_status = {
            "status": "complete",
            "osm_xml": str(osm_path),
            "drivable_segments": len(network.segments),
            "matcher_parameters": matcher_settings,
            "mean_snap_distance_m": float(np.mean(matched.candidate_distance_m)),
            "maximum_snap_distance_m": float(np.max(matched.candidate_distance_m)),
            "unique_segments": int(len(np.unique(matched.segment_id))),
            "unique_ways": int(len(np.unique(matched.way_id))),
            "change_from_step5": {
                "final_position_error_m": matched_metric["final_position_error_m"]
                - step5["final_position_error_m"],
                "position_rmse_m": matched_metric["position_rmse_m"]
                - step5["position_rmse_m"],
                "drift_percentage_points": matched_metric["drift_percent"]
                - step5["drift_percent"],
            },
        }

    experiment_dir = Path(outputs["experiment_dir"])
    metrics_dir = Path(outputs["metrics_dir"])
    plots_dir = Path(outputs["plots_dir"])
    for directory in (experiment_dir, metrics_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "cumulative_metrics.csv", index=False)
    trajectories = {
        "time_since_blackout_s": time,
        "reference_east_m": reference[:, 0],
        "reference_north_m": reference[:, 1],
        "step1_east_m": basic.position_enu_m[:, 0],
        "step1_north_m": basic.position_enu_m[:, 1],
        "step2_east_m": fixed.position_enu_m[:, 0],
        "step2_north_m": fixed.position_enu_m[:, 1],
        "step3_east_m": tilt_result.position_enu_m[:, 0],
        "step3_north_m": tilt_result.position_enu_m[:, 1],
        "step3b_east_m": yaw_result.position_enu_m[:, 0],
        "step3b_north_m": yaw_result.position_enu_m[:, 1],
        "step4_east_m": adaptive.position_enu_m[:, 0],
        "step4_north_m": adaptive.position_enu_m[:, 1],
    }
    if matched is not None:
        trajectories.update(
            {
                "step6_east_m": matched.position_enu_m[:, 0],
                "step6_north_m": matched.position_enu_m[:, 1],
            }
        )
    pd.DataFrame(trajectories).to_csv(experiment_dir / "cumulative_trajectories.csv", index=False)

    report = {
        "experiment": {
            "session": experiment["name"],
            "role": experiment["role"],
            "selection_policy": "earliest qualifying interval after stationary calibration; selected before observing navigation errors",
            "oracle_initialization": "VBOX speed and heading are used only at blackout entry; VBOX is reference-only afterward",
            "parameter_policy": "Vta08 IEKF, CNN, alignment, and HMM settings are reused without S3c error-based tuning",
            "blackout": asdict(blackout),
        },
        "dataset": {
            "samples": len(drive.time_s),
            "duration_s": float(drive.time_s[-1] - drive.time_s[0]),
            "sampling_hz": drive.smartphone_stats.estimated_hz,
            "timestamp_resets": drive.smartphone_stats.timestamp_resets,
            "phone_speed_unit_assessment": drive.phone_speed_unit_assessment,
        },
        "calibration": {
            "end_index_exclusive": calibration_end,
            "gyro_bias_body_rps": gyro_bias,
            "accel_bias_body_mps2": accel_bias,
            "tilt_alignment": asdict(tilt),
            "yaw_alignment": asdict(yaw),
            "pre_blackout_leakage_check": yaw.end_index_exclusive <= start,
        },
        "model": {
            "checkpoint": inputs["model"],
            "parameter_count": model.parameter_count,
            "training_session": "Vta2",
            "validation_session": "Vta16",
            "S3c_seen_during_training_or_selection": False,
        },
        "cumulative_metrics": rows,
        "step5_interpretation": "GNSS is unavailable by experiment definition, so Step 5 equals Step 4 inside the locked blackout.",
        "map_matching": map_status,
    }
    (experiment_dir / "evaluation.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(9, 9))
    axis.plot(reference[:, 0], reference[:, 1], color="black", linewidth=2.5, label="VBOX reference")
    axis.plot(basic.position_enu_m[:, 0], basic.position_enu_m[:, 1], linewidth=1.0, label="Step 1 strapdown")
    axis.plot(fixed.position_enu_m[:, 0], fixed.position_enu_m[:, 1], linewidth=1.0, label="Step 2 IEKF")
    axis.plot(yaw_result.position_enu_m[:, 0], yaw_result.position_enu_m[:, 1], linewidth=1.5, label="Step 3b alignment")
    axis.plot(adaptive.position_enu_m[:, 0], adaptive.position_enu_m[:, 1], linewidth=1.7, label="Step 4/5 CNN-IEKF")
    if matched is not None:
        axis.plot(matched.position_enu_m[:, 0], matched.position_enu_m[:, 1], linewidth=2.0, label="Step 6 OSM HMM")
    axis.scatter([0], [0], color="black", zorder=5, label="Blackout start")
    axis.set(title="S3c blind cross-driver blackout comparison", xlabel="East (m)", ylabel="North (m)")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "cumulative_trajectory.png", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for result, label in (
        (basic, "Step 1 strapdown"),
        (fixed, "Step 2 IEKF"),
        (tilt_result, "Step 3 tilt"),
        (yaw_result, "Step 3b yaw"),
        (adaptive, "Step 4/5 CNN-IEKF"),
    ):
        axis.plot(time, position_errors(result.position_enu_m, reference), label=label)
    if matched is not None:
        axis.plot(time, position_errors(matched.position_enu_m, reference), label="Step 6 OSM HMM")
    axis.set(title="S3c position error during the identical blackout", xlabel="Time since blackout (s)", ylabel="Horizontal error (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "position_error_comparison.png", dpi=240)
    plt.close(figure)

    if matched is not None and network is not None:
        figure, axis = plt.subplots(figsize=(9, 9))
        for segment in network.segments:
            points = np.vstack((segment.start_enu_m, segment.end_enu_m))
            if np.min(np.linalg.norm(points[:, :2], axis=1)) < 500.0:
                axis.plot(points[:, 0], points[:, 1], color="0.80", linewidth=0.8, zorder=0)
        axis.plot(reference[:, 0], reference[:, 1], color="black", linewidth=2.5, label="VBOX reference")
        axis.plot(adaptive.position_enu_m[:, 0], adaptive.position_enu_m[:, 1], color="#d95f02", linewidth=1.8, label="Step 5 filter output")
        axis.plot(matched.position_enu_m[:, 0], matched.position_enu_m[:, 1], color="#1b9e77", linewidth=2.0, label="Step 6 OSM HMM")
        axis.scatter([0], [0], color="black", zorder=5, label="Blackout start")
        axis.set(title="S3c blackout: frozen OSM HMM map matching", xlabel="East (m)", ylabel="North (m)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "osm_hmm_comparison.png", dpi=240)
        plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen pipeline on blind Driver-A session S3c")
    parser.add_argument("--config", type=Path, default=Path("configs/cross_session_s3c.yaml"))
    args = parser.parse_args()
    print(json.dumps(_jsonable(run(args.config)), indent=2))


if __name__ == "__main__":
    main()
