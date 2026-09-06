from __future__ import annotations

import argparse
import json
from pathlib import Path
import time as clock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.blackout import select_blackout_interval
from src.evaluation.metrics import position_errors, trajectory_metrics
from src.ins.adaptive_iekf import run_adaptive_iekf
from src.ins.iekf import IEKFInput, IEKFNoise, run_iekf
from src.models.adapter_cnn import DilatedCNNAdapter
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu
from src.preprocessing.stationary import estimate_biases_from_stationary_run, stationary_mask
from train import mounting_rotation


def _noise(config: dict) -> IEKFNoise:
    p, m, i = config["process_covariance"], config["measurement_covariance"], config["initial_covariance"]
    return IEKFNoise(
        gyro=p["gyro"], accelerometer=p["accelerometer"], gyro_bias=p["gyro_bias"],
        accelerometer_bias=p["accelerometer_bias"], car_to_imu_rotation=p["car_to_imu_rotation"],
        car_to_imu_translation=p["car_to_imu_translation"], lateral_velocity=m["lateral_velocity"],
        upward_velocity=m["upward_velocity"], initial_roll_pitch=i["roll_pitch"],
        initial_velocity=i["velocity"], initial_gyro_bias=i["gyro_bias"],
        initial_accelerometer_bias=i["accelerometer_bias"],
        initial_car_to_imu_rotation=i["car_to_imu_rotation"],
        initial_car_to_imu_translation=i["car_to_imu_translation"],
    )


def _load_model(path: Path) -> DilatedCNNAdapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = checkpoint["model_config"]
    model = DilatedCNNAdapter(settings["channels"], settings["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def _model_outputs(
    model: DilatedCNNAdapter,
    gyro_body: np.ndarray,
    accel: np.ndarray,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    indices: np.ndarray,
    context_samples: int,
) -> np.ndarray:
    imu = np.column_stack((gyro_body - gyro_bias, accel - accel_bias))
    windows = []
    for index in indices:
        start = max(0, index - context_samples + 1)
        window = imu[start:index + 1]
        if len(window) < context_samples:
            window = np.vstack((np.repeat(window[:1], context_samples - len(window), axis=0), window))
        windows.append(window)
    with torch.inference_mode():
        return model(torch.tensor(np.asarray(windows), dtype=torch.float32)).numpy()


def _stationary_bias(drive) -> tuple[np.ndarray, np.ndarray, int]:
    mask = stationary_mask(
        drive.reference_speed_mps, drive.gyro_ypr_rps, drive.accel_device_mps2,
        gravity_mps2=9.80665, maximum_reference_speed_mps=0.5,
        maximum_gyro_norm_rps=0.08, gravity_magnitude_tolerance_mps2=0.75,
    )
    bias = estimate_biases_from_stationary_run(
        drive.time_s, drive.accel_device_mps2, drive.gravity_device_mps2,
        drive.gyro_ypr_rps, mask, minimum_duration_s=5.0,
    )
    return (
        bias.gyro_ypr_bias_rps[[2, 1, 0]],
        bias.residual_accel_device_bias_mps2,
        bias.end_index_exclusive,
    )


def _evaluate_interval(
    drive,
    start: int,
    end_inclusive: int,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    car_rotation: np.ndarray,
    model: DilatedCNNAdapter,
    context_samples: int,
    noise: IEKFNoise,
    name: str,
) -> tuple[dict, dict, object, object, np.ndarray, np.ndarray]:
    selection = slice(start, end_inclusive + 1)
    indices = np.arange(start, end_inclusive + 1)
    time = drive.time_s[selection] - drive.time_s[start]
    gyro_body_full = drive.gyro_ypr_rps[:, [2, 1, 0]]
    inputs = IEKFInput(
        time_s=time,
        gyro_body_rps=gyro_body_full[selection],
        specific_force_body_mps2=drive.accel_device_mps2[selection],
        initial_speed_mps=float(drive.reference_speed_mps[start]),
        initial_bearing_deg=float(drive.reference_heading_deg[start]),
        initial_gyro_bias_body_rps=gyro_bias,
        initial_accel_bias_body_mps2=accel_bias,
        initial_rotation_car_to_imu=car_rotation,
    )
    outputs = _model_outputs(
        model, gyro_body_full, drive.accel_device_mps2, gyro_bias, accel_bias, indices, context_samples
    )
    fixed = run_iekf(inputs, noise)
    adaptive = run_adaptive_iekf(inputs, outputs, noise)
    reference = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )
    fixed_metrics, adaptive_metrics = trajectory_metrics(fixed.position_enu_m, reference), trajectory_metrics(adaptive.position_enu_m, reference)
    common = {
        "session": name, "blackout_start_time_s": float(drive.time_s[start]),
        "blackout_end_time_s": float(drive.time_s[end_inclusive]), "samples": len(time),
    }
    fixed_metrics.update(common | {"method": "yaw_aligned_fixed_covariance_iekf"})
    adaptive_metrics.update(common | {"method": "yaw_aligned_cnn_adaptive_iekf"})
    return fixed_metrics, adaptive_metrics, fixed, adaptive, reference, outputs


def _latency(model: DilatedCNNAdapter, context_samples: int) -> dict:
    sample = torch.zeros(1, context_samples, 6)
    with torch.inference_mode():
        for _ in range(20): model(sample)
        elapsed = []
        for _ in range(200):
            start = clock.perf_counter(); model(sample); elapsed.append((clock.perf_counter() - start) * 1000.0)
    return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95)), "runs": len(elapsed)}


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = _load_model(Path(config["outputs"]["model"]))
    context = config["model"]["context_samples"]
    car_rotation = mounting_rotation(config["mounting"])
    iekf_config = yaml.safe_load(Path("configs/iekf_vta08.yaml").read_text(encoding="utf-8"))
    noise = _noise(iekf_config["iekf"])

    test_spec = config["experiment"]["test"]
    test_drive = load_synchronized_drive(test_spec["smartphone_csv"], test_spec["reference_csv"])
    biases = json.loads(Path("results/metrics/baseline_vta08_biases.json").read_text())
    metadata = json.loads(Path("results/experiments/baseline_vta08/experiment_metadata.json").read_text())
    test_fixed, test_adaptive, _, test_result, test_reference, test_outputs = _evaluate_interval(
        test_drive, int(metadata["start_index"]), int(metadata["end_index_inclusive"]),
        np.asarray(biases["gyro_ypr_bias_rps"])[[2, 1, 0]],
        np.asarray(biases["residual_accel_device_bias_mps2"]), car_rotation, model, context,
        noise, test_spec["name"],
    )

    validation_spec = config["experiment"]["validation"]
    validation_drive = load_synchronized_drive(validation_spec["smartphone_csv"], validation_spec["reference_csv"])
    val_gyro_bias, val_accel_bias, calibration_end = _stationary_bias(validation_drive)
    blackout = select_blackout_interval(
        validation_drive.time_s, validation_drive.reference_speed_mps,
        earliest_index=calibration_end, duration_s=20.0, minimum_speed_mps=5.0,
        required_moving_fraction=0.90, maximum_gap_s=0.15, context_before_s=3.2,
        context_after_s=0.0,
    )
    val_fixed, val_adaptive, _, _, _, _ = _evaluate_interval(
        validation_drive, blackout.start_index, blackout.end_index_inclusive,
        val_gyro_bias, val_accel_bias, car_rotation, model, context, noise,
        validation_spec["name"],
    )

    step3b = json.loads(Path("results/experiments/alignment_yaw_vta08/comparison.json").read_text())
    prior = step3b["iekf_with_tilt_and_gps_heading_yaw"]
    report = {
        "vta08": {
            "step1": step3b["baseline"], "step2": step3b["iekf"],
            "step3_tilt": step3b["iekf_with_tilt_alignment"], "step3b_yaw": prior,
            "step4_cnn": test_adaptive,
            "change_from_step3b": {
                "final_position_error_m": test_adaptive["final_position_error_m"] - prior["final_position_error_m"],
                "position_rmse_m": test_adaptive["position_rmse_m"] - prior["position_rmse_m"],
                "drift_percentage_points": test_adaptive["drift_percent"] - prior["drift_percent"],
            },
        },
        "held_out_validation_session": {"fixed": val_fixed, "cnn": val_adaptive},
        "adapter_outputs_vta08": {
            "measurement_scale_mean": np.exp(np.log(10.0) * test_outputs[:, :2]).mean(axis=0).tolist(),
            "gyro_bias_delta_mean_rps": (0.02 * test_outputs[:, 2:5]).mean(axis=0).tolist(),
            "accel_bias_delta_mean_mps2": (0.5 * test_outputs[:, 5:8]).mean(axis=0).tolist(),
            "accel_scale_mean": (1.0 + 0.1 * test_outputs[:, 8:11]).mean(axis=0).tolist(),
        },
        "model": {
            "parameter_count": model.parameter_count,
            "file_size_bytes": Path(config["outputs"]["model"]).stat().st_size,
            "cpu_latency_per_window": _latency(model, context),
            "context_samples": context,
            "context_seconds_at_10hz": context / 10.0,
        },
        "split": json.loads(Path(config["outputs"]["split"]).read_text()),
    }
    output_dir = Path("results/experiments/adapter_cnn")
    plot_dir = Path("results/plots/adapter_cnn")
    output_dir.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([step3b["baseline"], step3b["iekf"], step3b["iekf_with_tilt_alignment"], prior, test_adaptive]).to_csv(
        "results/metrics/adapter_cnn_vta08_metrics.csv", index=False
    )
    pd.DataFrame({
        "time_since_blackout_s": test_result.time_s,
        "east_m": test_result.position_enu_m[:, 0], "north_m": test_result.position_enu_m[:, 1],
        "up_m": test_result.position_enu_m[:, 2],
    }).to_csv(output_dir / "vta08_cnn_trajectory.csv", index=False)

    step1 = pd.read_csv("results/experiments/baseline_vta08/basic_ins_trajectory.csv")
    step2 = pd.read_csv("results/experiments/iekf_vta08/iekf_trajectory.csv")
    yaw = pd.read_csv("results/experiments/alignment_yaw_vta08/yaw_aligned_iekf_trajectory.csv")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(test_reference[:, 0], test_reference[:, 1], linewidth=2.2, label="VBOX reference")
    ax.plot(step1.east_m, step1.north_m, linewidth=1.0, label="Step 1")
    ax.plot(step2.east_m, step2.north_m, linewidth=1.0, label="Step 2")
    ax.plot(yaw.east_m, yaw.north_m, linewidth=1.5, label="Step 3b yaw")
    ax.plot(test_result.position_enu_m[:, 0], test_result.position_enu_m[:, 1], linewidth=1.8, label="Step 4 CNN")
    ax.scatter([0], [0], label="Blackout start", zorder=5)
    ax.set(title="Vta08: Yaw-Aligned CNN-IEKF", xlabel="East (m)", ylabel="North (m)")
    ax.axis("equal"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "vta08_trajectory_comparison.png", dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_result.time_s, position_errors(yaw[["east_m", "north_m", "up_m"]].to_numpy(), test_reference), label="Step 3b yaw")
    ax.plot(test_result.time_s, position_errors(test_result.position_enu_m, test_reference), label="Step 4 CNN")
    ax.set(title="Vta08 Blackout Error: Structural Alignment vs Learned Adapter", xlabel="Time (s)", ylabel="Horizontal error (m)")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "vta08_error_comparison.png", dpi=240); plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step 4 CNN-adapted IEKF")
    parser.add_argument("--config", type=Path, default=Path("configs/adapter_cnn.yaml"))
    args = parser.parse_args(); print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
