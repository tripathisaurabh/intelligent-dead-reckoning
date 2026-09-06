from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import position_errors, trajectory_metrics
from src.ins.iekf import IEKFInput, IEKFNoise, run_iekf
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


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    baseline_config = yaml.safe_load(Path(experiment["baseline_config"]).read_text(encoding="utf-8"))
    baseline_experiment = baseline_config["experiment"]
    drive = load_synchronized_drive(
        baseline_experiment["smartphone_csv"],
        baseline_experiment["reference_csv"],
        maximum_gap_s=baseline_config["validation"]["maximum_gap_s"],
    )
    biases = json.loads(Path(experiment["baseline_biases"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(experiment["baseline_metadata"]).read_text(encoding="utf-8"))
    start = int(metadata["start_index"])
    end = int(metadata["end_index_inclusive"])
    selection = slice(start, end + 1)
    time = drive.time_s[selection] - drive.time_s[start]
    reference_enu = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )

    gyro_bias_ypr = np.asarray(biases["gyro_ypr_bias_rps"], dtype=float)
    # Dataset semantic order [yaw, pitch, roll] -> body XYZ [roll, pitch, yaw].
    gyro_body = drive.gyro_ypr_rps[selection][:, [2, 1, 0]]
    gyro_bias_body = gyro_bias_ypr[[2, 1, 0]]
    noise = _noise_from_config(config["iekf"])
    result = run_iekf(
        IEKFInput(
            time_s=time,
            gyro_body_rps=gyro_body,
            specific_force_body_mps2=drive.accel_device_mps2[selection],
            initial_speed_mps=float(drive.reference_speed_mps[start]),
            initial_bearing_deg=float(drive.reference_heading_deg[start]),
            initial_gyro_bias_body_rps=gyro_bias_body,
            initial_accel_bias_body_mps2=np.asarray(
                biases["residual_accel_device_bias_mps2"], dtype=float
            ),
        ),
        noise,
    )
    metrics = trajectory_metrics(result.position_enu_m, reference_enu)
    metrics.update(
        {
            "method": "fixed_covariance_iekf_nonholonomic",
            "blackout_start_time_s": float(metadata["start_time_s"]),
            "blackout_end_time_s": float(metadata["end_time_s"]),
            "blackout_duration_s": float(metadata["duration_s"]),
            "samples": len(time),
        }
    )
    baseline_metrics = pd.read_csv("results/metrics/baseline_vta08_metrics.csv").iloc[0].to_dict()
    comparison = {
        "baseline": baseline_metrics,
        "iekf": metrics,
        "final_error_improvement_percent": 100.0
        * (baseline_metrics["final_position_error_m"] - metrics["final_position_error_m"])
        / baseline_metrics["final_position_error_m"],
        "rmse_improvement_percent": 100.0
        * (baseline_metrics["position_rmse_m"] - metrics["position_rmse_m"])
        / baseline_metrics["position_rmse_m"],
        "fixed_covariances": {
            "process_diagonal": np.diag(noise.process_matrix()).tolist(),
            "measurement_diagonal": np.diag(noise.measurement_matrix()).tolist(),
        },
        "covariance_selection_note": (
            "The published KITTI default measurement diagonal [0.2, 300] was unstable "
            "on Vta08. The locked [10, 10] (m/s)^2 constraint is intentionally loose and "
            "symmetric. The tighter [1,1] endpoint looked better but had worse RMSE and "
            "a looping trajectory; all manual stability trials are reported separately."
        ),
    }

    outputs = config["outputs"]
    experiment_dir = Path(outputs["experiment_dir"])
    metrics_path = Path(outputs["metrics"])
    plot_path = Path(outputs["plot"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([baseline_metrics, metrics]).to_csv(metrics_path, index=False)
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
    ).to_csv(experiment_dir / "iekf_trajectory.csv", index=False)
    (experiment_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    covariance_trials = [
        {"lateral": 0.2, "upward": 300.0, "fpe_m": 311.53, "rmse_m": 148.94, "drift_percent": 219.4},
        {"lateral": 0.2, "upward": 10.0, "fpe_m": 570.45, "rmse_m": 249.38, "drift_percent": 401.7},
        {"lateral": 0.2, "upward": 1.0, "fpe_m": 101.49, "rmse_m": 68.38, "drift_percent": 71.5},
        {"lateral": 0.2, "upward": 0.2, "fpe_m": 94.16, "rmse_m": 69.72, "drift_percent": 66.3},
        {"lateral": 1.0, "upward": 1.0, "fpe_m": 56.68, "rmse_m": 67.69, "drift_percent": 39.9},
        {"lateral": 10.0, "upward": 10.0, "fpe_m": 78.72, "rmse_m": 41.07, "drift_percent": 55.4},
        {"lateral": 0.01, "upward": 0.01, "fpe_m": 210.81, "rmse_m": 121.41, "drift_percent": 148.5},
        {"lateral": 1.0, "upward": 0.1, "fpe_m": 156.44, "rmse_m": 80.23, "drift_percent": 110.2},
        {"lateral": 10.0, "upward": 0.1, "fpe_m": 74.66, "rmse_m": 52.02, "drift_percent": 52.6},
    ]
    (experiment_dir / "covariance_trials.json").write_text(
        json.dumps(
            {
                "purpose": "manual stability check after published KITTI covariance diverged",
                "selected": {"lateral": noise.lateral_velocity, "upward": noise.upward_velocity},
                "warning": "These trials use Vta08 and are not evidence of cross-drive generalization.",
                "trials": covariance_trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    baseline = pd.read_csv(experiment["baseline_trajectory"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(reference_enu[:, 0], reference_enu[:, 1], linewidth=2.2, label="VBOX reference")
    ax.plot(baseline["east_m"], baseline["north_m"], linewidth=1.5, label="Step 1 strapdown")
    ax.plot(result.position_enu_m[:, 0], result.position_enu_m[:, 1], linewidth=1.7, label="Step 2 fixed-covariance IEKF")
    ax.scatter([0], [0], label="Blackout start", zorder=5)
    ax.set(
        title="Vta08 20-Second Blackout: Strapdown vs IEKF",
        xlabel="East (m)",
        ylabel="North (m)",
    )
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=240)
    plt.close(fig)

    baseline_error = position_errors(
        baseline[["east_m", "north_m", "up_m"]].to_numpy(), reference_enu
    )
    iekf_error = position_errors(result.position_enu_m, reference_enu)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, baseline_error, label="Step 1 strapdown")
    ax.plot(time, iekf_error, label="Step 2 IEKF")
    ax.set(
        title="Position Error During Identical GNSS Blackout",
        xlabel="Time since blackout (s)",
        ylabel="Horizontal error (m)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path.parent / "position_error_comparison.png", dpi=240)
    plt.close(fig)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 2 fixed-covariance IEKF on Vta08")
    parser.add_argument("--config", type=Path, default=Path("configs/iekf_vta08.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
