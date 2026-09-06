from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.metrics import position_errors, trajectory_metrics
from src.ins.gnss import GnssQualityGate
from src.ins.iekf import (
    IEKFNoise, IEKFState, normalize_rotation, propagate_state,
    pseudo_measurement_model, update_gnss, update_nonholonomic,
)
from src.ins.mode_arbitration import GnssModeArbitrator, NavigationMode
from src.models.adapter_cnn import DilatedCNNAdapter
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import (
    bearing_deg_to_enu_velocity, geodetic_to_local_enu, vehicle_to_enu_rotation,
)
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


def _model(path: Path) -> DilatedCNNAdapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = checkpoint["model_config"]
    model = DilatedCNNAdapter(settings["channels"], settings["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def _adapter_outputs(model, drive, indices, gyro_bias, accel_bias, context_samples):
    gyro = drive.gyro_ypr_rps[:, [2, 1, 0]]
    imu = np.column_stack((gyro - gyro_bias, drive.accel_device_mps2 - accel_bias))
    windows = []
    for index in indices:
        start = max(0, index - context_samples + 1)
        window = imu[start:index + 1]
        if len(window) < context_samples:
            window = np.vstack((np.repeat(window[:1], context_samples - len(window), axis=0), window))
        windows.append(window)
    with torch.inference_mode():
        return model(torch.tensor(np.asarray(windows), dtype=torch.float32)).numpy()


def _initial_state(drive, index, phone_position, car_to_imu, gyro_bias, accel_bias, noise, accuracy):
    covariance = noise.initial_matrix()
    covariance[3:6, 3:6] = 1.5**2 * np.eye(3)
    covariance[6:9, 6:9] = max(accuracy, 3.0)**2 * np.eye(3)
    heading = float(drive.phone_heading_deg[index])
    return IEKFState(
        rotation_imu_to_enu=vehicle_to_enu_rotation(heading) @ car_to_imu.T,
        velocity_enu_mps=bearing_deg_to_enu_velocity(float(drive.phone_speed_raw[index]), heading),
        position_enu_m=phone_position.copy(),
        gyro_bias_body_rps=gyro_bias.copy(), accel_bias_body_mps2=accel_bias.copy(),
        rotation_car_to_imu=car_to_imu.copy(), translation_car_to_imu_m=np.zeros(3),
        covariance=covariance,
    )


def _oracle_initial_state(drive, index, car_to_imu, gyro_bias, accel_bias, noise):
    """Match the locked Steps 1-4 VBOX initialization exactly."""
    bearing = float(drive.reference_heading_deg[index])
    return IEKFState(
        rotation_imu_to_enu=vehicle_to_enu_rotation(bearing) @ car_to_imu.T,
        velocity_enu_mps=bearing_deg_to_enu_velocity(
            float(drive.reference_speed_mps[index]), bearing
        ),
        position_enu_m=np.zeros(3),
        gyro_bias_body_rps=gyro_bias.copy(), accel_bias_body_mps2=accel_bias.copy(),
        rotation_car_to_imu=car_to_imu.copy(), translation_car_to_imu_m=np.zeros(3),
        covariance=noise.initial_matrix(),
    )


def _propagate_adaptive(state, gyro, force, output, dt, noise):
    scale = np.exp(np.log(10.0) * output[:2])
    corrected_gyro = gyro - 0.02 * output[2:5]
    corrected_force = (
        (force - state.accel_bias_body_mps2 - 0.50 * output[5:8])
        * (1.0 + 0.10 * output[8:11]) + state.accel_bias_body_mps2
    )
    state = propagate_state(state, corrected_gyro, corrected_force, dt, noise)
    dynamic = replace(
        noise, lateral_velocity=noise.lateral_velocity * scale[0],
        upward_velocity=noise.upward_velocity * scale[1],
    )
    return update_nonholonomic(state, corrected_gyro, dynamic)[0]


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp, gc = config["experiment"], config["gnss"]
    drive = load_synchronized_drive(exp["smartphone_csv"], exp["reference_csv"])
    start = int(np.searchsorted(drive.time_s, exp["evaluation_start_s"], side="left"))
    end = int(np.searchsorted(drive.time_s, exp["evaluation_end_s"], side="right"))
    indices = np.arange(start, end)
    time = drive.time_s[indices]
    blackout = (time >= exp["blackout_start_s"]) & (time <= exp["blackout_end_s"])

    adapter_config = yaml.safe_load(Path(config["inputs"]["adapter_config"]).read_text())
    car_to_imu = mounting_rotation(adapter_config["mounting"])
    noise = _noise(yaml.safe_load(Path(config["inputs"]["iekf_config"]).read_text())["iekf"])
    biases = json.loads(Path(config["inputs"]["biases"]).read_text())
    gyro_bias = np.asarray(biases["gyro_ypr_bias_rps"])[[2, 1, 0]]
    accel_bias = np.asarray(biases["residual_accel_device_bias_mps2"])
    model = _model(Path(config["inputs"]["model"]))
    outputs = _adapter_outputs(
        model, drive, indices, gyro_bias, accel_bias, adapter_config["model"]["context_samples"]
    )

    origin_lat, origin_lon = float(drive.phone_lat_deg[start]), float(drive.phone_lon_deg[start])
    origin_alt = float(drive.phone_alt_m[start])
    phone_enu = geodetic_to_local_enu(
        drive.phone_lat_deg[indices], drive.phone_lon_deg[indices],
        altitude_m=drive.phone_alt_m[indices], origin_lat_deg=origin_lat,
        origin_lon_deg=origin_lon, origin_altitude_m=origin_alt,
    )
    reference = geodetic_to_local_enu(
        drive.reference_lat_deg[indices], drive.reference_lon_deg[indices],
        origin_lat_deg=origin_lat, origin_lon_deg=origin_lon, origin_altitude_m=origin_alt,
    )
    fused = _initial_state(
        drive, start, phone_enu[0], car_to_imu, gyro_bias, accel_bias, noise,
        drive.phone_accuracy_m[start],
    )
    inertial = _initial_state(
        drive, start, phone_enu[0], car_to_imu, gyro_bias, accel_bias, noise,
        drive.phone_accuracy_m[start],
    )
    gate = GnssQualityGate(
        gc["minimum_satellites"], gc["maximum_accuracy_m"], gc["maximum_fix_age_s"]
    )
    arbitrator = GnssModeArbitrator(gc["transition_duration_s"])
    positions, inertial_positions = np.zeros((len(indices), 3)), np.zeros((len(indices), 3))
    velocities = np.zeros((len(indices), 3))
    modes, weights, nis_values, corrections, reasons = [], [], [], [], []
    last_visible = 0
    accepted_new_fixes = 0
    rejected_innovations = 0
    transition_start_global = None
    last_aided_time = float(time[0])
    previous_blend_weight = 1.0
    last_fix_innovation_accepted = True

    for local, global_index in enumerate(indices):
        if local:
            dt = time[local] - time[local - 1]
            gyro = drive.gyro_ypr_rps[global_index, [2, 1, 0]]
            force = drive.accel_device_mps2[global_index]
            fused = _propagate_adaptive(fused, gyro, force, outputs[local], dt, noise)
            inertial = _propagate_adaptive(inertial, gyro, force, outputs[local], dt, noise)

        new_fix = local == 0 or any(
            drive_value[global_index] != drive_value[global_index - 1]
            for drive_value in (
                drive.phone_lat_deg, drive.phone_lon_deg, drive.phone_speed_raw,
                drive.phone_heading_deg,
            )
        )
        if new_fix and not blackout[local]:
            last_visible = local
        externally_available = not blackout[local]
        age = float(time[local] - time[last_visible])
        quality = gate.evaluate(
            satellites=float(drive.phone_satellites[indices[last_visible]]),
            accuracy_m=float(drive.phone_accuracy_m[indices[last_visible]]), age_s=age,
            externally_available=externally_available,
        )
        fix_position = phone_enu[last_visible].copy()
        fix_velocity = bearing_deg_to_enu_velocity(
            float(drive.phone_speed_raw[indices[last_visible]]),
            float(drive.phone_heading_deg[indices[last_visible]]),
        )
        fix_position = fix_position + fix_velocity * age
        position_std = max(float(drive.phone_accuracy_m[indices[last_visible]]), gc["minimum_position_std_m"])
        time_without_aiding = max(0.0, float(time[local]) - last_aided_time)
        gate_position_std = float(np.hypot(
            position_std, gc["reacquisition_position_uncertainty_growth_mps"] * time_without_aiding
        ))
        gate_velocity_std = float(np.hypot(
            gc["velocity_std_mps"],
            gc["reacquisition_velocity_uncertainty_growth_mps2"] * time_without_aiding,
        ))
        trial = None
        if quality.accepted and new_fix and not blackout[local]:
            trial = update_gnss(
                fused, fix_position, fix_velocity, gate_position_std, gate_velocity_std,
                gc["innovation_chi_square_gate"], measurement_weight=0.0,
            )
            last_fix_innovation_accepted = trial.accepted
            rejected_innovations += int(not trial.accepted)
        accepted = quality.accepted and last_fix_innovation_accepted
        reason = quality.reason if quality.reason != "accepted" else ("accepted" if accepted else "innovation")
        decision = arbitrator.update(float(time[local]), accepted)
        if decision.mode is NavigationMode.GNSS_AIDED and accepted:
            last_aided_time = float(time[local])
        if decision.mode is NavigationMode.TRANSITIONING and transition_start_global is None:
            transition_start_global = int(global_index)

        correction = 0.0
        nis = float(trial.normalized_innovation_squared) if trial is not None else np.nan
        if decision.mode is NavigationMode.DEAD_RECKONING:
            previous_blend_weight = 0.0
        blend_increment = max(0.0, decision.gnss_blend_weight - previous_blend_weight)
        should_update = accepted and (
            decision.mode is NavigationMode.TRANSITIONING
            or (decision.mode is NavigationMode.GNSS_AIDED and (new_fix or blend_increment > 0.0))
        )
        if should_update:
            before_state = fused
            before = fused.position_enu_m.copy()
            applied_weight = blend_increment if blend_increment > 0.0 else decision.gnss_blend_weight
            update = update_gnss(
                fused, fix_position, fix_velocity, position_std, gc["velocity_std_mps"],
                1.0e12, applied_weight,
            )
            fused = update.state
            correction = float(np.linalg.norm(fused.position_enu_m[:2] - before[:2]))
            cap = gc["maximum_transition_position_correction_m"]
            for _ in range(3):
                if blend_increment <= 0.0 or correction <= cap:
                    break
                applied_weight *= 0.999999 * cap / correction
                update = update_gnss(
                    before_state, fix_position, fix_velocity, position_std,
                    gc["velocity_std_mps"], 1.0e12, applied_weight,
                )
                fused = update.state
                correction = float(np.linalg.norm(fused.position_enu_m[:2] - before[:2]))
            if new_fix and update.accepted:
                accepted_new_fixes += 1
        previous_blend_weight = decision.gnss_blend_weight
        positions[local], inertial_positions[local] = fused.position_enu_m, inertial.position_enu_m
        velocities[local] = fused.velocity_enu_mps
        modes.append(decision.mode.value); weights.append(decision.gnss_blend_weight)
        nis_values.append(nis); corrections.append(correction); reasons.append(reason)

    blackout_indices = np.flatnonzero(blackout)
    b0, b1 = int(blackout_indices[0]), int(blackout_indices[-1])
    blackout_reference = reference[b0:b1 + 1] - reference[b0]
    blackout_fused = positions[b0:b1 + 1] - positions[b0]
    blackout_inertial = inertial_positions[b0:b1 + 1] - inertial_positions[b0]
    reacq = next((i for i in range(b1 + 1, len(modes)) if modes[i] == NavigationMode.TRANSITIONING.value), None)
    full_fused = trajectory_metrics(positions, reference)
    full_inertial = trajectory_metrics(inertial_positions, reference)

    # Fresh Step 5 run of the locked comparison: oracle VBOX initialization at
    # 91.6 s, then the Step 5 arbitrator disables GNSS for every blackout sample.
    # This is computed here rather than copied from the Step 4 JSON.
    oracle_global_indices = indices[b0:b1 + 1]
    oracle_outputs = outputs[b0:b1 + 1]
    oracle_state = _oracle_initial_state(
        drive, int(oracle_global_indices[0]), car_to_imu, gyro_bias, accel_bias, noise
    )
    oracle_positions = np.zeros((len(oracle_global_indices), 3))
    oracle_headings = np.zeros(len(oracle_global_indices))
    oracle_gyro_headings = np.zeros(len(oracle_global_indices))
    oracle_speeds = np.zeros(len(oracle_global_indices))
    oracle_modes = []
    oracle_arbitrator = GnssModeArbitrator(gc["transition_duration_s"])
    for local, global_index in enumerate(oracle_global_indices):
        if local:
            oracle_dt = drive.time_s[global_index] - drive.time_s[oracle_global_indices[local - 1]]
            oracle_state = _propagate_adaptive(
                oracle_state,
                drive.gyro_ypr_rps[global_index, [2, 1, 0]],
                drive.accel_device_mps2[global_index], oracle_outputs[local],
                oracle_dt,
                noise,
            )
            corrected_gyro = (
                drive.gyro_ypr_rps[global_index, [2, 1, 0]]
                - 0.02 * oracle_outputs[local, 2:5] - gyro_bias
            )
            oracle_gyro_headings[local] = (
                oracle_gyro_headings[local - 1] + np.rad2deg(corrected_gyro[2] * oracle_dt)
            ) % 360.0
        else:
            oracle_gyro_headings[local] = drive.reference_heading_deg[global_index]
        oracle_modes.append(
            oracle_arbitrator.update(float(drive.time_s[global_index]), False).mode.value
        )
        oracle_positions[local] = oracle_state.position_enu_m
        vehicle_rotation = oracle_state.rotation_imu_to_enu @ oracle_state.rotation_car_to_imu
        oracle_headings[local] = np.rad2deg(np.arctan2(
            vehicle_rotation[0, 0], vehicle_rotation[1, 0]
        )) % 360.0
        oracle_speeds[local] = np.linalg.norm(oracle_state.velocity_enu_mps[:2])
    oracle_metrics = trajectory_metrics(oracle_positions, blackout_reference)
    oracle_metrics.update({
        "session": "Vta08", "blackout_start_time_s": float(time[b0]),
        "blackout_end_time_s": float(time[b1]), "samples": len(oracle_positions),
        "method": "step5_oracle_initialized_cnn_iekf_gnss_suppressed",
    })
    step4 = json.loads(Path("results/experiments/adapter_cnn/evaluation.json").read_text())["vta08"]["step4_cnn"]
    first_nonzero = next(
        (i for i in range((reacq or 0) + 1, len(corrections)) if corrections[i] > 1.0e-9),
        None,
    )
    report = {
        "scope": "GNSS gating/arbitration/blending only; no map matching",
        "interpretation": "GNSS is suppressed throughout the blackout; blackout drift is pure inertial propagation.",
        "interval": {"start_s": float(time[0]), "end_s": float(time[-1]), "samples": len(time)},
        "blackout": {
            "start_s": exp["blackout_start_s"], "end_s": exp["blackout_end_s"],
            "fused_entry_then_no_gnss": trajectory_metrics(blackout_fused, blackout_reference),
            "same_initial_fix_dead_reckoning": trajectory_metrics(blackout_inertial, blackout_reference),
            "step5_oracle_initialized_no_gnss_rerun": oracle_metrics,
            "step4_reference_result": step4,
            "step5_minus_step4": {
                "final_position_error_m": oracle_metrics["final_position_error_m"] - step4["final_position_error_m"],
                "position_rmse_m": oracle_metrics["position_rmse_m"] - step4["position_rmse_m"],
                "drift_percentage_points": oracle_metrics["drift_percent"] - step4["drift_percent"],
            },
            "entry_horizontal_position_error_m": float(np.linalg.norm(positions[b0, :2] - reference[b0, :2])),
            "entry_horizontal_velocity_error_mps": float(np.linalg.norm(
                velocities[b0, :2] - bearing_deg_to_enu_velocity(
                    drive.reference_speed_mps[indices[b0]], drive.reference_heading_deg[indices[b0]]
                )[:2]
            )),
        },
        "extended_trace": {"gnss_fused": full_fused, "dead_reckoning_only": full_inertial},
        "reacquisition": {
            "first_transition_time_s": float(time[reacq]) if reacq is not None else None,
            "position_correction_on_first_reacquired_sample_m": float(corrections[reacq]) if reacq is not None else None,
            "zero_immediate_correction_is_intentional": True,
            "zero_immediate_correction_explanation": "The first reacquired fix starts TRANSITIONING at blend weight 0, preventing a position jump; correction begins on the next sample as the weight rises.",
            "first_nonzero_correction_time_s": float(time[first_nonzero]) if first_nonzero is not None else None,
            "first_nonzero_correction_elapsed_s": float(time[first_nonzero] - time[reacq]) if first_nonzero is not None and reacq is not None else None,
            "first_nonzero_blend_weight": float(weights[first_nonzero]) if first_nonzero is not None else None,
            "first_nonzero_position_correction_m": float(corrections[first_nonzero]) if first_nonzero is not None else None,
            "maximum_single_sample_transition_correction_m": float(max(
                (corrections[i] for i, mode in enumerate(modes) if mode == NavigationMode.TRANSITIONING.value),
                default=0.0,
            )),
            "transition_duration_s": gc["transition_duration_s"],
        },
        "gnss": {
            "accepted_new_fixes": accepted_new_fixes, "rejected_new_fix_innovations": rejected_innovations,
            "quality_reason_samples": dict(Counter(reasons)),
            "distinct_source_fixes_in_interval": int(1 + sum(
                any(values[i] != values[i - 1] for values in (
                    drive.phone_lat_deg, drive.phone_lon_deg, drive.phone_speed_raw, drive.phone_heading_deg
                )) for i in indices[1:]
            )),
        },
    }

    output_dir = Path(config["outputs"]["experiment_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["metrics"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["plot"]).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    trace = pd.DataFrame({
        "time_s": time, "reference_east_m": reference[:, 0], "reference_north_m": reference[:, 1],
        "fused_east_m": positions[:, 0], "fused_north_m": positions[:, 1],
        "dead_reckoning_east_m": inertial_positions[:, 0], "dead_reckoning_north_m": inertial_positions[:, 1],
        "mode": modes, "gnss_blend_weight": weights, "gnss_nis": nis_values,
        "gnss_position_correction_m": corrections, "gnss_quality_reason": reasons,
    })
    trace.to_csv(output_dir / "extended_trace.csv", index=False)
    pd.DataFrame({
        "time_since_blackout_s": drive.time_s[oracle_global_indices] - drive.time_s[oracle_global_indices[0]],
        "east_m": oracle_positions[:, 0], "north_m": oracle_positions[:, 1],
        "up_m": oracle_positions[:, 2], "mode": oracle_modes,
        "heading_deg": oracle_headings, "speed_mps": oracle_speeds,
        "gyro_integrated_heading_deg": oracle_gyro_headings,
    }).to_csv(output_dir / "oracle_initialized_blackout.csv", index=False)
    pd.DataFrame([
        {"interval": "locked_blackout", "method": "step5_oracle_initialized_no_gnss", **oracle_metrics},
        {"interval": "blackout", "method": "fused_entry_then_no_gnss", **report["blackout"]["fused_entry_then_no_gnss"]},
        {"interval": "blackout", "method": "dead_reckoning_only", **report["blackout"]["same_initial_fix_dead_reckoning"]},
        {"interval": "extended", "method": "gnss_fused", **full_fused},
        {"interval": "extended", "method": "dead_reckoning_only", **full_inertial},
    ]).to_csv(config["outputs"]["metrics"], index=False)

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(time, position_errors(positions, reference), label="GNSS-fused")
    axes[0].plot(time, position_errors(inertial_positions, reference), label="Dead reckoning only", alpha=0.8)
    axes[0].set_ylabel("Horizontal error (m)"); axes[0].legend(); axes[0].grid(alpha=0.25)
    mode_number = [{"DEAD_RECKONING": 0, "TRANSITIONING": 1, "GNSS_AIDED": 2}[mode] for mode in modes]
    axes[1].step(time, mode_number, where="post"); axes[1].set_yticks([0, 1, 2], ["DR", "Blend", "GNSS"])
    axes[1].set_ylabel("Mode"); axes[1].grid(alpha=0.25)
    axes[2].plot(time, weights, label="GNSS blend weight")
    axes[2].plot(time, corrections, label="position correction/sample (m)")
    axes[2].set(xlabel="Recording time (s)", ylabel="Blend / correction"); axes[2].legend(); axes[2].grid(alpha=0.25)
    for axis in axes:
        axis.axvspan(exp["blackout_start_s"], exp["blackout_end_s"], color="black", alpha=0.10)
    figure.suptitle("Vta08 Step 5: GNSS outage and smooth reacquisition")
    figure.tight_layout(); figure.savefig(config["outputs"]["plot"], dpi=240); plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step 5 GNSS arbitration and recovery")
    parser.add_argument("--config", type=Path, default=Path("configs/gnss_fusion_vta08.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
