from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.run_adapter_vta08 import _load_model, _model_outputs, _noise, _stationary_bias
from src.evaluation.blackout import select_blackout_interval
from src.evaluation.metrics import trajectory_metrics
from src.ins.adaptive_iekf import run_adaptive_iekf
from src.ins.iekf import IEKFInput
from src.mapmatch.hmm_matcher import HMMMapMatcher, OSMRoadNetwork
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu
from train import mounting_rotation


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp, inputs = config["experiment"], config["inputs"]
    drive = load_synchronized_drive(exp["smartphone_csv"], exp["reference_csv"])
    gyro_bias, accel_bias, calibration_end = _stationary_bias(drive)
    blackout = select_blackout_interval(
        drive.time_s, drive.reference_speed_mps, earliest_index=calibration_end,
        duration_s=20.0, minimum_speed_mps=5.0, required_moving_fraction=0.90,
        maximum_gap_s=0.15, context_before_s=3.2, context_after_s=0.0,
    )
    if not np.isclose(blackout.start_time_s, exp["expected_blackout_start_s"]) or not np.isclose(
        blackout.end_time_s, exp["expected_blackout_end_s"]
    ):
        raise RuntimeError("Vta16 blackout no longer matches the locked Step 4 validation interval")
    start, end = blackout.start_index, blackout.end_index_inclusive + 1
    indices = np.arange(start, end)
    selection = slice(start, end)

    adapter_config = yaml.safe_load(Path(inputs["adapter_config"]).read_text())
    model = _load_model(Path(inputs["model"]))
    outputs = _model_outputs(
        model, drive.gyro_ypr_rps[:, [2, 1, 0]], drive.accel_device_mps2,
        gyro_bias, accel_bias, indices, adapter_config["model"]["context_samples"],
    )
    noise = _noise(yaml.safe_load(Path(inputs["iekf_config"]).read_text())["iekf"])
    car_rotation = mounting_rotation(adapter_config["mounting"])
    time = drive.time_s[selection] - drive.time_s[start]
    result = run_adaptive_iekf(
        IEKFInput(
            time_s=time, gyro_body_rps=drive.gyro_ypr_rps[selection][:, [2, 1, 0]],
            specific_force_body_mps2=drive.accel_device_mps2[selection],
            initial_speed_mps=float(drive.reference_speed_mps[start]),
            initial_bearing_deg=float(drive.reference_heading_deg[start]),
            initial_gyro_bias_body_rps=gyro_bias, initial_accel_bias_body_mps2=accel_bias,
            initial_rotation_car_to_imu=car_rotation,
        ),
        outputs, noise,
    )
    reference = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )
    gyro_heading = np.zeros(len(indices))
    gyro_heading[0] = drive.reference_heading_deg[start]
    corrected_gyro = drive.gyro_ypr_rps[selection][:, [2, 1, 0]] - 0.02 * outputs[:, 2:5] - gyro_bias
    gyro_heading[1:] = (
        gyro_heading[0] + np.rad2deg(np.cumsum(corrected_gyro[1:, 2] * np.diff(time)))
    ) % 360.0

    matcher_settings = yaml.safe_load(Path(inputs["matcher_config"]).read_text())["matcher"]
    network = OSMRoadNetwork.from_xml(
        inputs["osm_xml"], origin_lat_deg=float(drive.reference_lat_deg[start]),
        origin_lon_deg=float(drive.reference_lon_deg[start]),
    )
    matched = HMMMapMatcher(network, **matcher_settings).match(result.position_enu_m, gyro_heading)
    before = trajectory_metrics(result.position_enu_m, reference)
    after = trajectory_metrics(matched.position_enu_m, reference)
    reference_arc = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1))
    ]
    closest_endpoint_reference_index = int(np.argmin(
        np.linalg.norm(reference[:, :2] - matched.position_enu_m[-1, :2], axis=1)
    ))
    recorded_step4 = json.loads(Path(inputs["step4_evaluation"]).read_text())[
        "held_out_validation_session"
    ]["cnn"]
    report = {
        "validation_role": "Vta16 was excluded from gradient updates but used for Step 4 epoch selection",
        "parameter_policy": "Exact Vta08 Step 6 matcher parameters reused without Vta16 tuning",
        "session": "Vta16", "blackout_start_s": blackout.start_time_s,
        "blackout_end_s": blackout.end_time_s, "samples": len(indices),
        "road_network": {
            "source": "OpenStreetMap via Overpass API", "license": "ODbL",
            "drivable_segments": len(network.segments), "graph_nodes": network.graph.number_of_nodes(),
            "graph_edges": network.graph.number_of_edges(),
        },
        "matcher": matcher_settings,
        "before_map_matching": before, "after_map_matching": after,
        "change": {
            "final_position_error_m": after["final_position_error_m"] - before["final_position_error_m"],
            "position_rmse_m": after["position_rmse_m"] - before["position_rmse_m"],
            "drift_percentage_points": after["drift_percent"] - before["drift_percent"],
        },
        "step4_reproduction_difference": {
            "final_position_error_m": before["final_position_error_m"] - recorded_step4["final_position_error_m"],
            "position_rmse_m": before["position_rmse_m"] - recorded_step4["position_rmse_m"],
        },
        "candidate_diagnostics": {
            "mean_snap_distance_m": float(np.mean(matched.candidate_distance_m)),
            "maximum_snap_distance_m": float(np.max(matched.candidate_distance_m)),
            "mean_heading_error_deg": float(np.mean(matched.heading_error_deg)),
            "unique_segments": int(len(np.unique(matched.segment_id))),
            "unique_ways": int(len(np.unique(matched.way_id))),
        },
        "endpoint_progress_diagnostic": {
            "closest_reference_time_since_blackout_s": float(
                time[closest_endpoint_reference_index]
            ),
            "distance_to_that_reference_point_m": float(np.linalg.norm(
                reference[closest_endpoint_reference_index, :2]
                - matched.position_enu_m[-1, :2]
            )),
            "reference_distance_remaining_after_that_point_m": float(
                reference_arc[-1] - reference_arc[closest_endpoint_reference_index]
            ),
            "interpretation": "The HMM selected the driven road but placed the endpoint much too far back along it; road constraints do not observe longitudinal progress.",
        },
    }

    output_dir = Path(config["outputs"]["experiment_dir"])
    metrics_path, plot_path = Path(config["outputs"]["metrics"]), Path(config["outputs"]["plot"])
    output_dir.mkdir(parents=True, exist_ok=True); metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({
        "time_since_blackout_s": time, "raw_east_m": result.position_enu_m[:, 0],
        "raw_north_m": result.position_enu_m[:, 1], "matched_east_m": matched.position_enu_m[:, 0],
        "matched_north_m": matched.position_enu_m[:, 1], "reference_east_m": reference[:, 0],
        "reference_north_m": reference[:, 1], "gyro_heading_deg": gyro_heading,
        "segment_id": matched.segment_id, "osm_way_id": matched.way_id,
        "snap_distance_m": matched.candidate_distance_m,
    }).to_csv(output_dir / "matched_trajectory.csv", index=False)
    pd.DataFrame([
        {"method": "step4_cnn_before_map_matching", **before},
        {"method": "step6_osm_hmm_map_matching", **after},
    ]).to_csv(metrics_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 9))
    for segment in network.segments:
        points = np.vstack((segment.start_enu_m, segment.end_enu_m))
        if np.min(np.linalg.norm(points[:, :2], axis=1)) < 250.0:
            axis.plot(points[:, 0], points[:, 1], color="0.78", linewidth=1.0, zorder=1)
    axis.plot(reference[:, 0], reference[:, 1], color="black", linewidth=2.4, label="VBOX reference", zorder=4)
    axis.plot(result.position_enu_m[:, 0], result.position_enu_m[:, 1], color="#d95f02", label="Step 4 CNN-IEKF", zorder=2)
    axis.plot(matched.position_enu_m[:, 0], matched.position_enu_m[:, 1], color="#1b9e77", linewidth=2.0, label="Step 6 OSM HMM", zorder=3)
    axis.scatter([0], [0], color="black", s=30, label="Blackout start", zorder=5)
    axis.set(title="Vta16 held-out blackout: unchanged OSM HMM pipeline", xlabel="East (m)", ylabel="North (m)")
    axis.axis("equal"); axis.grid(alpha=0.2); axis.legend(); figure.tight_layout()
    figure.savefig(plot_path, dpi=240); plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 6 on the Vta16 Step 4 blackout")
    parser.add_argument("--config", type=Path, default=Path("configs/map_matching_vta16.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
