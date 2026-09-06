from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import trajectory_metrics
from src.mapmatch.hmm_matcher import HMMMapMatcher, OSMRoadNetwork
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp, inputs, settings = config["experiment"], config["inputs"], config["matcher"]
    drive = load_synchronized_drive(exp["smartphone_csv"], exp["reference_csv"])
    start = int(np.searchsorted(drive.time_s, exp["blackout_start_s"], side="left"))
    end = int(np.searchsorted(drive.time_s, exp["blackout_end_s"], side="right"))
    selection = slice(start, end)
    reference = geodetic_to_local_enu(
        drive.reference_lat_deg[selection], drive.reference_lon_deg[selection]
    )
    trajectory_frame = pd.read_csv(inputs["step5_trajectory"])
    raw = trajectory_frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    if raw.shape != reference.shape:
        raise ValueError(f"Step 5 trajectory shape {raw.shape} != reference {reference.shape}")

    network = OSMRoadNetwork.from_xml(
        inputs["osm_xml"], origin_lat_deg=float(drive.reference_lat_deg[start]),
        origin_lon_deg=float(drive.reference_lon_deg[start]),
    )
    matcher = HMMMapMatcher(network, **settings)
    filter_heading = trajectory_frame["gyro_integrated_heading_deg"].to_numpy(dtype=float)
    matched = matcher.match(raw, filter_heading)
    before = trajectory_metrics(raw, reference)
    after = trajectory_metrics(matched.position_enu_m, reference)
    step5 = json.loads(Path(inputs["step5_evaluation"]).read_text())["blackout"][
        "step5_oracle_initialized_no_gnss_rerun"
    ]
    report = {
        "scope": "downstream OSM HMM/Viterbi map matching; no feedback into IEKF state",
        "session": "Vta08", "blackout_start_s": exp["blackout_start_s"],
        "blackout_end_s": exp["blackout_end_s"], "samples": len(raw),
        "road_network": {
            "source": "OpenStreetMap via Overpass API", "license": "ODbL",
            "xml": inputs["osm_xml"], "drivable_segments": len(network.segments),
            "graph_nodes": network.graph.number_of_nodes(), "graph_edges": network.graph.number_of_edges(),
        },
        "matcher": settings,
        "heading_source": "Step 5 bias/CNN-corrected gyro integration from the initial heading; no VBOX after initialization",
        "before_map_matching": before,
        "after_map_matching": after,
        "change": {
            "final_position_error_m": after["final_position_error_m"] - before["final_position_error_m"],
            "position_rmse_m": after["position_rmse_m"] - before["position_rmse_m"],
            "drift_percentage_points": after["drift_percent"] - before["drift_percent"],
        },
        "step5_input_consistency": {
            "endpoint_difference_m": before["final_position_error_m"] - step5["final_position_error_m"],
            "rmse_difference_m": before["position_rmse_m"] - step5["position_rmse_m"],
        },
        "candidate_diagnostics": {
            "mean_snap_distance_m": float(np.mean(matched.candidate_distance_m)),
            "maximum_snap_distance_m": float(np.max(matched.candidate_distance_m)),
            "mean_heading_error_deg": float(np.mean(matched.heading_error_deg)),
            "unique_segments": int(len(np.unique(matched.segment_id))),
            "unique_ways": int(len(np.unique(matched.way_id))),
        },
    }

    output_dir = Path(config["outputs"]["experiment_dir"])
    metrics_path, plot_path = Path(config["outputs"]["metrics"]), Path(config["outputs"]["plot"])
    output_dir.mkdir(parents=True, exist_ok=True); metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({
        "time_since_blackout_s": trajectory_frame["time_since_blackout_s"],
        "raw_east_m": raw[:, 0], "raw_north_m": raw[:, 1],
        "matched_east_m": matched.position_enu_m[:, 0], "matched_north_m": matched.position_enu_m[:, 1],
        "reference_east_m": reference[:, 0], "reference_north_m": reference[:, 1],
        "segment_id": matched.segment_id, "osm_way_id": matched.way_id,
        "snap_distance_m": matched.candidate_distance_m,
        "heading_error_deg": matched.heading_error_deg,
        "filter_heading_deg": filter_heading,
    }).to_csv(output_dir / "matched_trajectory.csv", index=False)
    pd.DataFrame([
        {"method": "step5_before_map_matching", **before},
        {"method": "step6_osm_hmm_map_matching", **after},
    ]).to_csv(metrics_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 9))
    for segment in network.segments:
        points = np.vstack((segment.start_enu_m, segment.end_enu_m))
        if np.min(np.linalg.norm(points[:, :2], axis=1)) < 250.0:
            axis.plot(points[:, 0], points[:, 1], color="0.78", linewidth=1.0, zorder=1)
    axis.plot(reference[:, 0], reference[:, 1], color="black", linewidth=2.4, label="VBOX reference", zorder=4)
    axis.plot(raw[:, 0], raw[:, 1], color="#d95f02", linewidth=1.7, label="Step 5 filter output", zorder=2)
    axis.plot(
        matched.position_enu_m[:, 0], matched.position_enu_m[:, 1], color="#1b9e77",
        linewidth=2.0, label="Step 6 OSM HMM", zorder=3,
    )
    axis.scatter([0.0], [0.0], color="black", s=30, label="Blackout start", zorder=5)
    axis.set(title="Vta08 blackout: OSM fixed-lag HMM map matching", xlabel="East (m)", ylabel="North (m)")
    axis.axis("equal"); axis.grid(alpha=0.2); axis.legend(); figure.tight_layout()
    figure.savefig(plot_path, dpi=240); plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 6 OSM HMM map matching")
    parser.add_argument("--config", type=Path, default=Path("configs/map_matching_vta08.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
