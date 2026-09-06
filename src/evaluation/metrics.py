from __future__ import annotations

import numpy as np


def trajectory_metrics(estimated_enu_m: np.ndarray, reference_enu_m: np.ndarray) -> dict[str, float]:
    estimated = np.asarray(estimated_enu_m, dtype=float)
    reference = np.asarray(reference_enu_m, dtype=float)
    if estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] < 2:
        raise ValueError("Estimated and reference trajectories must have equal shape (N, >=2)")
    errors = np.linalg.norm(estimated[:, :2] - reference[:, :2], axis=1)
    steps = np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1)
    distance = float(np.sum(steps))
    final_error = float(errors[-1])
    return {
        "final_position_error_m": final_error,
        "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mean_position_error_m": float(np.mean(errors)),
        "maximum_position_error_m": float(np.max(errors)),
        "distance_travelled_m": distance,
        "drift_percent": float(100.0 * final_error / distance) if distance > 0 else float("nan"),
    }


def position_errors(estimated_enu_m: np.ndarray, reference_enu_m: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(estimated_enu_m)[:, :2] - np.asarray(reference_enu_m)[:, :2], axis=1
    )
