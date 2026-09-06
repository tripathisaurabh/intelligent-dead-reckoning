import numpy as np

from src.evaluation.blackout import select_blackout_interval
from src.evaluation.metrics import trajectory_metrics


def test_metrics_known_offset():
    reference = np.column_stack((np.arange(6.0), np.zeros(6), np.zeros(6)))
    estimated = reference.copy()
    estimated[:, 1] = 3.0
    result = trajectory_metrics(estimated, reference)
    assert np.isclose(result["final_position_error_m"], 3.0)
    assert np.isclose(result["position_rmse_m"], 3.0)
    assert np.isclose(result["distance_travelled_m"], 5.0)
    assert np.isclose(result["drift_percent"], 60.0)


def test_blackout_selection_uses_earliest_qualifying_interval():
    time = np.arange(0.0, 60.1, 0.1)
    speed = np.zeros_like(time)
    speed[200:501] = 8.0
    result = select_blackout_interval(time, speed, 0, 20.0, 5.0, 0.9, 0.15, 10.0, 10.0)
    assert result.start_index == 200
    assert np.isclose(result.duration_s, 20.0)
