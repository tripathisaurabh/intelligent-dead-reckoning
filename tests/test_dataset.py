import numpy as np

import pandas as pd

from src.preprocessing.dataset import parse_satellite_count, sampling_stats


def test_sampling_stats_uses_measured_dt():
    result = sampling_stats(np.array([0.0, 0.1, 0.201, 0.301]), maximum_gap_s=0.15)
    assert np.isclose(result.median_dt_s, 0.1)
    assert np.isclose(result.estimated_hz, 10.0)
    assert result.timestamp_resets == 0
    assert result.gaps_over_limit == 0


def test_sampling_stats_reports_reset_and_gap():
    result = sampling_stats(np.array([0.0, 0.1, 0.4, 0.2]), maximum_gap_s=0.15)
    assert result.timestamp_resets == 1
    assert result.gaps_over_limit == 1


def test_phone_satellite_count_parser_uses_in_range_value():
    result = parse_satellite_count(pd.Series(["24 / 24", " 9 / 17", "12"]))
    assert np.allclose(result, [24.0, 9.0, 12.0])
