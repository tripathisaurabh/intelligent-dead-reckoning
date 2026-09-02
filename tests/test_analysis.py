import numpy as np
import pandas as pd

from src.preprocessing.analysis import find_time_columns, sampling_summary


def test_find_time_columns_detects_time_and_date():
    df = pd.DataFrame(
        {
            " TIME SINCE START (ms)": [0, 100, 200],
            " DATE (YYYY-MO-DD HH-MI-SS_SSS)": ["a", "b", "c"],
            "ACCELEROMETER X": [0.0, 0.1, 0.2],
        }
    )

    cols = find_time_columns(df)

    assert " TIME SINCE START (ms)" in cols
    assert " DATE (YYYY-MO-DD HH-MI-SS_SSS)" in cols


def test_sampling_summary_estimates_10_hz_from_milliseconds():
    df = pd.DataFrame({"time_ms": [0, 100, 200, 300, 400]})

    result = sampling_summary(df, "time_ms")

    assert result.duplicate_timestamps == 0
    assert result.non_monotonic_steps == 0
    assert np.isclose(result.median_dt_seconds, 0.1)
    assert np.isclose(result.estimated_hz, 10.0)


def test_sampling_summary_detects_reset():
    df = pd.DataFrame({"time_ms": [0, 100, 200, 10, 110]})

    result = sampling_summary(df, "time_ms")

    assert result.non_monotonic_steps == 1
