from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfilt


def hampel_mask(series: pd.Series, window_size: int = 11, n_sigma: float = 3.0) -> pd.Series:
    """Return True where samples look like isolated outliers using a Hampel/MAD rule."""
    x = pd.to_numeric(series, errors="coerce")
    rolling_median = x.rolling(window_size, center=True, min_periods=1).median()
    absolute_deviation = (x - rolling_median).abs()
    mad = absolute_deviation.rolling(window_size, center=True, min_periods=1).median()
    robust_sigma = 1.4826 * mad
    threshold = n_sigma * robust_sigma
    return absolute_deviation > threshold


def butterworth_lowpass_offline(
    values: np.ndarray,
    sampling_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase low-pass filter for offline dataset analysis.

    Do not use this implementation as-is for real-time Android inference because
    filtfilt uses future samples.
    """
    values = np.asarray(values, dtype=float)
    if sampling_hz <= 0:
        raise ValueError("sampling_hz must be positive")
    if not 0 < cutoff_hz < sampling_hz / 2:
        raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency")

    sos = butter(order, cutoff_hz, btype="low", fs=sampling_hz, output="sos")
    # sosfiltfilt is ideal, but filtfilt keeps compatibility with older scipy installs.
    b, a = butter(order, cutoff_hz, btype="low", fs=sampling_hz)
    return filtfilt(b, a, values)


def butterworth_lowpass_causal(
    values: np.ndarray,
    sampling_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Causal low-pass version closer to what can run online on a phone."""
    values = np.asarray(values, dtype=float)
    sos = butter(order, cutoff_hz, btype="low", fs=sampling_hz, output="sos")
    return sosfilt(sos, values)
