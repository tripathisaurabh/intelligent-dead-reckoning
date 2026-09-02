import numpy as np
import pandas as pd
import pytest

from src.preprocessing.filters import (
    butterworth_lowpass_causal,
    butterworth_lowpass_offline,
    hampel_mask,
)


def test_hampel_mask_flags_isolated_spike():
    x = pd.Series([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    mask = hampel_mask(x, window_size=5, n_sigma=3.0)
    assert bool(mask.iloc[3])


def test_offline_butterworth_preserves_length():
    t = np.arange(0, 5, 0.1)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 4.0 * t)
    filtered = butterworth_lowpass_offline(signal, sampling_hz=10.0, cutoff_hz=2.0)
    assert len(filtered) == len(signal)
    assert np.isfinite(filtered).all()


def test_causal_butterworth_preserves_length():
    signal = np.linspace(0.0, 1.0, 100)
    filtered = butterworth_lowpass_causal(signal, sampling_hz=10.0, cutoff_hz=2.0)
    assert len(filtered) == len(signal)
    assert np.isfinite(filtered).all()


def test_offline_butterworth_rejects_invalid_cutoff():
    signal = np.arange(20, dtype=float)
    with pytest.raises(ValueError):
        butterworth_lowpass_offline(signal, sampling_hz=10.0, cutoff_hz=5.0)
