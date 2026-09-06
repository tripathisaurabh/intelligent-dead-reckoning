from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.preprocessing.analysis import read_table


def _find_column(df: pd.DataFrame, *parts: str) -> str:
    matches = [
        str(column)
        for column in df.columns
        if all(part.lower() in str(column).lower() for part in parts)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one column containing {parts!r}; found {matches}")
    return matches[0]


def _find_axis_column(df: pd.DataFrame, sensor: str, axis: str) -> str:
    pattern = re.compile(rf"{re.escape(sensor)}\s+{re.escape(axis)}(?:\s|\()", re.IGNORECASE)
    matches = [str(column) for column in df.columns if pattern.search(str(column))]
    if len(matches) != 1:
        raise ValueError(f"Expected one {sensor} {axis}-axis column; found {matches}")
    return matches[0]


def _find_exact_column(df: pd.DataFrame, name: str) -> str:
    matches = [str(column) for column in df.columns if str(column).strip().lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exact column {name!r}; found {matches}")
    return matches[0]


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = np.flatnonzero(~np.isfinite(values))
        raise ValueError(f"Non-numeric values in {column!r} at rows {bad[:10].tolist()}")
    return values


def parse_satellite_count(values: pd.Series) -> np.ndarray:
    """Parse phone values such as ``"24 / 24"`` into satellites in range."""
    extracted = values.astype(str).str.extract(r"^\s*(\d+)", expand=False)
    counts = pd.to_numeric(extracted, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(counts).all():
        bad = np.flatnonzero(~np.isfinite(counts))
        raise ValueError(f"Invalid satellite counts at rows {bad[:10].tolist()}")
    return counts


@dataclass(frozen=True)
class SamplingStats:
    samples: int
    duration_s: float
    median_dt_s: float
    estimated_hz: float
    timestamp_resets: int
    gaps_over_limit: int
    maximum_dt_s: float


@dataclass(frozen=True)
class SynchronizedDrive:
    time_s: np.ndarray
    accel_device_mps2: np.ndarray
    gravity_device_mps2: np.ndarray
    # Source file order is semantic yaw, pitch, roll. It is deliberately retained.
    gyro_ypr_rps: np.ndarray
    orientation_ypr_deg: np.ndarray
    phone_speed_raw: np.ndarray
    phone_lat_deg: np.ndarray
    phone_lon_deg: np.ndarray
    phone_alt_m: np.ndarray
    phone_accuracy_m: np.ndarray
    phone_heading_deg: np.ndarray
    phone_satellites: np.ndarray
    reference_lat_deg: np.ndarray
    reference_lon_deg: np.ndarray
    reference_speed_mps: np.ndarray
    reference_heading_deg: np.ndarray
    reference_yaw_rate_rps: np.ndarray
    smartphone_stats: SamplingStats
    reference_stats: SamplingStats
    smartphone_missing_values: int
    reference_missing_values: int
    phone_speed_unit_assessment: dict[str, float | str]
    source_smartphone: str
    source_reference: str


def sampling_stats(time_s: np.ndarray, maximum_gap_s: float) -> SamplingStats:
    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) < 2:
        raise ValueError("At least two timestamps are required")
    dt = np.diff(time_s)
    positive = dt[dt > 0]
    if not len(positive):
        raise ValueError("No positive timestamp differences")
    median_dt = float(np.median(positive))
    return SamplingStats(
        samples=len(time_s),
        duration_s=float(time_s[-1] - time_s[0]),
        median_dt_s=median_dt,
        estimated_hz=1.0 / median_dt,
        timestamp_resets=int(np.sum(dt <= 0)),
        gaps_over_limit=int(np.sum(dt > maximum_gap_s)),
        maximum_dt_s=float(np.max(positive)),
    )


def _assess_phone_speed_units(phone: np.ndarray, reference_mps: np.ndarray) -> dict[str, float | str]:
    moving = reference_mps > 2.0
    if np.sum(moving) < 10:
        return {"assessment": "insufficient moving samples"}
    raw_mae_as_mps = float(np.mean(np.abs(phone[moving] - reference_mps[moving])))
    raw_mae_as_kmh = float(np.mean(np.abs(phone[moving] / 3.6 - reference_mps[moving])))
    assessment = "m/s despite Kmh label" if raw_mae_as_mps < raw_mae_as_kmh else "km/h"
    return {
        "assessment": assessment,
        "mae_if_raw_is_mps": raw_mae_as_mps,
        "mae_if_raw_is_kmh": raw_mae_as_kmh,
    }


def load_synchronized_drive(
    smartphone_csv: str | Path,
    reference_csv: str | Path,
    maximum_gap_s: float = 0.15,
) -> SynchronizedDrive:
    smartphone_path = Path(smartphone_csv)
    reference_path = Path(reference_csv)
    smartphone = read_table(smartphone_path)
    reference = read_table(reference_path)

    if len(smartphone) != len(reference):
        raise ValueError(
            f"Synchronized files have different row counts: {len(smartphone)} != {len(reference)}"
        )

    phone_time_ms = _numeric(smartphone, _find_column(smartphone, "time since start", "ms"))
    time_s = (phone_time_ms - phone_time_ms[0]) / 1000.0
    reference_time = _numeric(reference, _find_column(reference, "time since start of day"))
    reference_time = reference_time - reference_time[0]

    accel = np.column_stack(
        [_numeric(smartphone, _find_axis_column(smartphone, "accelerometer", axis)) for axis in "xyz"]
    )
    gravity = np.column_stack(
        [_numeric(smartphone, _find_axis_column(smartphone, "gravity", axis)) for axis in "xyz"]
    )
    gyro = np.column_stack(
        [
            _numeric(smartphone, _find_column(smartphone, "gyroscope", name))
            for name in ("yaw", "pitch", "roll")
        ]
    )
    orientation = np.column_stack(
        [
            _numeric(smartphone, _find_column(smartphone, "orientation", name))
            for name in ("yaw", "pitch", "roll")
        ]
    )

    reference_speed_mps = _numeric(reference, _find_exact_column(reference, "Velocity (km/hr)")) / 3.6
    phone_speed = _numeric(smartphone, _find_column(smartphone, "gps speed"))

    return SynchronizedDrive(
        time_s=time_s,
        accel_device_mps2=accel,
        gravity_device_mps2=gravity,
        gyro_ypr_rps=gyro,
        orientation_ypr_deg=orientation,
        phone_speed_raw=phone_speed,
        phone_lat_deg=_numeric(smartphone, _find_column(smartphone, "gps latitude")),
        phone_lon_deg=_numeric(smartphone, _find_column(smartphone, "gps longitude")),
        phone_alt_m=_numeric(smartphone, _find_column(smartphone, "gps altitude")),
        phone_accuracy_m=_numeric(smartphone, _find_column(smartphone, "gps accuracy")),
        phone_heading_deg=_numeric(smartphone, _find_column(smartphone, "gps orientation")),
        phone_satellites=parse_satellite_count(
            smartphone[_find_column(smartphone, "gps satellites in range")]
        ),
        reference_lat_deg=_numeric(reference, _find_column(reference, "latitude")),
        reference_lon_deg=_numeric(reference, _find_column(reference, "longitude")),
        reference_speed_mps=reference_speed_mps,
        reference_heading_deg=_numeric(reference, _find_column(reference, "heading")),
        reference_yaw_rate_rps=np.deg2rad(
            _numeric(reference, _find_column(reference, "yaw rate", "deg/sec"))
        ),
        smartphone_stats=sampling_stats(time_s, maximum_gap_s),
        reference_stats=sampling_stats(reference_time, maximum_gap_s),
        smartphone_missing_values=int(smartphone.isna().sum().sum()),
        reference_missing_values=int(reference.isna().sum().sum()),
        phone_speed_unit_assessment=_assess_phone_speed_units(phone_speed, reference_speed_mps),
        source_smartphone=str(smartphone_path),
        source_reference=str(reference_path),
    )
