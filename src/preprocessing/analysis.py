from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_KEYWORDS = ("time", "timestamp", "date")


@dataclass
class SamplingSummary:
    column: str
    samples: int
    duplicate_timestamps: int
    non_monotonic_steps: int
    median_dt_seconds: float | None
    estimated_hz: float | None
    max_gap_seconds: float | None


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read an IO-VNBD CSV/TXT file while tolerating legacy text encodings."""
    path = Path(path)

    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                nrows=nrows,
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        getattr(last_error, "encoding", "unknown"),
        getattr(last_error, "object", b""),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 1),
        f"Could not decode {path} using {', '.join(encodings)}",
    )


def find_data_files(root: str | Path, extensions: Iterable[str] = (".csv", ".txt")) -> list[Path]:
    root = Path(root)
    allowed = {ext.lower() for ext in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def column_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "rows": len(series),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 3),
                "numeric_values": int(numeric.notna().sum()),
                "min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "max": float(numeric.max()) if numeric.notna().any() else np.nan,
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "std": float(numeric.std()) if numeric.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def find_time_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if any(keyword in str(col).lower() for keyword in TIME_KEYWORDS)
    ]


def _numeric_time_to_seconds(values: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 2:
        return x

    diffs = np.diff(np.sort(x))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return x

    median_diff = float(np.median(positive))

    # Heuristic only for inspection. Final units must be confirmed from dataset docs.
    if median_diff > 1e6:       # likely nanoseconds
        return x / 1e9
    if median_diff > 1e3:       # likely microseconds
        return x / 1e6
    if median_diff > 1:         # often milliseconds for sensor logs
        return x / 1e3
    return x


def sampling_summary(df: pd.DataFrame, time_column: str) -> SamplingSummary:
    raw = df[time_column]

    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().sum() >= max(2, int(0.8 * len(raw))):
        t = _numeric_time_to_seconds(numeric)
    else:
        parsed = pd.to_datetime(raw, errors="coerce")
        parsed = parsed.dropna()
        if len(parsed) < 2:
            return SamplingSummary(time_column, len(raw), 0, 0, None, None, None)
        t = parsed.astype("int64").to_numpy(dtype=float) / 1e9

    if t.size < 2:
        return SamplingSummary(time_column, len(raw), 0, 0, None, None, None)

    dt = np.diff(t)
    positive_dt = dt[dt > 0]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else None
    hz = (1.0 / median_dt) if median_dt and median_dt > 0 else None

    return SamplingSummary(
        column=time_column,
        samples=int(t.size),
        duplicate_timestamps=int(np.sum(dt == 0)),
        non_monotonic_steps=int(np.sum(dt < 0)),
        median_dt_seconds=median_dt,
        estimated_hz=hz,
        max_gap_seconds=float(np.max(positive_dt)) if positive_dt.size else None,
    )


def keyword_columns(df: pd.DataFrame, keywords: Iterable[str]) -> list[str]:
    keys = tuple(k.lower() for k in keywords)
    return [col for col in df.columns if any(k in str(col).lower() for k in keys)]
