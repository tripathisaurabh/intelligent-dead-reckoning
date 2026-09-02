from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.analysis import read_table


def find_time_column(df: pd.DataFrame) -> str:
    preferred = [
        " TIME SINCE START (ms)",
        "TIME SINCE START (ms)",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    candidates = [c for c in df.columns if "time since start" in str(c).lower()]
    if not candidates:
        raise ValueError("Could not find a TIME SINCE START column.")
    return candidates[0]


def split_sessions(time_ms: np.ndarray) -> list[tuple[int, int]]:
    """Split at every non-monotonic timestamp step (time <= previous time)."""
    if len(time_ms) == 0:
        return []

    dt = np.diff(time_ms)
    reset_indices = np.where(dt <= 0)[0] + 1

    starts = np.concatenate(([0], reset_indices))
    ends = np.concatenate((reset_indices, [len(time_ms)]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def session_summary(time_ms: np.ndarray, start: int, end: int, session_id: int) -> dict:
    t = time_ms[start:end]
    dt_ms = np.diff(t)
    positive_dt = dt_ms[dt_ms > 0]

    median_dt_ms = float(np.median(positive_dt)) if len(positive_dt) else None
    estimated_hz = 1000.0 / median_dt_ms if median_dt_ms and median_dt_ms > 0 else None
    duration_s = float((t[-1] - t[0]) / 1000.0) if len(t) > 1 else 0.0

    return {
        "session_id": session_id,
        "start_row_zero_based": start,
        "end_row_zero_based_inclusive": end - 1,
        "rows": int(end - start),
        "start_time_ms": float(t[0]),
        "end_time_ms": float(t[-1]),
        "duration_seconds": duration_s,
        "median_dt_ms": median_dt_ms,
        "estimated_hz": estimated_hz,
        "duplicate_steps": int(np.sum(dt_ms == 0)),
        "negative_steps": int(np.sum(dt_ms < 0)),
        "max_positive_gap_seconds": float(np.max(positive_dt) / 1000.0) if len(positive_dt) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse IO-VNBD timestamp resets and gaps.")
    parser.add_argument("csv", type=Path, help="Path to one IO-VNBD CSV file")
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.25,
        help="Report positive gaps larger than this many seconds (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/timestamp_analysis"),
        help="Output directory",
    )
    args = parser.parse_args()

    df = read_table(args.csv)
    time_col = find_time_column(df)
    time_ms = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

    if np.isnan(time_ms).any():
        bad = np.where(np.isnan(time_ms))[0]
        raise ValueError(f"Time column contains non-numeric values at rows: {bad[:20].tolist()}")

    dt_ms = np.diff(time_ms)
    sessions = split_sessions(time_ms)

    reset_rows = np.where(dt_ms <= 0)[0] + 1
    gap_rows = np.where(dt_ms > args.gap_threshold * 1000.0)[0] + 1

    summaries = [
        session_summary(time_ms, start, end, i + 1)
        for i, (start, end) in enumerate(sessions)
    ]

    gap_records = []
    for row in gap_rows:
        gap_records.append(
            {
                "row_zero_based": int(row),
                "previous_row_zero_based": int(row - 1),
                "previous_time_ms": float(time_ms[row - 1]),
                "current_time_ms": float(time_ms[row]),
                "gap_seconds": float((time_ms[row] - time_ms[row - 1]) / 1000.0),
            }
        )

    reset_records = []
    for row in reset_rows:
        reset_records.append(
            {
                "row_zero_based": int(row),
                "previous_row_zero_based": int(row - 1),
                "previous_time_ms": float(time_ms[row - 1]),
                "current_time_ms": float(time_ms[row]),
                "delta_seconds": float((time_ms[row] - time_ms[row - 1]) / 1000.0),
            }
        )

    print(f"File: {args.csv}")
    print(f"Rows: {len(df):,}")
    print(f"Time column: {time_col!r}")
    print(f"Sessions detected: {len(sessions)}")
    print()

    for s in summaries:
        print(f"SESSION {s['session_id']}")
        print(f"  rows: {s['rows']:,}")
        print(f"  row range: {s['start_row_zero_based']} -> {s['end_row_zero_based_inclusive']}")
        print(f"  duration: {s['duration_seconds']:.3f} s ({s['duration_seconds']/60:.2f} min)")
        print(f"  median dt: {s['median_dt_ms']:.3f} ms" if s['median_dt_ms'] is not None else "  median dt: n/a")
        print(f"  estimated rate: {s['estimated_hz']:.6f} Hz" if s['estimated_hz'] is not None else "  estimated rate: n/a")
        print(f"  max positive gap: {s['max_positive_gap_seconds']:.3f} s" if s['max_positive_gap_seconds'] is not None else "  max positive gap: n/a")
        print()

    print(f"Timestamp resets / non-monotonic steps: {len(reset_records)}")
    for r in reset_records:
        print(
            f"  row {r['row_zero_based']}: "
            f"{r['previous_time_ms']} ms -> {r['current_time_ms']} ms "
            f"(delta {r['delta_seconds']:.3f} s)"
        )

    print()
    print(f"Large positive gaps > {args.gap_threshold:.3f} s: {len(gap_records)}")
    for g in gap_records[:50]:
        print(
            f"  row {g['row_zero_based']}: "
            f"{g['previous_time_ms']} ms -> {g['current_time_ms']} ms "
            f"(gap {g['gap_seconds']:.3f} s)"
        )
    if len(gap_records) > 50:
        print(f"  ... {len(gap_records) - 50} more gaps saved to file")

    out_dir = args.output / args.csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summaries).to_csv(out_dir / "session_summary.csv", index=False)
    pd.DataFrame(reset_records).to_csv(out_dir / "timestamp_resets.csv", index=False)
    pd.DataFrame(gap_records).to_csv(out_dir / "large_gaps.csv", index=False)

    with (out_dir / "timestamp_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "file": str(args.csv),
                "rows": len(df),
                "time_column": time_col,
                "gap_threshold_seconds": args.gap_threshold,
                "sessions": summaries,
                "timestamp_resets": reset_records,
                "large_gaps": gap_records,
            },
            f,
            indent=2,
        )

    print()
    print(f"Saved output to: {out_dir}")


if __name__ == "__main__":
    main()
